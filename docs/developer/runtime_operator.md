# Runtime Operator

The nn2FPGA runtime operator is the ONNX Runtime custom operator that executes a generated FPGA accelerator from an ONNX model.

During compilation, nn2FPGA replaces an FPGA-supported subgraph with a custom ONNX node, named `nn2fpgaPartition`. The generated ONNX model remains executable by ONNX Runtime, but the partition node is implemented by the external shared library `libnn2fpga_customop.so`.

## Generated Artifacts

A deployment build normally contains:

* `model.onnx`: ONNX model containing the `nn2fpgaPartition` custom node.
* `libnn2fpga_customop.so`: ONNX Runtime custom op library implementing the partition node.
* `pynq_program.py`: helper used by the runtime to program the FPGA bitstream.
* Embedded accelerator package: JSON stored as a custom op attribute, including the bitstream, HWH metadata, port metadata, static inputs, and runtime configuration.

The custom op library contains generated C++ code specialized for the accelerator specification. The central runtime files are:

* `nn2fpga/hw/operator_runtime/nn2FPGA_kernel.hpp`: ONNX Runtime custom op and FPGA runner.
* `nn2fpga/hw/operator_runtime/nn2FPGA_spec.hpp`: generated accelerator specification types.
* `nn2fpga/hw/operator_runtime/xrt_dma.h`: AXI DMA helper classes.
* `nn2fpga/hw/operator_runtime/xrt_mmio.hpp`: AXI-Lite mapping helpers.
* `nn2fpga/hw/operator_runtime/nn2FPGA_allocator.hpp`: optional XRT-backed ORT allocator support.

## Inference Modes

There are two implemented runtime modes for a generated model containing an `nn2fpgaPartition` node.

The first mode is the standard Python ONNX Runtime custom-op flow. It loads `libnn2fpga_customop.so` and executes the FPGA partition through the current copy-based data path.

The second mode is an allocator-enabled C++ ONNX Runtime flow. It creates the `OrtEnv` in C/C++, registers the nn2FPGA XRT-backed CPU allocator, then loads the same custom op library. In this mode, ONNX Runtime can allocate intermediate CPU tensors from XRT BO-backed memory, and the custom op can identify those pointers through the allocator lookup API.

Both modes execute the same ONNX model and the same custom op. The difference is how the ONNX Runtime session is created and how ORT tensor memory is allocated.

## Standard Python Session

The standard Python flow registers only the custom op library:

```python
import os
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library(os.path.abspath("libnn2fpga_customop.so"))
session = ort.InferenceSession("model.onnx", so)
```

In this mode, Python ONNX Runtime uses its normal CPU allocators. The custom op receives ordinary CPU tensor pointers. The runtime therefore uses the internal XRT BO staging path described in [Standard Copy-Based Inference](#standard-copy-based-inference).

## Allocator-Enabled C++ Session

The allocator-enabled flow must be created from C or C++, because the Python ONNX Runtime API does not expose registration for arbitrary user-provided `OrtAllocator` callbacks.

The implemented C++ session setup is:

```cpp
const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

OrtEnv *env = nullptr;
api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "nn2fpga", &env);

nn2fpga_register_xrt_cpu_allocator(env, api);

OrtSessionOptions *options = nullptr;
api->CreateSessionOptions(&options);
api->AddSessionConfigEntry(options, "session.use_env_allocators", "1");
api->DisableMemPattern(options);
api->RegisterCustomOpsLibrary(options, "libnn2fpga_customop.so", &handle);

OrtSession *session = nullptr;
api->CreateSession(env, "model.onnx", options, &session);
```

In this mode, ONNX Runtime can allocate CPU tensors through the nn2FPGA allocator. Those tensors are CPU-accessible pointers backed by XRT BOs. The custom op can query those pointers with `nn2fpga_allocator_lookup` and recover the BO device address.

The custom op advertises `CPUExecutionProvider` as its execution provider type in both modes. ONNX Runtime schedules the partition node like a CPU custom op, but the node implementation programs and drives the FPGA internally through XRT and memory-mapped DMA registers.

## Initialization Flow

The runtime performs FPGA initialization once per process through `FpgaRunnerT<Spec>::ensure_loaded(...)`.

At session creation time, the custom op kernel constructor receives the `accelerator_package` attribute from the ONNX node. The runtime then:

1. Parses the accelerator package JSON.
2. Decodes the embedded bitstream and HWH data.
3. Writes them to the local `Overlay/` directory.
4. Programs the FPGA through `pynq_program.py`.
5. Maps the AXI-Lite control window.
6. Sets the FPGA clock.
7. Allocates XRT BOs for dynamic inputs, dynamic outputs, and internal buffers.
8. Builds DMA helper objects for each stream port.
9. Uploads static inputs, such as weights or constants, if they are present in the package.

Initialization is guarded by `std::call_once`, so multiple ONNX Runtime kernel instances share the same loaded FPGA runner within the process.

## Inference Flow

At inference time, ONNX Runtime calls `Nn2FpgaKernelT<Spec>::Compute(...)` for the partition node.

`Compute(...)` validates the ORT tensor types and shapes against the generated `Spec`, extracts input tensor pointers, creates output tensors, and calls:

```cpp
FpgaRunnerT<Spec>::instance().run(in_ptrs, out_ptrs, batch);
```

`FpgaRunnerT::run(...)` is serialized by a mutex because it programs shared FPGA and DMA resources.

## Standard Copy-Based Inference

The current production data path is copy-based. This is the compatibility path and remains the fallback even when allocator diagnostics are enabled.

For each dynamic input:

1. Copy from the ORT input tensor into an internal XRT input BO.
2. Sync the input BO to the device.
3. Start the MM2S DMA transfer from that BO.

For each dynamic output:

1. Start the S2MM DMA transfer into an internal XRT output BO.
2. Wait for S2MM completion.
3. Sync the output BO from the device.
4. Copy from the internal output BO into the ORT output tensor.

Conceptually:

```text
ORT input tensor
  -> memcpy
  -> internal input XRT BO
  -> MM2S DMA
  -> FPGA accelerator
  -> S2MM DMA
  -> internal output XRT BO
  -> memcpy
  -> ORT output tensor
```

This path works with normal Python ONNX Runtime usage and does not require a custom allocator.

## Allocator-Backed Tensor Diagnostics

The allocator-enabled C++ session changes how ORT can allocate tensors. The runtime operator uses allocator-backed partition inputs directly as MM2S DMA sources and keeps the copy-based output path.

The implemented allocator behavior is:

1. The C++ runner registers `nn2fpga_register_xrt_cpu_allocator(...)` on the `OrtEnv`.
2. The session enables environment allocators with `session.use_env_allocators=1`.
3. ORT allocates eligible CPU tensors through the nn2FPGA allocator.
4. The allocator creates an XRT BO, maps it to CPU memory, returns the mapped pointer to ORT, and records the pointer range and device address.
5. The custom op uses `dlsym(RTLD_DEFAULT, "nn2fpga_allocator_lookup")` to discover the optional lookup function at runtime.
6. During `Compute(...)`, the custom op logs whether each real partition input and output pointer is allocator-backed when `NN2FPGA_ZERO_COPY_LOG=1` is set.
7. Allocator-backed inputs are synced to device and passed directly to MM2S by device address.
8. Outputs still use the internal S2MM output BO and are copied back to the ORT output tensor.

Conceptually, the implemented allocator-enabled diagnostic flow is:

```text
allocator-enabled C++ ORT session
  -> registers nn2FPGA XRT allocator
  -> loads libnn2fpga_customop.so
  -> ORT tensor may be backed by XRT BO
  -> nn2fpgaPartition receives normal CPU pointer
  -> allocator lookup reports registered=1 and device_addr=0x...
  -> input MM2S reads directly from allocator BO device_addr
  -> output S2MM still writes internal BO and copies back
```

This mode removes the partition input staging copy when allocator lookup succeeds. If lookup or sync fails, the runtime falls back to the standard input copy path.

## Static Inputs

Some input ports are marked as `PortMode::StaticInit`. These ports are not exposed as dynamic ORT inputs.

During initialization, the runtime reads static values from the accelerator package `input_map`, copies them into the corresponding internal XRT BO, syncs the BO to the device, and performs the required DMA upload once.

Static inputs are skipped during normal `Compute(...)` calls.

## DMA Helpers

The runtime currently uses two DMA helper classes from `xrt_dma.h`.

`Mm2sSimple` handles memory-to-stream input transfers. It owns a source XRT BO reference and programs the MM2S source address and transfer length.

`S2mmSG` handles stream-to-memory output transfers using AXI DMA scatter-gather descriptors. Each descriptor writes one batch element into the output BO. The descriptor BO is synced before starting the transfer and synced back while checking completion.

Internal accelerator buffers used by HLS `m_axi` pointer ports are allocated as XRT BOs and their device addresses are written into the control register window.

## Allocator Lookup API

nn2FPGA also includes an optional XRT-backed ONNX Runtime CPU allocator. The allocator returns CPU-accessible pointers to ORT, but each allocation is backed by an XRT BO and has a valid FPGA device address.

The allocator records metadata for each allocation:

```text
host pointer range -> allocation size, XRT BO, device address
```

The runtime operator can discover this allocator at runtime with:

```cpp
dlsym(RTLD_DEFAULT, "nn2fpga_allocator_lookup")
```

This is intentionally a runtime lookup, not a mandatory link-time dependency. If the allocator runtime is not present, the custom op continues to use the copy-based path.

Set the following environment variable to log whether real partition tensors are allocator-backed:

```bash
export NN2FPGA_ZERO_COPY_LOG=1
```

Allocator allocation/free logging is separate and disabled by default. Enable it only when debugging allocator behavior:

```bash
export NN2FPGA_ALLOCATOR_LOG=1
```

Detailed allocator size histograms are disabled by default. Enable them with:

```bash
export NN2FPGA_ALLOCATOR_STATS=1
```

The allocator can also keep freed XRT BOs in an exact-size process-local pool instead of destroying them immediately:

```bash
export NN2FPGA_ALLOCATOR_POOL=1
```

The generic C++ runner prints only setup, final output metadata, and average latency by default. Enable per-run logs with:

```bash
export NN2FPGA_RUNNER_VERBOSE=1
```

Set the following environment variable to force the input copy path even when allocator lookup succeeds:

```bash
export NN2FPGA_DISABLE_ZERO_COPY=1
```

Example output when an input or output tensor is backed by the nn2FPGA allocator:

```text
[nn2fpga zero-copy] input 0 ptr=0xffff93339000 registered=1 alloc_size=3072 bytes=3072 device_addr=0x77ec4000
[nn2fpga zero-copy] output 0 ptr=0xffff93338000 registered=1 alloc_size=3072 bytes=3072 device_addr=0x77595000
```

Example output without allocator registration:

```text
[nn2fpga zero-copy] input 0 ptr=0xffff93339000 registered=0 alloc_size=0 bytes=3072 device_addr=0x0
```

For inputs, allocator-backed pointers use the direct MM2S path. For outputs, these logs are still diagnostics only; output DMA still uses the internal BO and copy-back path described in [Standard Copy-Based Inference](#standard-copy-based-inference).

## Testing

To test the standard Python flow, run the generated model through the usual Python ONNX Runtime path and enable diagnostics:

```bash
export NN2FPGA_ZERO_COPY_LOG=1
python3 inference.py
```

Expected behavior is correct inference. If diagnostics are printed, pointers should normally report `registered=0` because the Python process did not register the nn2FPGA allocator.

To test allocator-backed tensors, build and use the generic allocator-enabled C++ runner:

```bash
./tools/build_allocator_test.sh
export NN2FPGA_ZERO_COPY_LOG=1
./artifacts/allocator_test/aarch64/run_ort_with_allocator model.onnx libnn2fpga_customop.so 1
```

The same runner can also execute without allocator registration, which is useful to separate normal C++ ONNX Runtime overhead from allocator overhead:

```bash
./run_ort_with_allocator --no-allocator model.onnx libnn2fpga_customop.so 20
```

Add `--profile` to emit an ONNX Runtime profiling trace:

```bash
./run_ort_with_allocator --profile model.onnx libnn2fpga_customop.so 20
./run_ort_with_allocator --no-allocator --profile model.onnx libnn2fpga_customop.so 20
```

The runner implements this setup:

1. Create an `OrtEnv`.
2. Register `nn2fpga_register_xrt_cpu_allocator(...)` on that environment.
3. Enable environment allocators for the session.
4. Register `libnn2fpga_customop.so`.
5. Allocate model inputs through the nn2FPGA allocator.
6. Run the generated ONNX model.

Expected diagnostics in allocator mode are `registered=1` and nonzero `device_addr` values for tensors allocated by ONNX Runtime through the nn2FPGA allocator.

For an input-copy versus input-zero-copy A/B comparison, use the same allocator-enabled runner and toggle only `NN2FPGA_DISABLE_ZERO_COPY`:

```bash
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
unset NN2FPGA_ZERO_COPY_LOG
unset NN2FPGA_ALLOCATOR_LOG
unset NN2FPGA_RUNNER_VERBOSE

NN2FPGA_DISABLE_ZERO_COPY=1 \
  ./run_ort_with_allocator nn2FPGA_resnet8.onnx ./libnn2fpga_customop.so 20

unset NN2FPGA_DISABLE_ZERO_COPY
./run_ort_with_allocator nn2FPGA_resnet8.onnx ./libnn2fpga_customop.so 20
```

The first run should report `input 0 path=copy`. The second run should report `input 0 path=zero-copy` when the partition input is allocator-backed.

When investigating allocator overhead, compare these three modes:

```bash
./run_ort_with_allocator --no-allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20

NN2FPGA_DISABLE_ZERO_COPY=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20

unset NN2FPGA_DISABLE_ZERO_COPY
./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20
```

The first mode measures C++ ONNX Runtime without the nn2FPGA allocator. The second mode measures allocator overhead while preserving the input copy path. The third mode measures allocator overhead plus input direct DMA.

To test whether allocator overhead is dominated by repeated XRT BO allocation/free, repeat the allocator modes with pooling enabled:

```bash
export NN2FPGA_ALLOCATOR_POOL=1
export NN2FPGA_ALLOCATOR_STATS=1

NN2FPGA_DISABLE_ZERO_COPY=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20

unset NN2FPGA_DISABLE_ZERO_COPY
./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20
```

If pooling removes most of the allocator slowdown, repeated `xrt::bo` construction/destruction is the main issue. If pooling does not help, CPU execution on XRT-mapped memory or cache synchronization is likely the dominant cost.

## Limitations

The current Python ONNX Runtime API cannot register arbitrary custom C allocator callbacks. Allocator-enabled execution therefore requires an NN2FPGA-owned runtime/session wrapper or another C/C++ integration layer that owns the `OrtEnv` and registers the allocator before session creation.

The current production custom op remains copy-compatible. Allocator lookup is used for input-side direct DMA when available, but output DMA does not yet target ORT tensor device addresses.

Direct output DMA to allocator-backed ORT tensors requires additional scatter-gather descriptor retargeting work in `S2mmSG`.
