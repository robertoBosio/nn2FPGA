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

## Inference Mode

The production runtime mode is the standard ONNX Runtime custom-op flow. It loads `libnn2fpga_customop.so` and executes the FPGA partition through the copy-compatible SG data path described below.

The runtime is designed to work with ordinary Python ONNX Runtime sessions and normal CPU tensors. Optional allocator-backed zero-copy experiments are separate from the production path in this document.

## Standard Python Session

The standard Python flow registers only the custom op library:

```python
import os
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library(os.path.abspath("libnn2fpga_customop.so"))
session = ort.InferenceSession("model.onnx", so)
```

In this mode, Python ONNX Runtime uses its normal CPU allocators. The custom op receives ordinary CPU tensor pointers. The runtime therefore uses the internal XRT BO staging path described in [Inference Flow](#inference-flow).

The custom op advertises `CPUExecutionProvider` as its execution provider type. ONNX Runtime schedules the partition node like a CPU custom op, but the node implementation programs and drives the FPGA internally through XRT and memory-mapped DMA registers.

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

The production runtime accepts only `batch == 1`. If ONNX Runtime calls the custom operator with `batch > 1`, the runtime rejects the call with an invalid-argument error. Throughput is obtained by launching multiple `session.run(...)` calls concurrently against the same ONNX Runtime session.

The runtime is copy-compatible with normal Python ONNX Runtime tensors:

```text
ORT input tensor
  -> memcpy
  -> per-slot input XRT BO
  -> MM2S SG DMA descriptor
  -> FPGA accelerator
  -> S2MM SG DMA descriptor
  -> per-slot output XRT BO
  -> memcpy
  -> ORT output tensor
```

Each concurrent call gets a private request slot containing dynamic input BOs, dynamic output BOs, mapped host pointers, and descriptor handles. `Spec::N_MAX` is used as the number of request slots and SG descriptors per dynamic stream, not as an accepted ONNX batch size.

## SG Runtime Scheduler

Dynamic input streams and output streams use AXI DMA scatter-gather rings managed by `AxiDmaSgRing` in `xrt_dma.h`. Static inputs remain simple one-shot MM2S transfers.

The scheduler preserves accelerator ordering with two sequence counters:

1. `next_submit_sequence_`: permits request descriptors to be submitted in acquisition order.
2. `next_complete_sequence_`: permits output copy-back and descriptor reclamation in the same order.

For each request sequence, submission is ordered as:

1. Copy dynamic input tensors into the slot input BOs.
2. Sync input BOs to device.
3. Enqueue all output S2MM descriptors.
4. Enqueue all dynamic input MM2S descriptors.
5. Wait for output descriptors to complete.
6. In sequence order, sync output BOs from device and copy into ORT output tensors.
7. Reclaim input and output descriptors.
8. Release the request slot.

Output descriptors must be queued before input descriptors for the same sequence. This ensures the output DMA is ready before the accelerator can produce data.

Descriptor allocation is strict circular producer order, not first-free order. This keeps the DMA-visible descriptor chain contiguous:

```text
BD0 -> BD1 -> BD2 -> ... -> BD(N-1) -> BD0
```

A descriptor may be software-free but still unsafe to reuse if it is behind the DMA current descriptor and would create a hole in the ring. The producer-order rule prevents this failure mode.

When a ring becomes empty after descriptor reclamation, the runtime resets that DMA channel and marks it not started. This handles the single-inflight gap case where a later request arrives after the DMA has advanced beyond the previous tail descriptor.

If an SG DMA error or timeout occurs, the runtime resets and releases all SG rings before releasing the slot. This avoids leaking descriptor capacity after the first fault.

## Static Inputs

Some input ports are marked as `PortMode::StaticInit`. These ports are not exposed as dynamic ORT inputs.

During initialization, the runtime reads static values from the accelerator package `input_map`, copies them into the corresponding internal XRT BO, syncs the BO to the device, and performs the required DMA upload once through `Mm2sSimple`.

Static input DMAs are intentionally not configured for SG. They are skipped during normal `Compute(...)` calls.

## DMA Configuration

The generated Vivado design configures DMAs by port role:

* Dynamic input DMAs: MM2S SG enabled.
* Output DMAs: S2MM SG enabled.
* Static input DMAs: simple MM2S, SG disabled.

For SG DMAs, the generated TCL disables AXI DMA status/control streams:

```tcl
CONFIG.C_SG_INCLUDE_STSCNTRL_STRM {0}
```

This is required for the generated S2MM SG output path used by the custom operator.

Internal accelerator buffers used by HLS `m_axi` pointer ports are allocated as XRT BOs and their device addresses are written into the control register window during initialization.

## Debug Tracing

Runtime tracing is disabled by default. Enable it only when diagnosing scheduler or DMA behavior:

```bash
export NN2FPGA_TRACE=1
export NN2FPGA_TRACE_DMA=1
export NN2FPGA_TRACE_CTRL=1
```

`NN2FPGA_TRACE` prints high-level request, submit, wait, complete, reclaim, and slot events. `NN2FPGA_TRACE_DMA` prints DMA descriptor and status snapshots. `NN2FPGA_TRACE_CTRL` prints control register snapshots on selected timeout/error paths.

Useful SG error information includes descriptor index, sequence, DMA status, current descriptor, tail descriptor, descriptor status, and actual byte count.

## Correctness Testing

Correctness must be checked on raw ONNX outputs, not only through video or throughput tests. Video NMS can hide or amplify tensor corruption.

For YOLO COCO validation, use the generated benchmark with ORT optimizations disabled in correctness mode:

```bash
python3 ../onnx_inference_coco.py \
  --num-images 30 \
  --mode correctness \
  --model nn2FPGA_yolov5nu.onnx \
  --atol 10 \
  --rtol 0 \
  --inflight-runs 1
```

Then repeat with concurrent calls:

```bash
python3 ../onnx_inference_coco.py \
  --num-images 30 \
  --mode correctness \
  --model nn2FPGA_yolov5nu.onnx \
  --atol 10 \
  --rtol 0 \
  --inflight-runs 2

python3 ../onnx_inference_coco.py \
  --num-images 30 \
  --mode correctness \
  --model nn2FPGA_yolov5nu.onnx \
  --atol 10 \
  --rtol 0 \
  --inflight-runs 4
```

The tolerance is intentionally loose enough to ignore benign CPU/original versus FPGA numeric drift while still catching runtime corruption. If failures are only a few elements around `atol=10`, retest with a slightly larger tolerance before treating them as descriptor corruption. Large max errors or many bad elements indicate a runtime issue.

For throughput, use speed mode:

```bash
python3 ../onnx_inference_coco.py \
  --num-images 30 \
  --mode speed \
  --model nn2FPGA_yolov5nu.onnx \
  --inflight-runs 6
```

Speed mode measures concurrent `session.run(...)` calls on preprocessed tensors. It does not include video decode, YOLO preprocessing, NMS, rendering, or encoding.

## Video Benchmarking

The accelerated video benchmark should not use multiple independent Ultralytics `model.predict()` sessions for the FPGA model. The validated path is:

```text
OpenCV frame
  -> Ultralytics-compatible preprocess
  -> shared ORT InferenceSession.run(...)
  -> Ultralytics NMS/postprocess
  -> Ultralytics Results object
  -> render/encode
```

This matches the COCO correctness concurrency model: one shared ONNX Runtime session with multiple concurrent `session.run(...)` calls.

The video benchmark reports the explicit preprocess, ORT run, and postprocess block as accelerated `inference_s`. This is expected to be slower than COCO speed-mode `session.run(...)` latency because COCO speed mode measures only the ONNX Runtime call on already-preprocessed tensors.

For live-threaded video tests:

```bash
python3 ../build_yolov10n_2906/video_demo_benchmark.py \
  --video normalized.mp4 \
  --model-orig ../build_yolov5nu_0207/original_model_qcdq.onnx \
  --model-accel ../build_yolov5nu_0207/nn2FPGA_yolov5nu.onnx \
  --imgsz 640 \
  --batch 1 \
  --warmup-batches 0 \
  --measure-batches 200 \
  --custom-op ../build_yolov5nu_0207/libnn2fpga_customop.so \
  --mode realistic \
  --live-threaded \
  --live-threaded-hold-inference-frames \
  --save-video \
  --output-video yolov5nu_accel_live_threaded.mp4 \
  --half \
  --inflight-runs 6
```

Use COCO correctness tests to validate tensors first. Use video tests to validate end-to-end demo behavior and CPU-side preprocessing/postprocessing overhead.

## Limitations

The custom operator runtime currently supports only `batch == 1` per ONNX Runtime call.

The production Python path remains copy-based at the ORT tensor boundary. Dynamic input and output transfers between per-slot BOs and the accelerator use SG DMA, but ORT tensors are still copied into and out of those BOs.

Allocator-backed zero-copy experiments are not the production path described here. Direct output DMA into ORT-owned tensors would require a separate allocator/session integration and additional safety work.

Global fatal-error propagation is still minimal. The runtime resets SG rings after local DMA exceptions, but a more robust design should store the first fatal error, notify all scheduler condition variables, and make all waiters fail promptly.
