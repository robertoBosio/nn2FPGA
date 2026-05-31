# Zero-Copy Investigation

This note summarizes the YOLOv5nu zero-copy investigation on Kria/ZynqMP. It records the measured behavior of the XRT-backed ONNX Runtime allocator, the input-side direct-DMA path, and why full zero-copy is not currently the next best optimization target.

The short conclusion is that input zero-copy works, but it saves only about `0.4-0.7 ms/image` for this model. The standard copy-based custom-op path remains the fastest measured end-to-end path because normal ONNX Runtime CPU tensors are still faster than globally replacing ORT allocations with XRT BO-backed memory.

## Tested Modes

Three main execution modes were compared.

Standard C++ ONNX Runtime without the nn2FPGA allocator:

```bash
./run_ort_with_allocator --no-allocator \
  nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20
```

Allocator enabled, but input zero-copy disabled. This measures global allocator overhead while preserving the input copy path inside `nn2fpgaPartition`:

```bash
NN2FPGA_DISABLE_ZERO_COPY=1 \
NN2FPGA_ALLOCATOR_CACHEABLE=1 \
NN2FPGA_ALLOCATOR_POOL=1 \
NN2FPGA_ENABLE_MEM_PATTERN=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20
```

Allocator enabled with input zero-copy. This lets the custom op use allocator-backed partition inputs directly as MM2S source buffers:

```bash
NN2FPGA_ALLOCATOR_CACHEABLE=1 \
NN2FPGA_ALLOCATOR_POOL=1 \
NN2FPGA_ENABLE_MEM_PATTERN=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20
```

For batch-size experiments, pass the dynamic batch as the final argument:

```bash
./run_ort_with_allocator --no-allocator \
  nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20 10
```

## Important Environment Variables

`NN2FPGA_ALLOCATOR_CACHEABLE=1` requests `xrt::bo::flags::cacheable` when the allocator creates BOs. This is required for reasonable CPU performance on embedded platforms when ORT CPU kernels operate on allocator-backed tensors.

`NN2FPGA_ALLOCATOR_POOL=1` keeps freed XRT BOs in an exact-size process-local pool. This reduces repeated BO allocation/destruction overhead, but did not change the main performance conclusion.

`NN2FPGA_ENABLE_MEM_PATTERN=1` enables ORT memory pattern planning in the allocator-enabled runner.

`NN2FPGA_DISABLE_ZERO_COPY=1` forces the custom op to use the existing input copy path even when the input tensor is allocator-backed.

`NN2FPGA_ZERO_COPY_LOG=1` logs whether partition inputs and outputs are allocator-backed and whether each input used the copy or zero-copy path.

## Cacheable BO Finding

The first allocator implementation used default XRT BO flags. On YOLOv5nu, this caused a severe slowdown in ORT CPU kernels. For example, `global_in_transpose` was previously about `135 ms/run` with the global allocator, versus about `5-6 ms/run` with normal ORT memory.

XRT documents `xrt::bo::flags::cacheable` as effective on embedded platforms. Enabling it with `NN2FPGA_ALLOCATOR_CACHEABLE=1` fixed the catastrophic slowdown:

```text
no allocator global_in_transpose:         ~6.06 ms/run
cacheable allocator global_in_transpose:  ~5.93-5.98 ms/run
```

This means default non-cacheable or weakly cached BO mappings are not appropriate for general ORT CPU tensors on this board. Cacheable BOs are required for any future global allocator experiments.

## Batch 1 Results

Clean non-profiled batch-1 results:

```text
mode                              batch latency    image latency
no allocator                         54.813 ms        54.813 ms
cacheable allocator + copy           55.777 ms        55.777 ms
cacheable allocator + zero-copy      55.334 ms        55.334 ms
```

Input zero-copy saved:

```text
55.777 - 55.334 = 0.443 ms/image
```

But the allocator zero-copy path was still slightly slower than normal ORT allocation:

```text
55.334 - 54.813 = 0.521 ms/image slower
```

The profiled batch-1 zero-copy run showed:

```text
nn2fpgaPartition_FPGA_kernel_time: 777.826 ms / 20 = 38.891 ms/run
global_in_transpose_kernel_time:   119.543 ms / 20 = 5.977 ms/run
global_in_quantize_kernel_time:     55.974 ms / 20 = 2.799 ms/run
Sigmoid_0_kernel_time:              63.939 ms / 20 = 3.197 ms/run
```

For comparison, the raw accelerator benchmark reported:

```text
throughput_test.py avg batch latency: 36.024 ms
```

So the approximate breakdown for batch 1 is:

```text
raw accelerator:                  36.024 ms/image
ORT nn2fpgaPartition zero-copy:   38.891 ms/image
full ORT graph zero-copy:         55.334 ms/image
```

The custom-op FPGA path is only about `2.87 ms` slower than the raw accelerator benchmark. The larger gap is the remaining ONNX Runtime CPU graph around the FPGA partition.

## Batch 10 Results

Clean non-profiled batch-10 results:

```text
mode                              batch latency    image latency
no allocator                        399.709 ms        39.971 ms
cacheable allocator + copy          421.613 ms        42.161 ms
cacheable allocator + zero-copy     414.892 ms        41.489 ms
```

Input zero-copy saved:

```text
421.613 - 414.892 = 6.721 ms/batch
6.721 / 10 = 0.672 ms/image
```

But allocator zero-copy was still slower than normal ORT allocation:

```text
414.892 - 399.709 = 15.183 ms/batch
15.183 / 10 = 1.518 ms/image slower
```

The batch-10 profile showed that zero-copy reduced the partition time, but the global allocator slowed CPU preprocessing:

```text
no allocator nn2fpgaPartition:        258.292 ms/batch
allocator copy nn2fpgaPartition:      259.039 ms/batch
allocator zero-copy nn2fpgaPartition: 254.199 ms/batch
```

The partition improvement from zero-copy was:

```text
259.039 - 254.199 = 4.840 ms/batch
4.840 / 10 = 0.484 ms/image
```

But `global_in_transpose` became slower with the global allocator:

```text
no allocator global_in_transpose:        39.247 ms/batch
allocator copy global_in_transpose:      58.800 ms/batch
allocator zero-copy global_in_transpose: 59.097 ms/batch
```

That single CPU node adds about `20 ms/batch`, wiping out the input zero-copy gain.

## Raw Accelerator Versus Full ONNX Runtime

The generated `throughput_test.py` benchmark measures the raw FPGA/DMA path, not the full ONNX model. It starts timing after input buffers have already been allocated and after dynamic input data has been copied into the PYNQ buffer.

The full ONNX Runtime model also executes CPU-side ONNX nodes around the FPGA partition, including preprocessing and postprocessing. For YOLOv5nu, important CPU-side costs include:

```text
global_in_transpose
global_in_quantize
Sigmoid_0
Quant_400_out0_transpose
Quant_400_out0_dequantize
Concat_21
Conv_75_out0_transpose
```

Therefore, raw accelerator latency and full ORT latency should not be expected to match. The useful comparison is:

```text
raw accelerator throughput_test.py
ORT nn2fpgaPartition node time
full ORT graph latency
```

In the measured batch-1 run, the raw accelerator was about `36.024 ms`, the ORT partition was about `38.891 ms`, and the full ORT graph was about `55.334 ms`.

## Decision

Input zero-copy is functionally correct, but it is not a large performance win for this model.

The measured input zero-copy gain is about `0.4-0.7 ms/image`. That is small compared with the full graph latency and smaller than the overhead introduced by globally allocating ORT tensors from XRT BO-backed memory.

The standard copy-based custom-op path should remain the default. The allocator and input zero-copy path should remain experimental and diagnostic unless a future model shows that FPGA boundary copies are dominant.

## Recommended Future Work

The next useful optimization target is not full global zero-copy. Better candidates are:

1. Fuse `global_in_transpose` and `global_in_quantize` into an FPGA-boundary preprocessing path that writes directly in the accelerator input layout and dtype.
2. Reduce or move output-side postprocessing, especially `Sigmoid`, output transposes, dequantization, and concat operations.
3. Add detailed timers inside `FpgaRunnerT::run()` to split partition time into input copy, input sync, DMA wait, output sync, and output copy.
4. Consider selective boundary allocation only if a future model has larger partition boundary tensors or profiles show copy/sync dominating.

The global allocator can still be useful for experiments, but any future global allocator test on embedded platforms should use:

```bash
NN2FPGA_ALLOCATOR_CACHEABLE=1
```

without this flag, ORT CPU kernels can become dramatically slower on XRT BO-backed tensors.

## Replication Notes

This investigation used experimental runtime and tool files that may not remain in the main development tree. If the zero-copy implementation is removed, preserve this note together with a branch, tag, or archived patch containing the experimental files.

Suggested archival name:

```text
zero-copy-investigation-2026-05
```

The experiment depended on the following pieces:

```text
nn2fpga/hw/operator_runtime/nn2FPGA_allocator.hpp
nn2fpga/hw/operator_runtime/nn2FPGA_allocator.cpp
tools/run_ort_with_allocator.cpp
tools/test_ort_cpu_allocator.cpp
tools/allocator_probe_op.cpp
tools/build_allocator_test.sh
tools/make_allocator_test_model.py
tools/make_allocator_probe_model.py
tools/summarize_ort_profile.py
```

The custom-op runtime also contained temporary hooks in:

```text
nn2fpga/hw/operator_runtime/nn2FPGA_kernel.hpp
nn2fpga/hw/operator_runtime/xrt_dma.h
```

Those hooks provided:

```text
dlsym lookup for nn2fpga_allocator_lookup
dlsym lookup for nn2fpga_allocator_sync_to_device
NN2FPGA_ZERO_COPY_LOG diagnostics
NN2FPGA_DISABLE_ZERO_COPY toggle
input-side MM2S direct DMA from allocator BO device address
```

The allocator provided:

```text
OrtEnv allocator registration through nn2fpga_register_xrt_cpu_allocator
XRT BO-backed CPU allocations
pointer range lookup to recover BO device addresses
explicit sync_to_device and sync_from_device helpers
optional exact-size BO pooling
optional cacheable BO allocation
allocator statistics and size histograms
```

The most important implementation detail was the cacheable BO mode:

```cpp
xrt::bo::flags::cacheable
```

Without this flag, global ORT allocation from XRT BO-backed memory made CPU-heavy ORT nodes much slower on the tested embedded platform.

The cross-build command used for allocator test tooling was:

```bash
CXX=aarch64-linux-gnu-g++-11 \
SYSROOT=/opt/sysroots/board \
ONNXRUNTIME_SDK_INCLUDE=/opt/onnxruntime-sdk/include \
ONNXRUNTIME_LIBDIR=/opt/onnxruntime-sdk/lib \
XRT_EXTRA_LIBDIRS=$PWD/tools \
  ./tools/build_allocator_test.sh
```

The generated runner had this usage:

```bash
./run_ort_with_allocator [--no-allocator] [--profile] \
  <model.onnx> <custom_op.so> [runs] [dynamic_batch]
```

The batch-1 replication matrix was:

```bash
./run_ort_with_allocator --no-allocator \
  nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20

NN2FPGA_DISABLE_ZERO_COPY=1 \
NN2FPGA_ALLOCATOR_CACHEABLE=1 \
NN2FPGA_ALLOCATOR_POOL=1 \
NN2FPGA_ENABLE_MEM_PATTERN=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20

NN2FPGA_ALLOCATOR_CACHEABLE=1 \
NN2FPGA_ALLOCATOR_POOL=1 \
NN2FPGA_ENABLE_MEM_PATTERN=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20
```

The batch-10 replication matrix was:

```bash
./run_ort_with_allocator --no-allocator \
  nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20 10

NN2FPGA_DISABLE_ZERO_COPY=1 \
NN2FPGA_ALLOCATOR_CACHEABLE=1 \
NN2FPGA_ALLOCATOR_POOL=1 \
NN2FPGA_ENABLE_MEM_PATTERN=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20 10

NN2FPGA_ALLOCATOR_CACHEABLE=1 \
NN2FPGA_ALLOCATOR_POOL=1 \
NN2FPGA_ENABLE_MEM_PATTERN=1 \
  ./run_ort_with_allocator nn2FPGA_yolov5nu.onnx ./libnn2fpga_customop.so 20 10
```

Profile traces were collected by adding `--profile` and summarized with:

```bash
python3 summarize_ort_profile.py nn2fpga_runner_profile*.json --top 30
```

If this experiment is revisited after deleting the active implementation, restore the archived files first, rebuild both the allocator runner and `libnn2fpga_customop.so`, then repeat the command matrices above.
