# Runtime Layout Transform Experiment

This note records the YOLOv5nu experiment that moved FPGA boundary layout transforms from ONNX Runtime `Transpose` nodes into the nn2FPGA runtime copy path.

## Motivation

The accelerator streams tensors in the layout inferred by nn2FPGA. For convolutional inputs this is commonly NHWC, while ONNX tensors are canonical NCHW.

The standard runtime path already copies ONNX Runtime tensor data into XRT-backed buffers before starting DMA transfers. The experiment tested whether that unavoidable copy could also perform the layout conversion, removing separate CPU-side ONNX `Transpose` nodes.

Conceptually, the tested path replaced:

```text
ONNX tensor
  -> ONNX Runtime Transpose
  -> nn2FPGA runtime memcpy into XRT BO
  -> FPGA DMA
```

with:

```text
ONNX tensor
  -> nn2FPGA runtime copy-transpose into XRT BO
  -> FPGA DMA
```

## Tested Transforms

The implementation was intentionally narrow and covered only the YOLOv5nu boundary layouts observed during the test.

Input transform:

```text
layout: L[0,2,3,1]
ONNX input:      NCHW
accelerator BO: NHWC
```

Output transform:

```text
layout: L[0,2,1]
accelerator BO: NFC
ONNX output:    NCF
```

Output transform:

```text
layout: L[0,3,2,1]
accelerator BO: NWHC
ONNX output:    NCHW
```

All other layouts were left to the existing fallback: insert ONNX `Transpose` nodes and use a plain runtime `memcpy`.

## Measurements

The following measurements were taken on the ZynqMP target for YOLOv5nu, batch size 1.

Raw accelerator benchmark, using `throughput_test.py`:

```text
Avg image latency: 27.935 ms
```

Previous optimized ONNX Runtime path, with CPU ONNX `Transpose` nodes plus runtime `memcpy`:

```text
Avg image latency: 79.646 ms
```

Fused runtime copy-transpose path, with scalar copy-transpose loops inside `FpgaRunnerT::run()`:

```text
Avg image latency: 161.307 ms
```

The fused runtime path was about 2x slower than the previous optimized ONNX Runtime path for this test.

## Interpretation

The optimization idea was valid in terms of memory traffic, but the implementation was not competitive with ONNX Runtime's optimized transpose kernels.

The runtime implementation used simple scalar nested loops inside the custom operator. ONNX Runtime's `Transpose` implementation is likely better optimized for the CPU, including cache behavior, threading, and vectorization. Moving the transform inside the custom op also makes the nn2FPGA partition time include the full host-side transpose cost.

The measured regression indicates that an unfused but optimized ONNX Runtime transpose can be faster than a fused but naive custom copy-transpose.

## Decision

Runtime layout transforms should not be enabled by default based on this implementation.

The safe default remains:

```text
unsupported or unoptimized boundary layout
  -> insert ONNX Transpose
  -> use runtime memcpy
```

Runtime copy-transpose should be treated as experimental until a faster implementation is measured on target hardware.

## Future Work

Before enabling runtime layout transforms by default, add fine-grained runtime timers around:

```text
input copy or input pack
input BO sync to device
MM2S transfer and wait
S2MM transfer and wait
output BO sync from device
output copy or output unpack
```

Potential optimized implementations to benchmark:

```text
specialized C == 3 input NCHW -> NHWC path
tiled output transposes for large feature dimensions
NEON/vectorized implementations
multi-threaded host copy/transpose, if safe in the runtime context
compiler flags for the generated custom operator library
```

Any future change should be evaluated against both:

```text
raw accelerator latency from throughput_test.py
full ONNX Runtime latency and profile traces
```

The relevant comparison is not whether copy-transpose is faster than `memcpy`; it must be faster than ONNX Runtime `Transpose` plus runtime `memcpy`.
