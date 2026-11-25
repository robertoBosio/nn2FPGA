# Catapult Backend Strategy

This document defines the incremental plan for adding a Mentor Catapult HLS
backend to nn2FPGA. It is a design and delivery plan, not a claim of current
support.

## Goal

Compile nn2FPGA streaming accelerators with Catapult while retaining the
existing AXI DMA data path on Xilinx platforms. The initial target is Catapult
10.5c with the Zynq UltraScale+ -2LV behavioral library. Vivado remains the
implementation and bitstream tool for Kria targets.

The Catapult library does not contain the exact K26 part. Catapult synthesis
therefore provides Zynq UltraScale+ -2LV scheduling and resource estimates;
the exact K26 part is selected later when Vivado packages and implements the
RTL.

## Initial Scope

The first supported Catapult design is a free-running streaming accelerator:

```text
AXI DMA <-> AXI4-Stream wrapper <-> Catapult ready/valid kernel
```

The Catapult kernel has clock, reset, and streaming data ports only. It has no
start/done/idle protocol and no control register bank.

Supported initially:

- AXI4-Stream DMA data movement at accelerator boundaries.
- Streamed model parameters as additional AXI4-Stream ports.
- Catapult C simulation, synthesis, extracted RTL, and schedule reports.
- Vivado IP packaging of the Catapult RTL plus generated AXI4-Stream wrapper.

Explicitly unsupported initially:

- `m_axi` interfaces and external-memory buffers.
- AXI-Lite control and pointer registers.
- `ap_ctrl_none`, `s_axilite`, and other Vitis interface pragmas.
- SILVIA, which depends on the Vitis LLVM flow.
- Vitis-only LightningSim generation.
- Existing Vitis generated-Verilog patching and driver-header register parsing.

The compiler must reject unsupported Catapult configurations early with clear
errors rather than silently producing a Vitis-oriented design.

## Boundary Contract

Catapult `ac_channel` ports synthesize to a leaf-level ready/valid protocol.
The generated AXI4-Stream wrapper adapts that protocol to the DMA-facing IP
interface. This wrapper replaces Vitis AXIS inference, not the AXI DMA design.

The Catapult boundary payload must preserve all sidebands used by nn2FPGA:

```cpp
struct AxisWord {
    ac_int<DATA_WIDTH, false> data;
    ac_int<DATA_WIDTH / 8, false> keep;
    ac_int<DATA_WIDTH / 8, false> strb;
    ac_int<1, false> last;
};
```

Input `TKEEP`, `TSTRB`, and `TLAST` are currently not consumed by the
nn2FPGA input adapter, but output `TKEEP`, `TSTRB`, and `TLAST` are required:
`StreamToAXIS` uses them to describe the final partial word and end of tensor.
The wrapper must map all four fields rather than hardcoding them as in the
fixed-size Catapult example.

`TUSER`, `TID`, and `TDEST` are not needed by the current design because its
Vitis ports use `ap_axiu<W, 0, 0, 0>`.

## Backend Architecture

The compiler must separate backend-neutral graph assembly from backend-specific
source, project, and platform generation.

Shared responsibilities:

- ONNX/QONNX transforms and supported-partition selection.
- Stream graph topology, quantization metadata, and operator parameters.
- Operator declaration, object construction, and invocation order.
- Logical FIFO-depth analysis and host-side tensor marshalling intent.

Vitis backend responsibilities:

- Existing `ap_*` types, `hls::stream`, HLS pragmas, Tcl, IP export, and
  deployment behavior.

Catapult backend responsibilities:

- Algorithmic C types, channels, scheduling directives, and Catapult Tcl.
- Catapult schedule-report handling.
- Free-running streaming top generation.
- AXI4-Stream wrapper generation and Vivado IP packaging.

The Vitis implementation must stay behaviorally unchanged while this boundary
is introduced.

## Library Strategy

The current library is Vitis-oriented: it uses `ap_int`, `ap_uint`,
`ap_float`, `hls::stream`, and nearly two hundred `#pragma HLS` directives.
The Catapult backend cannot be a pragma text substitution.

Use a deliberately separated library surface:

```text
hw/library/include/common/    Backend-neutral algorithms and traits
hw/library/include/vitis/     Existing Vitis implementations and policies
hw/library/include/catapult/  Algorithmic C and Catapult implementations
```

The code generator must include only headers required by the lowered graph.
The current approach includes every library header, which prevents an
incremental Catapult bring-up because unused Vitis-only headers are parsed.

Porting priorities:

1. Integer traits and packed-word helpers.
2. A synthesizable channel abstraction with blocking `read()` and `write()`.
3. AXIS input/output adapters using the `AxisWord` payload.
4. Basic elementwise operators and quantization.
5. Static-weight convolution, line buffering, and pooling.
6. Storage policies, streamed parameters, and larger operator families.

Non-synthesis `step()` logic, FIFO registries, and Vitis self-timed simulation
support should not be compiled into the initial Catapult synthesis library.

## Compiler Configuration

Add an explicit backend option, defaulting to the current Vitis behavior:

```toml
[project]
hls_backend = "vitis" # or "catapult"
```

For `hls_backend = "catapult"`:

- Require streaming boundary mode (`use_dma = true` in the current
  configuration vocabulary).
- Reject `silvia_packing = true`.
- Reject external memory buffers and `m_axi` ports.
- Reject unsupported build/deployment steps until their Catapult variants are
  implemented.
- Validate that `CATAPULT_HOME` is available when a Catapult tool step runs.

The overloaded `use_dma` name should eventually be replaced by a neutral
interface-mode setting. Until then, Catapult treats it as a request for
streaming boundary ports, not as a request to synthesize a DMA.

## Milestones

### M0: Container and Tool Access

Completed foundation:

- Optional `CATAPULT_HOME` host mount in `run.sh`.
- Optional `MGLS_LICENSE_FILE` forwarding.
- `MGC_HOME` and `PATH` setup in the container entrypoint.

Acceptance: `catapult -version` runs inside the project container when the
host provides a valid installation.

### M1: Backend Selection and Validation

- Add `hls_backend` configuration and metadata.
- Introduce a backend capability model or equivalent validation path.
- Preserve Vitis defaults and test coverage.
- Reject Catapult/Vitis-incompatible options before lowering.

Acceptance: both backend selections load correctly; invalid Catapult
combinations produce actionable errors without invoking a tool.

### M2: Minimal Catapult Source and Project Generation

- Split common source assembly from Vitis-specific `EmbedHLSCode` behavior.
- Generate a Catapult top using `ac_int` and `ac_channel`.
- Generate Catapult Tcl using C++11, the Zynq UltraScale+ -2LV library,
  `Xilinx_RAMS`, project clock frequency, and explicit pipeline directives.
- Compile, assemble, and extract RTL through Catapult.

Acceptance: a direct channel-to-channel test kernel compiles and synthesizes;
the generated reports and RTL are saved as build artifacts.

### M3: Core Streaming Library

- Port integer quantization and packed-word utilities.
- Port `DequantQuantEqual`, `StreamingReLU`, elementwise operations, split,
  concat, and reshape.
- Support AXIS input and output adapters using `AxisWord`.

Acceptance: Catapult C simulation and RTL simulation match the software model
for a one-input/one-output streaming graph, including a partial final word.

### M4: AXI4-Stream Wrapper and Vivado IP

- Generate an RTL wrapper around every Catapult leaf port.
- Preserve `TDATA`, `TVALID`, `TREADY`, `TKEEP`, `TSTRB`, and `TLAST`.
- Package the Catapult RTL and wrapper as a Vivado IP.
- Adapt the Vivado block-design flow to consume the packaged Catapult IP.

Acceptance: Vivado connects the generated IP to AXI DMA and a hardware test
transfers multiple packets with correct output data and end-of-tensor markers.

### M5: CNN Dataflow Operators

- Port static-weight `StreamingConv` and fused activation.
- Port line buffers and max-pooling.
- Map array partitioning, storage, and pipeline requirements to Catapult
  directives and Xilinx RAM policies.

Acceptance: a small quantized CNN synthesizes and matches Vitis/software
functional results under DMA transfer.

### M6: FIFO and Schedule Feedback

- Keep logical FIFO-depth inference initially.
- Add a Catapult schedule parser only after stable report fields are identified.
- Validate Catapult pipeline skew and FIFO requirements independently from
  Vitis `.autopilot` assumptions.

Acceptance: the backend does not read Vitis report paths and produces stable
FIFO sizing for the supported streaming operator set.

### M7: Advanced Features

- Streamed parameters beyond the initial boundary support.
- `StreamingMemory`, RAM/URAM policies, and larger models.
- Optional AXI-Lite and `m_axi` support, designed as explicit new contracts.
- Catapult deployment/runtime packaging parity where practical.

Acceptance criteria are defined per feature; these capabilities are not
implied by early streaming support.

## Verification Strategy

Each milestone adds the smallest appropriate test layer:

- Python configuration and generated-source tests without licensed tools.
- Catapult C simulation tests for each newly ported operator.
- Catapult synthesis smoke tests when the toolchain is available.
- RTL simulation of the generated AXIS wrapper, including backpressure and
  partial final words.
- Vivado DMA integration tests after M4.
- End-to-end model equivalence against the existing software reference.

Vitis tests remain the regression baseline and must not be replaced by
Catapult tests.

## Deferred Decisions

- The exact Catapult representation and bit-accurate policy for float values.
- Catapult equivalents for Vitis dataflow, storage binding, and array banking.
- A stable Catapult schedule-report schema for pipeline-skew analysis.
- The AXI-Lite and `m_axi` contracts, if they are needed after the streaming
  backend is established.
