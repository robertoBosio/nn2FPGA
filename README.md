
# nn2FPGA

## Introduction

nn2FPGA is a framework for accelerating quantized ONNX inference on AMD FPGAs.

It operates on QONNX models by identifying the largest FPGA-supported subgraph, generating the corresponding hardware accelerator, and reintegrating it into the ONNX model for execution on heterogeneous systems.

The generated accelerator follows a **dataflow execution model**, departing from the traditional Von Neumann architecture. Instead of relying on sequential instruction execution and shared memory, computations are mapped into streaming pipelines, enabling high parallelism and efficient data movement—an approach well suited to FPGA architectures.

nn2FPGA targets resource-constrained embedded platforms, where performance, latency, and energy efficiency are critical. By offloading supported portions of a neural network to FPGA-based accelerators, it enables efficient heterogeneous execution while maintaining compatibility with ONNX Runtime.

The project is open-source and released under the MIT license.

---

## Installation

### Prerequisites

* Linux system (tested on CentOS 8 and Ubuntu 20.04)
* Docker
* AMD Xilinx tools (tested with Vivado/Vitis HLS 2025.1)
* Xilinx FPGA board with a Vitis license (required for full deployment)

### Setup

```bash id="setup02"
git clone git@github.com:robertoBosio/NN2FPGA.git
cd NN2FPGA
git submodule update --init --recursive
./run.sh
```

---

## Quick Start

The easiest way to get started is by running one of the provided examples.

1. **Select an example configuration.**
   Example configurations are available in:

   ```
   config_examples/
   ```

2. **Run the framework.**

    ```bash id="run02"
    python3 py/nn2fpga.py --config config_examples/<example>.toml
    ```

3. **Collect the output.**
   Copy the generated `deploy` directory to the target board. It contains:

   * The ONNX model with the embedded bitstream
   * The compiled nn2FPGA custom operator

4. **Run the ONNX model.**
   Add the following lines to your inference script to register the custom operator:

    ```python
    CUSTOM_OP_SO = os.path.abspath("libnn2fpga_customop.so")
    so = ort.SessionOptions()
    so.register_custom_ops_library(CUSTOM_OP_SO)
    ```

    Examples of inference scripts are available in the `deploy` directory.
