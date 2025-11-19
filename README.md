# NN2FPGA
## Introduction
nn2FPGA is a framework which generates quantized convolution neural networs accelerators in C++ for AMD FPGAs.
The main goal of this project is to provide a tool targeting embedded FPGAs keeping state-of-the-art performance.

The project is completely open-source, and it is released under the MIT license.
We would be happy to receive contributions from the community.

## Installation
### Prerequisites
A machine with a Linux distribution (tested on Centos8 and Ubuntu 20.04), Docker and a recent version of the Xilinx suite (tested with Vivado/Vitis HLS 2025.1) is required.
To perform the whole flow, it is required to have a Xilinx FPGA board with a Vitis license.
### Installation
To install the framework, it is required to clone the repository and run the run.sh script to launch the docker.
```bash
git clone git@github.com:robertoBosio/NN2FPGA.git
cd NN2FPGA
./run.sh
```
## Usage
### Quick start
To run the framework, it is required to have a trained model in the QONNX format.
The framework is able to convert models from the ONNX format to the C++ code.
To convert a model, it is required to run the following command:
```bash
python3 py/nn2fpga.py --config config.toml
```

# TODO 
- [ ] Create the bias node when folding asymmetric quantization in nodes without biases.
