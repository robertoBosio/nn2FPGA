#!/bin/bash
set -e

# Read project root from environment, with fallback
SILVIA_DIR="${NN2FPGA_ROOT_DIR}/deps/SILVIA"
PASS_LIB_PATH="${SILVIA_DIR}/build/SILVIAMuladd/LLVMSILVIAMuladd.so"

if [[ "$(printf '%s\n2025.1' "$XILINX_VERSION" | sort -V | tail -n1)" == "$XILINX_VERSION" ]]; then
    # old structure
    VIVADO_PATH="${XILINX_DIR}/Xilinx/${XILINX_VERSION}/Vivado"
    VITIS_PATH="${XILINX_DIR}/Xilinx/${XILINX_VERSION}/Vitis"
else
    # new structure
    VIVADO_PATH="${XILINX_DIR}/Xilinx/Vivado/${XILINX_VERSION}"
    VITIS_PATH="${XILINX_DIR}/Xilinx/Vitis/${XILINX_VERSION}"
fi
echo "Sourcing Xilinx tools from $VITIS_PATH and $VIVADO_PATH"


HOME="/tmp/homedir"

echo "Project root: $NN2FPGA_ROOT_DIR"
echo "Sourcing Xilinx tools from $XILINX_DIR"

if [ -f "$VITIS_PATH/settings64.sh" ]; then
    source "$VITIS_PATH/settings64.sh"
else
    echo "Unable to find Vitis" >&2
    exit 1
fi
if [ -f "$VIVADO_PATH/settings64.sh" ]; then
    source "$VIVADO_PATH/settings64.sh"
else
    echo "Unable to find Vivado" >&2
    exit 1
fi

echo "Looking for compiled pass at: $PASS_LIB_PATH"

# Compile SILVIA pass if missing
if [ ! -f "$PASS_LIB_PATH" ]; then
    echo "LLVM pass not found — compiling..."

    cd "$SILVIA_DIR"
    # chmod +x install_llvm.sh build_pass.sh

    if ! ./install_llvm.sh; then
        echo "install_llvm.sh failed." >&2
        exit 1
    fi

    if ! ./build_pass.sh; then
        echo "compile_pass.sh failed." >&2
        exit 1
    fi

    if [ ! -f "$PASS_LIB_PATH" ]; then
        echo "Compilation finished, but LLVM pass not found at $PASS_LIB_PATH" >&2
        exit 1
    fi


    echo "LLVM pass compiled successfully."
else
    echo "LLVM pass already compiled. Skipping rebuild."
fi

mkdir -p "$HOME"
mkdir -p "$HOME/.Xilinx"

# Set up environment variables
export XILINX_VIVADO="${VIVADO_PATH}"
export XILINX_VITIS="${VITIS_PATH}"
export XILINX_XRT="/opt/xilinx/xrt"
export SILVIA_ROOT="${SILVIA_DIR}"
export SILVIA_LLVM_ROOT="${SILVIA_DIR}/llvm-project/install"
export HOME="${HOME}"
export SYSROOT=${SYSROOT:-/opt/sysroots/board}
export ONNXRUNTIME_SDK_INCLUDE=${ONNXRUNTIME_SDK_INCLUDE:-/opt/onnxruntime-sdk/include}
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++

# chmod +x tools/build_customop.sh

cd $NN2FPGA_ROOT_DIR/nn2fpga

# Run user-supplied command
exec "$@"
