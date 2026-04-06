#!/usr/bin/env bash
set -euo pipefail

: "${SYSROOT:=/opt/sysroots/board}"
: "${ONNXRUNTIME_SDK_INCLUDE:=/opt/onnxruntime-sdk/include}"
: "${XRT_INC:=${SYSROOT}/usr/include/xrt}"
: "${XRT_LIBDIR:=${SYSROOT}/usr/lib/aarch64-linux-gnu}"
: "${TP_JSON:=/workspace/NN2FPGA/deps/json/single_include}"
: "${TP_BASE64:=/workspace/NN2FPGA/deps/cpp-base64}"
: "${NN2FPGA_HW_LIB:=/workspace/NN2FPGA/nn2fpga/hw/operator_runtime}"

SRC=${1:-nn2fpga/hw/operator_runtime/register_op.cpp}
OUT_DIR=${2:-artifacts/aarch64}
OUT_SO=${OUT_DIR}/libnn2fpga_customop.so
mkdir -p "${OUT_DIR}"

# Locate GCC-11 libstdc++ headers from the host toolchain
if   [[ -d /usr/aarch64-linux-gnu/include/c++/11 ]]; then
  CXXINC_BASE=/usr/aarch64-linux-gnu/include/c++/11
  CXXINC_ARCH=/usr/aarch64-linux-gnu/include/c++/11/aarch64-linux-gnu
elif [[ -d /usr/include/aarch64-linux-gnu/c++/11 ]]; then
  CXXINC_BASE=/usr/include/aarch64-linux-gnu/c++/11
  CXXINC_ARCH=/usr/include/aarch64-linux-gnu/c++/11/aarch64-linux-gnu
else
  echo "Could not find GCC-11 libstdc++ headers. Install:"
  echo "  sudo apt-get install libstdc++-11-dev-arm64-cross"
  echo "or:"
  echo "  sudo dpkg --add-architecture arm64 && sudo apt-get update && sudo apt-get install libstdc++-11-dev:arm64"
  exit 1
fi

cxx=aarch64-linux-gnu-g++-11
cxxflags=(
  -std=gnu++17 -O3 -DNDEBUG -fPIC -shared
  -DORT_API_MANUAL_INIT
  -I"${ONNXRUNTIME_SDK_INCLUDE}"
  -I"${XRT_INC}"
  -I"${TP_JSON}"
  -I"${TP_BASE64}"
  -I"${NN2FPGA_HW_LIB}"
  --sysroot="${SYSROOT}"
  -isystem "${CXXINC_BASE}"
  -isystem "${CXXINC_ARCH}"
  -isystem "${SYSROOT}/usr/include/aarch64-linux-gnu"
)
ldflags=(
  -L"${XRT_LIBDIR}"
  -Wl,-rpath-link,"${SYSROOT}/usr/lib/aarch64-linux-gnu"
  -Wl,--no-undefined
)
libs=(-lxrt_core -lxrt_coreutil -lpthread -ldl)

"$cxx" "${cxxflags[@]}" \
  "${SRC}" "${TP_BASE64}/base64.cpp" \
  -o "${OUT_SO}" \
  "${ldflags[@]}" "${libs[@]}"

aarch64-linux-gnu-strip --strip-unneeded "${OUT_SO}"
echo "Built ${OUT_SO}"