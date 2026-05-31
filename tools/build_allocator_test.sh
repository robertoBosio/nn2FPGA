#!/usr/bin/env bash
set -euo pipefail

: "${ONNXRUNTIME_SDK_INCLUDE:=/opt/onnxruntime-sdk/include}"
: "${ONNXRUNTIME_LIBDIR:=/opt/onnxruntime-sdk/lib}"
: "${CXX:=g++}"

HOST_ARCH=$(uname -m)
if [[ -z "${TARGET_ARCH:-}" ]]; then
  case "$(basename "${CXX}")" in
    aarch64-*|arm64-*) TARGET_ARCH=aarch64 ;;
    arm-*|armhf-*) TARGET_ARCH=arm ;;
    *) TARGET_ARCH=${HOST_ARCH} ;;
  esac
fi
OUT_ROOT=${1:-artifacts/allocator_test}
OUT_DIR=${OUT_ROOT}/${TARGET_ARCH}
OUT_BIN=${OUT_DIR}/test_ort_cpu_allocator
OUT_RUNNER=${OUT_DIR}/run_ort_with_allocator
OUT_PROBE_SO=${OUT_DIR}/liballocator_probe_op.so
MODEL_PATH=${OUT_ROOT}/allocator_test.onnx
PROBE_MODEL_PATH=${OUT_ROOT}/allocator_probe.onnx

mkdir -p "${OUT_DIR}"

cxxflags=(
  -std=gnu++17 -O2 -DNDEBUG
  -I"${ONNXRUNTIME_SDK_INCLUDE}"
  -Inn2fpga/hw/operator_runtime
)
ldflags=(
  -L"${ONNXRUNTIME_LIBDIR}"
  -Wl,-rpath,"${ONNXRUNTIME_LIBDIR}"
)
runner_libs=(-Wl,-export-dynamic -lonnxruntime -pthread -ldl)

if [[ -n "${SYSROOT:-}" ]]; then
  # Locate GCC-11 libstdc++ headers from the host toolchain, matching
  # tools/build_customop.sh for board cross-compilation.
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

  cxxflags+=(
    --sysroot="${SYSROOT}"
    -isystem "${CXXINC_BASE}"
    -isystem "${CXXINC_ARCH}"
    -isystem "${SYSROOT}/usr/include/aarch64-linux-gnu"
  )
  ldflags+=(
    -Wl,-rpath-link,"${SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
fi

: "${XRT_INC:=${SYSROOT:-}/usr/include/xrt}"
: "${XRT_LIBDIR:=${SYSROOT:-}/usr/lib/aarch64-linux-gnu}"
: "${TARGET_LIBDIR:=${SYSROOT:-}/lib/aarch64-linux-gnu}"
: "${XRT_EXTRA_LIBDIRS:=}"
: "${BOOST_FILESYSTEM_LIB:=libboost_filesystem.so}"
: "${BOOST_SYSTEM_LIB:=libboost_system.so}"
: "${UUID_LIB:=libuuid.so}"
cxxflags+=(
  -I"${XRT_INC}"
)
ldflags+=(
  -L"${XRT_LIBDIR}"
  -Wl,-rpath-link,"${XRT_LIBDIR}"
  -Wl,-rpath-link,"${TARGET_LIBDIR}"
)
IFS=: read -r -a extra_libdirs <<< "${XRT_EXTRA_LIBDIRS}"
for libdir in "${extra_libdirs[@]}"; do
  if [[ -n "${libdir}" ]]; then
    ldflags+=(
      -L"${libdir}"
      -Wl,-rpath-link,"${libdir}"
    )
  fi
done
runner_libs+=(
  -lxrt_core
  -lxrt_coreutil
  -l:"${BOOST_FILESYSTEM_LIB}"
  -l:"${BOOST_SYSTEM_LIB}"
  -l:"${UUID_LIB}"
)

echo "Building allocator test"
echo "  host arch:        ${HOST_ARCH}"
echo "  target arch:      ${TARGET_ARCH}"
echo "  compiler:         ${CXX}"
echo "  sysroot:          ${SYSROOT:-<none>}"
echo "  allocator backend:XRT BO mapped memory"
echo "  ORT include dir:  ${ONNXRUNTIME_SDK_INCLUDE}"
echo "  ORT library dir:  ${ONNXRUNTIME_LIBDIR}"
echo "  XRT include dir:  ${XRT_INC}"
echo "  XRT library dir:  ${XRT_LIBDIR}"
echo "  target lib dir:   ${TARGET_LIBDIR}"
echo "  extra lib dirs:   ${XRT_EXTRA_LIBDIRS:-<none>}"
echo "  boost filesystem: ${BOOST_FILESYSTEM_LIB}"
echo "  boost system:     ${BOOST_SYSTEM_LIB}"
echo "  uuid:             ${UUID_LIB}"
echo "  output:           ${OUT_BIN}"
echo "  generic runner:   ${OUT_RUNNER}"
echo "  probe op:         ${OUT_PROBE_SO}"

"${CXX}" "${cxxflags[@]}" \
  tools/test_ort_cpu_allocator.cpp \
  nn2fpga/hw/operator_runtime/nn2FPGA_allocator.cpp \
  "${ldflags[@]}" \
  "${runner_libs[@]}" \
  -o "${OUT_BIN}"

"${CXX}" "${cxxflags[@]}" \
  tools/run_ort_with_allocator.cpp \
  nn2fpga/hw/operator_runtime/nn2FPGA_allocator.cpp \
  "${ldflags[@]}" \
  "${runner_libs[@]}" \
  -o "${OUT_RUNNER}"

"${CXX}" "${cxxflags[@]}" -fPIC -shared -DORT_API_MANUAL_INIT \
  tools/allocator_probe_op.cpp \
  -pthread -ldl \
  -o "${OUT_PROBE_SO}"

if command -v file >/dev/null 2>&1; then
  file "${OUT_BIN}"
  file "${OUT_RUNNER}"
  file "${OUT_PROBE_SO}"
fi

echo "Built ${OUT_BIN}"
echo "Built ${OUT_RUNNER}"
echo "Built ${OUT_PROBE_SO}"
echo "Generate model: python3 tools/make_allocator_test_model.py ${MODEL_PATH}"
echo "Run test:      ${OUT_BIN} ${MODEL_PATH} 3"
echo "Generate probe model: python3 tools/make_allocator_probe_model.py ${PROBE_MODEL_PATH}"
echo "Run probe test:      ${OUT_BIN} ${PROBE_MODEL_PATH} 3 ${OUT_PROBE_SO}"
echo "Run generic model:   ${OUT_RUNNER} <model.onnx> <custom_op.so> [runs] [dynamic_batch]"
