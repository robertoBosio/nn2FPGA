#!/usr/bin/env bash
set -e

# ----- Configuration -----
IMAGE_NAME="nn2fpga-container-image"
WORKSPACE_DIR="$(pwd)/.."
USERNAME="$(whoami)"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
XILINX_DIR="/tools"
XRT_DIR="/opt/xilinx/xrt"
XILINX_VERSION="2025.1"
DATASET_DIR="/home-ssd/datasets"
WORKSPACE_ROOT_DIR="/workspace"
CONTAINER_NAME="nn2fpga-container-${USERNAME}"

# Optional local overrides (untracked)
LOCAL_CONFIG="$(dirname "$0")/.run.local.sh"
if [ -f "${LOCAL_CONFIG}" ]; then
    # shellcheck disable=SC1090
    source "${LOCAL_CONFIG}"
fi

# Per-user home dir for the container (on the host)
DOCKER_HOME_DIR="$(pwd)/.docker_home_${USERNAME}"
mkdir -p "${DOCKER_HOME_DIR}"

# ----- Build image if not already built -----
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Building Docker image '${IMAGE_NAME}'..."
    docker build \
        --file docker/Dockerfile \
        --build-arg NN2FPGA_ROOT_DIR="${WORKSPACE_ROOT_DIR}/NN2FPGA" \
        -t "${IMAGE_NAME}" .
fi

# Optional local-only mounts
EXTRA_DOCKER_ARGS=()
if [ -n "${WORK_DIR:-}" ]; then
    EXTRA_DOCKER_ARGS+=(
        -v "${WORK_DIR}:${WORKSPACE_ROOT_DIR}/NN2FPGA/work"
    )
fi

# ----- Run the container -----
docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    -v "${WORKSPACE_DIR}:${WORKSPACE_ROOT_DIR}" \
    -v "${XILINX_DIR}:${XILINX_DIR}" \
    -v "${XRT_DIR}:${XRT_DIR}" \
    -v "${DATASET_DIR}:/home/datasets" \
    -v "${DOCKER_HOME_DIR}:/home/${USERNAME}" \
    "${EXTRA_DOCKER_ARGS[@]}" \
    --network=host \
    --memory 80g \
    --env NN2FPGA_ROOT_DIR="${WORKSPACE_ROOT_DIR}/NN2FPGA" \
    --env XILINX_DIR="${XILINX_DIR}" \
    --env XRT_DIR="${XRT_DIR}" \
    --env XILINX_VERSION="${XILINX_VERSION}" \
    --env USER_NAME="${USERNAME}" \
    --env USER_ID="${USER_ID}" \
    --env GROUP_ID="${GROUP_ID}" \
    "${IMAGE_NAME}"