FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ARG NN2FPGA_ROOT_DIR

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PROMPT_DIRTRIM=1

# Install system packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 vim curl wget git build-essential \
    zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev liblzma-dev libncurses5-dev zip time udev \
    libudev-dev python3 python3-pip python3-venv cmake libgtk2.0-dev \
    pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    locales libncurses5 libtinfo5 libncursesw5-dev unzip gosu \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    update-locale LANG=en_US.UTF-8

# Set workspace and environment variables
WORKDIR ${NN2FPGA_ROOT_DIR}

# Copy and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# ---- Cross toolchain & helpers ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-11-aarch64-linux-gnu g++-11-aarch64-linux-gnu libstdc++-11-dev-arm64-cross ninja-build rsync file \
    && rm -rf /var/lib/apt/lists/*

# ---- Install ONNX Runtime headers only (no lib) ----
ARG ONNXRT_VER=1.15.0
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRT_VER}/onnxruntime-linux-aarch64-${ONNXRT_VER}.tgz" \
    && tar -xzf "onnxruntime-linux-aarch64-${ONNXRT_VER}.tgz" -C /opt/ \
    && mv "/opt/onnxruntime-linux-aarch64-${ONNXRT_VER}" /opt/onnxruntime-sdk
ENV ONNXRUNTIME_SDK_INCLUDE=/opt/onnxruntime-sdk/include

# ---- Bake the AArch64 sysroot into the image ----
ADD docker/sysroot-aarch64.tar.zst /opt/sysroots/
RUN set -e; \
    cd /opt/sysroots && \
    mv sysroot-aarch64 sysroot-aarch64.raw || true && \
    mv sysroot-aarch64.raw board
ENV SYSROOT=/opt/sysroots/board

# Entrypoint
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/xilinx_entrypoint.sh /usr/local/bin/xilinx_entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/xilinx_entrypoint.sh

# Run as root by default; user/uid is set at runtime by docker run
USER root

ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]

