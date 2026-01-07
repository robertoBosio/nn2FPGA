FROM ubuntu:22.04

ARG NN2FPGA_ROOT_DIR

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PROMPT_DIRTRIM=1

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget git gosu \
      python3 python3-pip python3-venv \
      build-essential make \
      zip unzip xz-utils \
      file patch rsync \
      libncurses5 libtinfo5 \
      locales time \
	  libxml2 libxslt1.1 \
	  tcl8.6 \
	  libx11-6 \
	  libusb-1.0-0 \
	  libxerces-c3.2 \
	  libgoogle-perftools4 \
      libpixman-1-0 \
      xvfb \
      libnss3 \
      libgdk-pixbuf2.0-0 \
      libgtk-3-0 \
      libxss1 \
      libsecret-1-0 \
      libasound2 \
      net-tools graphviz fdisk \
      zlib1g-dev \ 
    && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    update-locale LANG=en_US.UTF-8

WORKDIR ${NN2FPGA_ROOT_DIR}

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# ---- Cross toolchain & helpers (keep only if you need ARM target builds) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-11-aarch64-linux-gnu g++-11-aarch64-linux-gnu \
    libstdc++-11-dev-arm64-cross \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ---- Install ONNX Runtime headers only (keep only if build needs them) ----
ARG ONNXRT_VER=1.15.0
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRT_VER}/onnxruntime-linux-aarch64-${ONNXRT_VER}.tgz" \
    && tar -xzf "onnxruntime-linux-aarch64-${ONNXRT_VER}.tgz" -C /opt/ \
    && mv "/opt/onnxruntime-linux-aarch64-${ONNXRT_VER}" /opt/onnxruntime-sdk
ENV ONNXRUNTIME_SDK_INCLUDE=/opt/onnxruntime-sdk/include

ADD docker/sysroot-aarch64.tar.zst /opt/sysroots/
RUN set -e; \
    cd /opt/sysroots && \
    mv sysroot-aarch64 sysroot-aarch64.raw || true && \
    mv sysroot-aarch64.raw board
ENV SYSROOT=/opt/sysroots/board

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/xilinx_entrypoint.sh /usr/local/bin/xilinx_entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/xilinx_entrypoint.sh

USER root
ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]
