ARG BASE_IMAGE=jiuyuansoft/base_image:arm
FROM ${BASE_IMAGE}

ARG HYPERVEC_OPT_LEVEL=sve

ENV DEBIAN_FRONTEND=noninteractive \
    VENV_DIR=/app/venv \
    HYPERVEC_OPT_LEVEL=${HYPERVEC_OPT_LEVEL} \
    PYTHONUNBUFFERED=1 \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY=

RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 8.8.4.4" >> /etc/resolv.conf

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        bzip2 \
        xz-utils \
        build-essential \
        make \
        ninja-build \
        swig \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        libopenblas-dev \
        liblapack-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        tk-dev \
        uuid-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/hypervector
COPY . /app/hypervector

RUN ARCH="$(uname -m)"; \
    case "$ARCH" in \
      aarch64|arm64) echo "ARM platform: Kunpeng/Phytium compatible" ;; \
      x86_64|amd64) echo "x86 platform: Intel/Hygon compatible" ;; \
      *) echo "Unknown platform: $ARCH" ;; \
    esac

RUN python3 -m venv "$VENV_DIR" && \
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel packaging cmake==3.28.3 numpy tqdm fastapi uvicorn hypercorn h2 grpcio protobuf && \
    "$VENV_DIR/bin/cmake" -S . -B build-docker -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DHYPERVEC_ENABLE_PYTHON=ON \
      -DHYPERVEC_ENABLE_EXTRAS=OFF \
      -DHYPERVEC_OPT_LEVEL=${HYPERVEC_OPT_LEVEL} \
      -DPython_EXECUTABLE="$VENV_DIR/bin/python" && \
    SWIG_TARGET="swighypervec_sve" && \
    if [ "${HYPERVEC_OPT_LEVEL}" = "avx512" ]; then SWIG_TARGET="swighypervec_avx512"; fi && \
    if [ "${HYPERVEC_OPT_LEVEL}" = "avx512_spr" ]; then SWIG_TARGET="swighypervec_avx512_spr"; fi && \
    if [ "${HYPERVEC_OPT_LEVEL}" = "avx2" ]; then SWIG_TARGET="swighypervec_avx2"; fi && \
    if [ "${HYPERVEC_OPT_LEVEL}" = "generic" ]; then SWIG_TARGET="swighypervec"; fi && \
    echo "Building ${SWIG_TARGET} for HYPERVEC_OPT_LEVEL=${HYPERVEC_OPT_LEVEL}" && \
    "$VENV_DIR/bin/cmake" --build build-docker --target "${SWIG_TARGET}" -j"$(nproc)" && \
    cd build-docker/src/python && \
    "$VENV_DIR/bin/python" setup.py bdist_wheel && \
    "$VENV_DIR/bin/python" -m pip install --force-reinstall dist/*.whl && \
    cd /app/hypervector && \
    "$VENV_DIR/bin/python" -m pip install ./pyhypervec && \
    mkdir -p /data/hypervec_data

EXPOSE 8080

CMD ["/app/venv/bin/python", "-m", "hypervec.hypervec_http_server", "--data-root", "/data/hypervec_data", "--host", "0.0.0.0", "--port", "8080", "--server", "hypercorn"]
