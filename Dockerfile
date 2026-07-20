ARG BASE_IMAGE=docker.m.daocloud.io/library/ubuntu:20.04
FROM ${BASE_IMAGE}

ARG HYPERVEC_OPT_LEVEL=generic

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON310_DIR=/opt/python3.10 \
    VENV_DIR=/app/venv \
    HYPERVEC_OPT_LEVEL=${HYPERVEC_OPT_LEVEL} \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4 \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY=

RUN if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        ca-certificates curl wget bzip2 xz-utils build-essential make ninja-build swig \
        python3 python3-dev python3-venv python3-pip \
        libopenblas-dev liblapack-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev && \
      rm -rf /var/lib/apt/lists/*; \
    elif command -v dnf >/dev/null 2>&1; then \
      dnf install -y \
        ca-certificates curl wget bzip2 xz make ninja-build swig gcc gcc-c++ \
        python3 python3-devel python3-pip \
        openblas-devel lapack-devel openssl-devel zlib-devel bzip2-devel readline-devel \
        sqlite-devel libffi-devel xz-devel tk-devel libuuid-devel && \
      dnf clean all; \
    else \
      echo "Neither apt-get nor dnf found" >&2; exit 1; \
    fi

RUN mkdir -p /tmp && chmod 1777 /tmp && cd /tmp && \
    curl -fsSLO https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --prefix="$PYTHON310_DIR" --enable-shared --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make install && \
    echo "$PYTHON310_DIR/lib" > /etc/ld.so.conf.d/python310.conf && \
    ldconfig && \
    "$PYTHON310_DIR/bin/python3.10" --version && \
    rm -rf /tmp/Python-3.10.14 /tmp/Python-3.10.14.tgz

WORKDIR /app/hypervector
COPY . /app/hypervector

RUN ARCH="$(uname -m)"; \
    case "$ARCH" in \
      aarch64|arm64) echo "ARM platform: Kunpeng/Phytium compatible" ;; \
      x86_64|amd64) echo "x86 platform: Intel/Hygon compatible" ;; \
      *) echo "Unknown platform: $ARCH" ;; \
    esac

RUN "$PYTHON310_DIR/bin/python3.10" -m venv "$VENV_DIR" && \
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel packaging cmake==3.28.3 numpy tqdm fastapi uvicorn hypercorn h2 grpcio protobuf && \
    "$VENV_DIR/bin/cmake" -S . -B build-docker -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DHYPERVEC_ENABLE_PYTHON=ON \
      -DHYPERVEC_ENABLE_EXTRAS=OFF \
      -DHYPERVEC_OPT_LEVEL=${HYPERVEC_OPT_LEVEL} \
      -DPython_EXECUTABLE="$VENV_DIR/bin/python" && \
    "$VENV_DIR/bin/cmake" --build build-docker --target swighypervec -j"$(nproc)" && \
    if [ "$HYPERVEC_OPT_LEVEL" = "sve" ]; then \
      "$VENV_DIR/bin/cmake" --build build-docker --target swighypervec_sve -j"$(nproc)"; \
    fi && \
    cd build-docker/src/python && \
    "$VENV_DIR/bin/python" setup.py bdist_wheel && \
    "$VENV_DIR/bin/python" -m pip install --force-reinstall dist/*.whl && \
    cd /app/hypervector && \
    "$VENV_DIR/bin/python" -m pip install ./pyhypervec && \
    mkdir -p /data/hypervec_data

EXPOSE 8080 50052

CMD ["/app/venv/bin/python", "-m", "hypervec.hypervec_grpc_server", "--data-root", "/data/hypervec_data", "--host", "0.0.0.0", "--port", "50052", "--workers", "64"]
