ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG HYPERVEC_OPT_LEVEL=generic
ARG http_proxy=
ARG https_proxy=
ARG all_proxy=
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG ALL_PROXY=

ENV DEBIAN_FRONTEND=noninteractive \
    VENV_DIR=/app/venv \
    HYPERVEC_DATA_ROOT=/data/hypervec_data \
    HYPERVEC_HOST=0.0.0.0 \
    HYPERVEC_PORT=8080 \
    HYPERVEC_SERVER=hypercorn \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        make \
        ninja-build \
        swig \
        libopenblas-dev \
        liblapack-dev \
        python3 \
        python3-pip \
        python3-venv \
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
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel packaging cmake numpy tqdm fastapi uvicorn hypercorn h2 && \
    "$VENV_DIR/bin/cmake" -S . -B build-docker -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DHYPERVEC_ENABLE_PYTHON=ON \
      -DHYPERVEC_ENABLE_EXTRAS=OFF \
      -DHYPERVEC_OPT_LEVEL="$HYPERVEC_OPT_LEVEL" \
      -DPython_EXECUTABLE="$VENV_DIR/bin/python" && \
    "$VENV_DIR/bin/cmake" --build build-docker --target swighypervec -j"$(nproc)" && \
    cd build-docker/src/python && \
    "$VENV_DIR/bin/python" -m pip wheel . --no-deps -w dist && \
    "$VENV_DIR/bin/python" -m pip install --force-reinstall dist/*.whl && \
    cd /app/hypervector && \
    "$VENV_DIR/bin/python" -m pip install ./pyhypervec && \
    mkdir -p "$HYPERVEC_DATA_ROOT"

EXPOSE 8080

CMD mkdir -p "$HYPERVEC_DATA_ROOT" && \
    exec "$VENV_DIR/bin/python" -m hypervec.hypervec_http_server \
    --data-root "$HYPERVEC_DATA_ROOT" \
    --host "$HYPERVEC_HOST" \
    --port "$HYPERVEC_PORT" \
    --server "$HYPERVEC_SERVER"
