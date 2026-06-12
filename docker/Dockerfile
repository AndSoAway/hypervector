# HyperVector Docker Image for Intel x86_64
# Based on Ubuntu 22.04 with Python 3.10 and full dependencies

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    swig \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    libsqlite3-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default (has sqlite3 support)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Create working directory
WORKDIR /app/hypervector

# Copy project files
COPY . .

# Create virtual environment and install Python dependencies
RUN python3 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel packaging numpy && \
    pip install "cmake>=3.24" && \
    pip install fastapi uvicorn hypercorn h2 httpx

# Build HyperVec C++ core and Python bindings
RUN . /app/venv/bin/activate && \
    mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DHYPERVEC_OPT_LEVEL=generic \
        -DHYPERVEC_ENABLE_PYTHON=ON \
        -DHYPERVEC_ENABLE_EXTRAS=OFF \
        -DBUILD_TESTING=OFF && \
    make -j$(nproc) swighypervec hypervec_python_callbacks hypervec

# Install hypervec Python package
RUN . /app/venv/bin/activate && \
    cd build/src/python && \
    cp _swighypervec.so ../../.. && \
    cp swighypervec.py ../../.. && \
    cd /app/hypervector/src/python && \
    cp /app/hypervector/build/src/python/_swighypervec.so . && \
    cp /app/hypervector/build/src/python/swighypervec.py . && \
    pip install -e .

# Install pyhypervec client
RUN . /app/venv/bin/activate && \
    cd /app/hypervector/pyhypervec && \
    pip install -e .

# Create data directory
RUN mkdir -p /data/hypervec_data

# Expose HTTP server port
EXPOSE 8080

# Set environment variables
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="/app/hypervector/src/python:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: start HTTP server
CMD ["/app/venv/bin/python", "-m", "hypervec.hypervec_http_server", \
     "--data-root", "/data/hypervec_data", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--server", "hypercorn"]
