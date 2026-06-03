FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        make \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/hypervector

WORKDIR /workspace/hypervector

RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    make -j$(nproc) hypervec

WORKDIR /workspace/hypervector/build
