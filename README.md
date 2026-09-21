# HyperVec

> 最新 `main` 同时支持 HTTP 与 gRPC 协议。gRPC 的安装、启动、URI 和 RPC 覆盖说明见 [docs/pyhypervec_grpc_server.md](docs/pyhypervec_grpc_server.md)。

HyperVec is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. HyperVec is written in C++ with complete wrappers for Python/numpy.

## News

See [CHANGELOG.md](CHANGELOG.md) for detailed information about latest features.

## Introduction

HyperVec contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer, and that the vectors can be compared with L2 (Euclidean) distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

Some of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server. Other methods, like HNSW and NSG add an indexing structure on top of the raw vectors to make searching more efficient.

## Installing

HyperVec is mostly implemented in C++. A source build requires a C++20
compiler, CMake 3.16 or newer, OpenMP, BLAS, and LAPACK. The optional Python
extension additionally requires Python development files, NumPy, and SWIG.

### Building from source

The current CMake build provides the portable `generic` CPU target. Other
optimization levels, the MKL-specific switch, and the C API switch are reserved
but not implemented; requesting one fails during configuration instead of
silently producing the wrong build.

The default configuration builds only the core library and does not download
dependencies:

```bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_OPT_LEVEL=generic \
  -DBUILD_TESTING=OFF
cmake --build build/release -j
```

To build the C++ unit tests with an installed GoogleTest package:

```bash
cmake -S . -B build/test \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DHYPERVEC_UNIT_TESTS_ONLY=ON \
  -DHYPERVEC_FETCH_DEPS=OFF
cmake --build build/test -j
ctest --test-dir build/test --output-on-failure
```

If GoogleTest is unavailable, configuration stops with an actionable error.
Set `HYPERVEC_FETCH_DEPS=ON` only when CMake is explicitly allowed to download
the pinned test dependency.

Examples are opt-in and build as normal targets:

```bash
cmake -S . -B build/examples \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_EXTRAS=ON \
  -DBUILD_TESTING=OFF
cmake --build build/examples -j
```

## How HyperVec works

HyperVec is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison. Some index types are simple baselines, such as exact search. Most of the available indexing structures correspond to various trade-offs with respect to

- search time
- search quality
- memory used per index vector
- training time
- adding time
- need for external data for unsupervised training

## Full documentation of HyperVec

The following are entry points for documentation:

- the full documentation can be found on the [wiki page](http://github.com/QY-Graph/hypervec/wiki), including a [tutorial](https://github.com/QY-Graph/hypervec/wiki/Getting-started), a [FAQ](https://github.com/QY-Graph/hypervec/wiki/FAQ) and a [troubleshooting section](https://github.com/QY-Graph/hypervec/wiki/Troubleshooting)
- the [doxygen documentation](https://hypervec.ai/) gives per-class information extracted from code comments
- to reproduce results from our research papers, [Polysemous codes](https://arxiv.org/abs/1609.01882), refer to the [benchmarks README](benchs/README.md). For [Link and code: Fast indexing with graphs and compact regression codes](https://arxiv.org/abs/1804.09996), see the [link_and_code README](benchs/link_and_code)
- the [Collection Data Bundle API](docs/collection_bundle.md) documents the collection export / import / purge flow added for the UltraRAG user-exit scenario

## Authors

HyperVec is developed by the open source community. Contributions are welcome!

## Join the HyperVec community

For public discussion of HyperVec or for questions, visit https://github.com/QY-Graph/hypervec/discussions.

We monitor the [issues page](http://github.com/QY-Graph/hypervec/issues) of the repository.
You can report bugs, ask questions, etc.

## Legal

HyperVec is Mulan Permissive Software License v2 licensed, refer to the [LICENSE file](LICENSE) in the top level directory.

Copyright (c) 2024 HyperVec Authors. All rights reserved.
