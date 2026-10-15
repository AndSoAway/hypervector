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

The default `generic` build uses portable compiler-generated code (which may
include baseline SIMD). On x86-64 with GCC/Clang, `HYPERVEC_OPT_LEVEL=dd` adds
AVX2/FMA and (when supported by the compiler) AVX-512 FP32 distance kernels to
the same library, selected only when CPU and OS support them. Other optimization
levels, the MKL-specific switch, and the C API switch remain unavailable.

The default configuration builds only the core library and does not download
dependencies:

```bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_OPT_LEVEL=generic \
  -DBUILD_TESTING=OFF
cmake --build build/release -j
```

To enable runtime SIMD dispatch, use `-DHYPERVEC_OPT_LEVEL=dd` in a separate
build directory. Only isolated kernel objects receive AVX2/FMA or AVX-512 flags;
do not add global `-march=native` flags to a portable build. C++ and Python
use the same dispatch; no separate AVX Python module is needed.

In a `dd` build, `HYPERVEC_SIMD_LEVEL=NONE` forces the baseline distance
implementation, `HYPERVEC_SIMD_LEVEL=AVX2` requests AVX2/FMA, and
`HYPERVEC_SIMD_LEVEL=AVX512` requests AVX-512F/DQ/BW/VL (also requiring AVX2/FMA).
Unsupported or invalid requests raise an exception on first dispatch, rather than execute
unsupported instructions. Leave the variable unset to retain automatic
AVX2/FMA selection with a generic fallback. AVX-512 is opt-in in this release:
kernel speedups do not necessarily translate into faster end-to-end searches.
This differs from `HYPERVEC_OPT_LEVEL`, which controls the build (and the
legacy Python module loader), not runtime kernel selection.

C++ callers can inspect `hypervec::SIMDConfig::get_level_name()` from
`<utils/simd/simd_levels.h>`. Set the environment before using the library;
existing distance computers retain their selected kernel. Runtime SIMD changes
neither the index format nor its parameters. Floating-point accumulation order
may change slightly. LVQ8 also uses an AVX2 distance kernel, selected once per
distance computer or scan, shared by LVQ, IVF-LVQ and HNSW-LVQ. LVQ1–7 retain
their generic bit-packed implementation. PQ4/PQ8/PQ16 table scans use portable
bit-width-specialized kernels with independent accumulators, selected once per
scanner (also effective in `generic` builds). Other PQ widths retain the generic
decoder. RaBitQ table scans also use portable independent accumulators, retaining
double-precision tables and estimates. AVX-512 specializes FP32 L2, inner
product, squared norm and four-vector batches; short vectors below 32 dimensions
retain AVX2. LVQ8 continues to select its AVX2 kernel even at the AVX-512 level;
PQ/RaBitQ code scans are unchanged. No AVX-512-specific quantizer kernels or
AVX512_SPR level are enabled. Performance depends on CPU and workload; use the
runtime override to compare against AVX2 on the same indices.

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

### Using an installed CMake package

Install HyperVec into a prefix, then point a separate CMake project at that
prefix:

```bash
cmake --install build/release --prefix /path/to/hypervec-install
cmake -S /path/to/consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH=/path/to/hypervec-install
```

Consumers use the namespaced target instead of copying HyperVec link flags:

```cmake
find_package(hypervec 1.14 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE hypervec::hypervec)
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
