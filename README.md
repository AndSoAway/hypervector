# HyperVec

HyperVec is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. HyperVec is written in C++ with complete wrappers for Python/numpy.

## News

See [CHANGELOG.md](CHANGELOG.md) for detailed information about latest features.

## Introduction

HyperVec contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer, and that the vectors can be compared with L2 (Euclidean) distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

Some of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server. Other methods, like HNSW and NSG add an indexing structure on top of the raw vectors to make searching more efficient.

## Installing

HyperVec comes with precompiled libraries for Anaconda in Python, see [hypervec-cpu](https://anaconda.org/pytorch/hypervec-cpu). The library is mostly implemented in C++, the only dependency is a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. The Python interface is also optional. It compiles with cmake. See [INSTALL.md](INSTALL.md) for details.

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

## Authors

HyperVec is developed by the open source community. Contributions are welcome!

## Join the HyperVec community

For public discussion of HyperVec or for questions, visit https://github.com/QY-Graph/hypervec/discussions.

We monitor the [issues page](http://github.com/QY-Graph/hypervec/issues) of the repository.
You can report bugs, ask questions, etc.

## Legal

HyperVec is Mulan Permissive Software License v2 licensed, refer to the [LICENSE file](LICENSE) in the top level directory.

Copyright (c) 2024 HyperVec Authors. All rights reserved.
