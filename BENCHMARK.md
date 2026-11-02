# HNSW 并行构建 — 编译与基准测试

本分支 (`feat/hnsw-parallel-build`) 将 `IndexHNSW::Add` 的图构建从串行改为
OpenMP 按 level 分批并行。本文档说明如何编译并复现基准测试。

## 1. 编译核心库

依赖：C++20 编译器、CMake、OpenMP、OpenBLAS/LAPACK。

```bash
# 方式 A：一键脚本（缺依赖时自动用 Miniconda 补齐）
./build.sh

# 方式 B：手动 CMake（已有系统编译器时更快）
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DHYPERVEC_ENABLE_EXTRAS=ON
make -j"$(nproc)" hypervec      # 产物: build/src/libhypervec.a
```

> 系统 g++ < 10 时用 `-std=c++2a` 代替 `-std=c++20`。

## 2. 控制构建线程数

并行度由 OpenMP 环境变量控制，无需改代码或重编：

```bash
OMP_NUM_THREADS=64 ./your_bench     # 并行构建
OMP_NUM_THREADS=1  ./your_bench     # 单线程基线（对照）
```

## 3. 基准测试工具

示例测试程序（独立编译，链接上面的静态库）：

```bash
g++ -std=c++2a -O3 -fopenmp \
  -I <repo>/src/include -I <repo>/src \
  bench.cpp <repo>/build/src/libhypervec.a \
  -lopenblas -llapack -o bench
```

测试内容：构建墙钟时间、Recall@k、检索 QPS / 延迟（扫 `ef_search`）。
索引可用 `WriteIndex` / `ReadIndex`（`persistence/index_io.h`）存盘后重复加载，
避免每次重建。

## 4. 数据集路径

测试程序在源码顶部用常量写死数据集路径，按环境修改即可。格式：

| 格式 | 头部 | 数据 | 用途 |
|------|------|------|------|
| `.fbin`  | `int32 n`, `int32 d` | `float32[n*d]` | base / query 向量 |
| `.u8bin` | `int32 n`, `int32 d` | `uint8[n*d]`   | base / query（加载时转 float） |
| `.ibin`  | `int32 n`, `int32 k` | `int32[n*k]`   | groundtruth 近邻 id |

示例数据集路径（本机）：

```
Wiki 1M (768d, L2):  /root/vector/data/wiki_all_1M/{base.1M.fbin,queries.fbin,groundtruth.1M.neighbors.ibin}
SIFT 10M (128d, L2): /data/gudongtian/datasets/sift10m/{base.10M.u8bin,query.public.10K.u8bin,groundtruth.public.10K.bin}
```

> 子集测试（只取前 N 条 base）时，Recall 需对应子集的 groundtruth；
> 用完整 groundtruth 配子集 base 会得到偏低的召回，属预期。

## 5. 实测结果（64 核）

| 数据集 | 规模 × 维度 | 并行构建 (64T) | 最佳 Recall@10 |
|--------|------------|----------------|----------------|
| Wiki 1M  | 1M × 768  | 44.8 s  | 0.970 (ef=512) |
| SIFT 10M | 10M × 128 | 331.6 s | 0.9975 (ef=256) |

Wiki 1M 单线程基线 1062 s → 并行 44.8 s，**加速约 23.7×，召回无损**
(0.9771 → 0.9760)。
