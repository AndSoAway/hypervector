# ARM 环境从源码编译 HyperVector Python 库并使用本地 Backend

本文档说明如何在 ARM/Linux 环境从零编译 HyperVector，生成 Python 可导入的 `hypervec` 包，并使用 `HyperVectorIndexBackend` 完成本地向量与标量数据检索。该 backend 不依赖 `pymilvus`，不提供 server 能力，所有数据保存在本地文件系统。

## 1. 环境假设

目标环境：

- Linux ARM64 / AArch64，例如 Ubuntu 22.04、Debian、麒麟、统信等
- Python 3.10+
- CMake 3.24+
- C++20 编译器
- BLAS/LAPACK
- SWIG

检查架构：

```bash
uname -m
python3 --version
cmake --version
```

`uname -m` 应返回 `aarch64` 或类似 ARM64 标识。

## 2. 安装系统依赖

Ubuntu/Debian 示例：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  swig \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  libopenblas-dev \
  liblapack-dev \
  libomp-dev
```

如果发行版没有 `libomp-dev`，可先只安装 OpenMP 编译器支持；GCC 通常通过 `libgomp` 提供 OpenMP。

## 3. 获取源码

```bash
git clone <your-hypervector-repo-url> hypervector
cd hypervector
git checkout hypervector_backend
```

如果你已经有源码目录，直接进入源码根目录即可。

## 4. 创建 Python 虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging numpy
```

可选安装 demo 进度条依赖：

```bash
python -m pip install tqdm
```

## 5. 配置 CMake

推荐使用 generic 或 SVE 构建：

```bash
cmake -S . -B build-arm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_PYTHON=ON \
  -DHYPERVEC_ENABLE_EXTRAS=OFF \
  -DHYPERVEC_OPT_LEVEL=generic
```

如果目标 ARM CPU 明确支持 SVE，可以尝试：

```bash
cmake -S . -B build-arm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_PYTHON=ON \
  -DHYPERVEC_ENABLE_EXTRAS=OFF \
  -DHYPERVEC_OPT_LEVEL=sve
```

建议先用 `generic` 打通编译与运行，再评估是否启用 `sve`。

## 6. 编译 C++ 库和 Python SWIG 模块

```bash
cmake --build build-arm --target hypervec -j"$(nproc)"
cmake --build build-arm --target swighypervec -j"$(nproc)"
```

如果使用 `HYPERVEC_OPT_LEVEL=sve`，还需要构建对应目标：

```bash
cmake --build build-arm --target swighypervec_sve -j"$(nproc)"
```

## 7. 构建并安装 Python 包

进入 CMake 生成的 Python 构建目录：

```bash
cd build-arm/src/python
python setup.py bdist_wheel
python -m pip install --force-reinstall dist/*.whl
```

安装后检查：

```bash
python - <<'PY'
import hypervec
print("hypervec version:", hypervec.__version__)
print("has backend:", hasattr(hypervec, "HyperVectorIndexBackend"))
PY
```

若 `has backend: True`，说明 Python 包已包含本地 backend。

## 8. 本地 Backend 数据布局

`HyperVectorIndexBackend` 使用本地目录模拟 collection：

```text
./hv_backend_data/
  demo_collection/
    manifest.json
    index.hypervec
    rows.jsonl
```

- `index.hypervec`：HyperVector 向量索引
- `rows.jsonl`：文本、外部 ID、metadata
- `manifest.json`：collection 配置、维度、metric、索引参数

## 9. 运行 Demo

从源码根目录运行：

```bash
python examples/python/demo_hypervector_backend.py
```

demo 会：

1. 构造 5 条二维向量和对应文本/metadata
2. 使用 `HyperVectorIndexBackend.build_index()` 写入本地 collection
3. 使用 `search()` 返回文本列表
4. 使用 `search_with_meta()` 返回结构化结果
5. 使用 filter 查询指定 metadata
6. 重新加载本地 collection 验证持久化可用

## 10. 最小使用示例

```python
import logging
import numpy as np
from hypervec import HyperVectorIndexBackend

backend = HyperVectorIndexBackend(
    contents=[],
    config={
        "uri": "./hv_backend_data",
        "collection_name": "demo",
        "metric_type": "L2",
        "index_params": {"index_type": "HNSWFlat", "M": 32},
    },
    logger=logging.getLogger("hypervector-demo"),
)

backend.build_index(
    embeddings=np.random.rand(100, 128).astype("float32"),
    ids=np.array([f"chunk-{i}" for i in range(100)]),
    contents=[f"document chunk {i}" for i in range(100)],
    metadatas=[{"source": "demo", "doc_id": str(i // 10)} for i in range(100)],
    overwrite=True,
)

docs = backend.search(np.random.rand(1, 128).astype("float32"), top_k=5)
rows = backend.search_with_meta(
    np.random.rand(1, 128).astype("float32"),
    top_k=5,
    filter="source == 'demo'",
)
```

## 11. 注意事项

- MVP 版本是本地文件型 backend，不支持远程 server。
- filter 目前只支持 `field == 'value'` 和 `AND` 连接。
- 多进程并发写同一个 collection 暂不支持。
- 如果 Python 绑定未暴露 `IndexHNSWLVQ`，请先使用 `HNSWFlat` 或 `Flat`。
- `metric_type="IP"` 时分数来自 HyperVector 返回值；业务层如需统一“越大越好/越小越好”，应在上层约定。

