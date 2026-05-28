# ARM 环境从源码编译 HyperVector Python 库并使用本地 Backend

本文说明如何在 ARM/Linux 环境从源码编译 HyperVector，生成可导入的 `hypervec` Python 包，并使用 `HyperVectorIndexBackend` 完成本地向量与标量数据检索。当前 Python 绑定是面向替换 `milvus_backend.py` 的 MVP 绑定，暴露后端需要的最小索引能力，不是完整 C++ API 的 Python 映射。

## 1. 环境假设

目标环境：

- Linux ARM64 / AArch64，例如 Ubuntu、Debian、麒麟、统信等
- Python 3.10+
- CMake 3.24+
- C++20 编译器
- BLAS/LAPACK
- SWIG

检查架构和工具版本：

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

如果发行版没有 `libomp-dev`，可先只使用 GCC 自带的 OpenMP runtime，通常由 `libgomp` 提供。

## 3. 获取源码

```bash
git clone <your-hypervector-repo-url> hypervector
cd hypervector
git checkout hypervector_backend
```

如果已有源码目录，直接进入仓库根目录即可。

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

建议先使用 `generic` 打通编译和运行：

```bash
cmake -S . -B build-arm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_PYTHON=ON \
  -DHYPERVEC_ENABLE_EXTRAS=OFF \
  -DHYPERVEC_OPT_LEVEL=generic \
  -DPython_EXECUTABLE="$(which python)"
```

如果目标 ARM CPU 明确支持 SVE，可后续尝试：

```bash
cmake -S . -B build-arm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_PYTHON=ON \
  -DHYPERVEC_ENABLE_EXTRAS=OFF \
  -DHYPERVEC_OPT_LEVEL=sve \
  -DPython_EXECUTABLE="$(which python)"
```

注意：`Python_EXECUTABLE` 应指向安装了 `numpy` 且包含开发头文件的 Python 环境。

## 6. 编译 C++ 库和 Python SWIG 模块

```bash
cmake --build build-arm --target hypervec -j"$(nproc)"
cmake --build build-arm --target swighypervec -j"$(nproc)"
```

如果配置了 `HYPERVEC_OPT_LEVEL=sve`，再构建对应目标：

```bash
cmake --build build-arm --target swighypervec_sve -j"$(nproc)"
```

当前 `swighypervec` 暴露的关键 Python 能力包括：

- `IndexFlatIP`
- `IndexFlatL2`
- `IndexHNSWFlat`
- `IndexLVQ`
- `IndexIVFLVQ`
- `IndexHNSWLVQ`
- `ReadIndex` / `WriteIndex`
- `HyperVectorIndexBackend`

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
print("package:", hypervec.__file__)
print("has backend:", hasattr(hypervec, "HyperVectorIndexBackend"))
print("has HNSW:", hasattr(hypervec, "IndexHNSWFlat"))
print("has HNSWLVQ:", hasattr(hypervec, "IndexHNSWLVQ"))
PY
```

如果以上结果均为 `True`，说明 Python 包已包含本地 backend 和 C++ 向量索引绑定。

## 8. 验证 C++ 索引可从 Python 调用

```bash
python - <<'PY'
import numpy as np
import hypervec

x = np.random.rand(20, 8).astype("float32")
index = hypervec.IndexHNSWFlat(8, 16, hypervec.kMetricL2)
index.Train(x)
index.Add(x)
distances, labels = index.Search(x[:2], 3)
print("total:", index.n_total)
print("labels:", labels)
PY
```

预期输出中 `total` 为 `20`，`labels` 返回二维列表。

## 9. 本地 Backend 数据布局

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

## 10. 运行 Demo

从源码根目录运行：

```bash
python examples/python/demo_hypervector_backend.py
```

demo 会执行：

1. 构造示例向量、文本和 metadata
2. 使用 `HyperVectorIndexBackend.build_index()` 写入本地 collection
3. 使用 `search()` 返回文本列表
4. 使用 `search_with_meta()` 返回结构化结果
5. 使用 filter 查询指定 metadata
6. 重新加载本地 collection 验证持久化可用

## 11. 最小使用示例

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

## 12. 注意事项

- 当前 backend 是本地文件型 MVP，不提供 Milvus server 能力。
- 当前 Python 绑定服务于 `HyperVectorIndexBackend`，不是完整 hypervec Python API。
- filter 目前支持 `field == 'value'` 和 `AND` 连接。
- 多进程并发写同一个 collection 暂不支持。
- `metric_type="IP"` 时分数来自 HyperVector 返回值；业务侧如需统一“越大越好/越小越好”，应在上层约定。
- 如果运行时提示找不到动态库，确认 Python 虚拟环境、OpenMP runtime、BLAS/LAPACK runtime 均在系统动态库搜索路径中。
