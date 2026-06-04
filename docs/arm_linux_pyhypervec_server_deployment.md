# ARM Linux pyhypervec 与 HyperVec HTTP Server 部署指南

本文说明如何在 ARM Linux 服务器上从源码编译并部署：

- `hypervec` server 包：包含 HyperVec C++/Python binding 和 HTTP server。
- `pyhypervec` client 包：纯 Python HTTP client，提供 `HypervecClient`。

目标调用链：

```text
client process -> pyhypervec.HypervecClient -> HTTP -> HyperVec server -> hypervec core
```

## 1. 安装系统依赖

Ubuntu/Debian ARM64 示例：

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

检查架构：

```bash
uname -m
python3 --version
cmake --version
```

`uname -m` 应返回 `aarch64` 或类似 ARM64 标识。

## 2. 获取源码

```bash
git clone <your-hypervector-repo-url> hypervector
cd hypervector
```

## 2.1 一键编译安装

如果系统依赖已经安装好，可以直接运行脚本完成服务端和客户端包安装：

```bash
bash scripts/build_arm_pyhypervec_server.sh
```

脚本会执行：

- 创建 `.venv`
- 安装 Python build 依赖
- 安装 `fastapi` 和 `uvicorn`
- 配置 `build-arm`
- 编译 `hypervec` C++ core
- 编译 Python binding
- 构建并安装 `hypervec` wheel
- 安装 `pyhypervec`
- 验证 `hypervec`、HTTP server 入口和 `pyhypervec` 可导入

可选环境变量：

```bash
BUILD_DIR=build-arm-release \
VENV_DIR=.venv-arm \
HYPERVEC_OPT_LEVEL=generic \
bash scripts/build_arm_pyhypervec_server.sh
```

如果目标 CPU 支持 SVE：

```bash
HYPERVEC_OPT_LEVEL=sve bash scripts/build_arm_pyhypervec_server.sh
```

如果系统默认 `python3` 不是 Python 3.10+，显式指定 Python 3.12：

```bash
PYTHON_BIN=$HOME/opt/python-3.12/bin/python3.12 \
bash scripts/build_arm_pyhypervec_server.sh
```

如果要构建完成后直接拉起 HyperVec HTTP Server：

```bash
START_SERVER=1 \
PYTHON_BIN=$HOME/opt/python-3.12/bin/python3.12 \
DATA_ROOT=$HOME/hypervec_data \
SERVER_HOST=0.0.0.0 \
SERVER_PORT=8080 \
bash scripts/build_arm_pyhypervec_server.sh
```

如果只想编译安装服务端，不安装 `pyhypervec`：

```bash
INSTALL_PYHYPERVEC=0 bash scripts/build_arm_pyhypervec_server.sh
```

完成后启动服务：

```bash
source .venv/bin/activate
python -m hypervec.hypervec_http_server \
  --data-root $HOME/hypervec_data \
  --host 0.0.0.0 \
  --port 8080
```

## 3. 创建服务端 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging numpy
```

## 4. 编译 hypervec server 包

推荐先使用 `generic` 优化级别，兼容性最好：

```bash
cmake -S . -B build-arm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYPERVEC_ENABLE_PYTHON=ON \
  -DHYPERVEC_ENABLE_EXTRAS=OFF \
  -DHYPERVEC_OPT_LEVEL=generic \
  -DPython_EXECUTABLE="$(which python)"
```

编译 C++ core 和 Python binding：

```bash
cmake --build build-arm --target hypervec -j"$(nproc)"
cmake --build build-arm --target swighypervec -j"$(nproc)"
```

构建 wheel 并安装 server extra：

```bash
cd build-arm/src/python
python setup.py bdist_wheel
python -m pip install --force-reinstall "dist/"*.whl
python -m pip install "fastapi" "uvicorn"
```

返回源码根目录：

```bash
cd ../../..
```

验证 `hypervec` 可导入：

```bash
python - <<'PY'
import hypervec
print("hypervec:", hypervec.__file__)
print("has IndexHNSWFlat:", hasattr(hypervec, "IndexHNSWFlat"))
PY
```

验证 HTTP server 入口可导入：

```bash
python - <<'PY'
import hypervec.hypervec_http_server as server
print("server module:", server.__file__)
PY
```

## 5. 安装 pyhypervec client 包

`pyhypervec` 是纯 Python 包，不依赖 HyperVec C++ binding。可以安装在服务端，
也可以安装在远程 client 机器上。

在源码根目录执行：

```bash
python -m pip install ./pyhypervec
```

验证：

```bash
python - <<'PY'
from pyhypervec import HypervecClient, DataType
print(HypervecClient)
print(DataType.FLOAT_VECTOR)
PY
```

## 6. 准备服务端数据目录

```bash
mkdir -p $HOME/hypervec_data
```

## 7. 拉起 HyperVec HTTP Server

```bash
source .venv/bin/activate
python -m hypervec.hypervec_http_server \
  --data-root $HOME/hypervec_data \
  --host 0.0.0.0 \
  --port 8080
```

本机检查：

```bash
curl http://127.0.0.1:8080/health
```

远程 client 检查：

```bash
curl http://<server-ip>:8080/health
```

如果远程访问失败，检查：

- server 是否使用 `--host 0.0.0.0`
- Linux 防火墙
- 云服务器安全组
- 端口 `8080` 是否被占用

## 8. pyhypervec 远程访问示例

在任意已安装 `pyhypervec` 的 client 环境中：

```python
from pyhypervec import DataType, HypervecClient

client = HypervecClient("http://<server-ip>:8080")

schema = HypervecClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
    description="demo collection",
)
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
schema.add_field("contents", DataType.VARCHAR, max_length=60000)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="L2",
    index_type="Flat",
)

if client.has_collection("demo"):
    client.drop_collection("demo")

client.create_collection(
    collection_name="demo",
    schema=schema,
    index_params=index_params,
)

client.insert(
    "demo",
    data=[
        {"id": "a", "vector": [0, 0], "contents": "zero", "source": "manual"},
        {"id": "b", "vector": [1, 1], "contents": "one", "source": "manual"},
        {"id": "c", "vector": [10, 10], "contents": "ten", "source": "other"},
    ],
)
client.flush("demo")
client.load_collection("demo")

results = client.search(
    collection_name="demo",
    data=[[0.1, 0.1]],
    limit=2,
    output_fields=["id", "contents", "source"],
    filter="source == 'manual'",
)
print(results)
```

## 9. systemd 后台运行

创建 `/etc/systemd/system/hypervec-http.service`：

```ini
[Unit]
Description=HyperVec HTTP Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/hypervector
Environment=PATH=/path/to/hypervector/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/path/to/hypervector/.venv/bin/python -m hypervec.hypervec_http_server --data-root /home/ubuntu/hypervec_data --host 0.0.0.0 --port 8080
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hypervec-http
sudo systemctl status hypervec-http
```

日志：

```bash
journalctl -u hypervec-http -f
```

## 10. 产物说明

服务端产物：

```text
build-arm/src/libhypervec.*
build-arm/src/python/dist/hypervec-*.whl
```

客户端产物：

```text
pyhypervec source package
```

如果要制作 `pyhypervec` wheel：

```bash
cd pyhypervec
python -m pip install build
python -m build
ls dist/*.whl
```

## 11. 注意事项

- `pyhypervec` client 不需要本地 `hypervec` C++ binding。
- HyperVec server 必须安装编译好的 `hypervec` Python 包。
- 第一版 server 使用进程内 index cache，建议单 worker 运行。
- 大规模数据不建议一次性通过巨大 JSON 插入；后续应扩展分批 ingest 或文件导入接口。
