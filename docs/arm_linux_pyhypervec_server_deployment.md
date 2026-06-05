# ARM Linux pyhypervec 与 HyperVec HTTP Server 部署指南

本文说明如何在 ARM Linux 服务器上从源码编译 HyperVec，安装：

- `hypervec`：包含 HyperVec C++/Python binding 和 HTTP server。
- `pyhypervec`：纯 Python HTTP client 包，提供 `HypervecClient`。

目标调用链：

```text
client process -> pyhypervec.HypervecClient -> HTTP -> HyperVec server -> hypervec core
```

## 1. 安装原则

优先把工具和依赖安装在当前用户目录，避免影响系统 Python、系统 CMake、系统 SWIG 或其他用户环境。

推荐路径：

```text
$HOME/opt/python-3.12
$HOME/opt/cmake-3.24
$HOME/opt/swig-4.1.1
$HOME/hypervector
$HOME/hypervec_data
```

不要直接替换系统 Python。很多 Linux 发行版的 `yum`、系统管理工具依赖系统 Python，替换后可能导致系统工具异常。

## 2. 系统依赖

如果系统包仓库可用，先安装基础编译依赖。不同发行版包名可能不同，以下命令仅作参考。

Ubuntu/Debian：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  ninja-build \
  swig \
  python3-dev \
  python3-venv \
  libopenblas-dev \
  liblapack-dev \
  libomp-dev
```

RHEL/CentOS/Kylin：

```bash
sudo yum install -y \
  gcc \
  gcc-c++ \
  make \
  ninja-build \
  swig \
  openblas-devel \
  lapack-devel \
  libgomp
```

如果系统仓库找不到 `gcc`、`openssl-devel`、`readline-devel` 等包，需要根据当前发行版启用对应软件源，或改为用户目录源码编译缺失工具。

检查基础环境：

```bash
uname -m
python3 --version
cmake --version
swig -version
```

ARM64 机器通常返回：

```text
aarch64
```

## 3. Python 版本

一键脚本要求 Python >= 3.10。推荐使用 Python 3.12，并安装在用户目录：

```bash
$HOME/opt/python-3.12/bin/python3.12 --version
```

如果 `pip3` 或 `pip3.12` 不在 PATH，可以追加：

```bash
export PATH="$HOME/opt/python-3.12/bin:$PATH"
```

长期生效可以写入：

```bash
echo 'export PATH="$HOME/opt/python-3.12/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 4. 获取源码

如果 ARM 机器能访问 Git：

```bash
git clone <your-hypervector-repo-url> hypervector
cd hypervector
```

如果 ARM 机器不能访问 GitHub，可以从本地打包上传源码：

```powershell
tar -czf hypervector.tar.gz hypervector
scp -P 22222 hypervector.tar.gz xyq@172.22.163.151:~/
```

在 ARM 机器解压：

```bash
tar -xzf ~/hypervector.tar.gz -C ~/
cd ~/hypervector
```

## 5. 一键编译安装

在源码根目录执行：

```bash
PYTHON_BIN=$HOME/opt/python-3.12/bin/python3.12 \
bash scripts/build_arm_pyhypervec_server.sh
```

脚本会执行：

- 创建 `.venv`
- 安装 Python build 依赖
- 安装 `cmake>=3.24` 到 venv
- 安装 `fastapi` 和 `uvicorn`
- 配置 `build-arm`
- 编译 HyperVec C++ core
- 编译 Python binding
- 构建并安装 `hypervec` wheel
- 安装 `pyhypervec`
- 验证 server 新模块和 `HypervecClient` 同步接口

可选参数：

```bash
BUILD_DIR=build-arm-release \
VENV_DIR=.venv-arm \
HYPERVEC_OPT_LEVEL=generic \
PYTHON_BIN=$HOME/opt/python-3.12/bin/python3.12 \
bash scripts/build_arm_pyhypervec_server.sh
```

如果 CPU 支持 SVE：

```bash
HYPERVEC_OPT_LEVEL=sve \
PYTHON_BIN=$HOME/opt/python-3.12/bin/python3.12 \
bash scripts/build_arm_pyhypervec_server.sh
```

如果只想构建安装，不启动 server：

```bash
START_SERVER=0 bash scripts/build_arm_pyhypervec_server.sh
```

如果构建完成后直接启动 server：

```bash
START_SERVER=1 \
DATA_ROOT=$HOME/hypervec_data \
SERVER_HOST=0.0.0.0 \
SERVER_PORT=8080 \
PYTHON_BIN=$HOME/opt/python-3.12/bin/python3.12 \
bash scripts/build_arm_pyhypervec_server.sh
```

## 6. 手动启动 server

编译安装完成后：

```bash
cd ~/hypervector
source .venv/bin/activate
python -m hypervec.hypervec_http_server \
  --data-root $HOME/hypervec_data \
  --host 0.0.0.0 \
  --port 8080
```

本机健康检查：

```bash
curl http://127.0.0.1:8080/health
```

远程 client 检查：

```bash
curl http://<server-ip>:8080/health
```

如果远程无法直连，可以用 SSH tunnel：

```powershell
ssh -p 22222 -L 8080:127.0.0.1:8080 xyq@172.22.163.151
```

本地再访问：

```powershell
curl.exe http://127.0.0.1:8080/health
```

## 7. 后台运行 server

如果希望断开 SSH 后 server 继续运行，可以使用 `nohup`：

```bash
cd ~/hypervector
source .venv/bin/activate
nohup python -m hypervec.hypervec_http_server \
  --data-root $HOME/hypervec_data \
  --host 0.0.0.0 \
  --port 8080 \
  > $HOME/hypervector/hypervec_server.log 2>&1 &
echo $! > $HOME/hypervector/hypervec_server.pid
```

查看日志：

```bash
tail -f $HOME/hypervector/hypervec_server.log
```

查看进程：

```bash
ps -fp "$(cat $HOME/hypervector/hypervec_server.pid)"
```

停止进程：

```bash
kill "$(cat $HOME/hypervector/hypervec_server.pid)"
```

## 8. 运行时数据目录

server 的 `--data-root` 下会保存：

```text
$HOME/hypervec_data/
  collections.json
  scalar.db
  collections/
    <collection_name>/
      index.hypervec
```

说明：

- `collections.json`：collection 元数据、version、schema、index checksum。
- `scalar.db`：SQLite 标量字段、文本、metadata、原始 vector BLOB。
- `index.hypervec`：HyperVec 序列化索引文件。

当前第一版按单进程 server 设计，不建议 uvicorn 多 worker。

## 9. pyhypervec client 示例

在任意已安装 `pyhypervec` 的 client 环境中：

```python
from pyhypervec import DataType, HypervecClient

client = HypervecClient("http://127.0.0.1:8080")

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

client.create_collection("demo", schema=schema, index_params=index_params)
client.insert(
    "demo",
    [
        {"id": "a", "vector": [0.0, 0.0], "contents": "zero", "source": "manual"},
        {"id": "b", "vector": [1.0, 1.0], "contents": "one", "source": "manual"},
    ],
)
client.flush("demo")
client.load_collection("demo")

results = client.search(
    collection_name="demo",
    data=[[0.1, 0.1]],
    limit=2,
    output_fields=["id", "contents", "source"],
)
print(results)
```

## 10. 索引版本同步

查询版本：

```python
version = client.get_version("demo")
print(version)
```

检查是否需要同步：

```python
sync = client.sync_check(
    "demo",
    client_version=version["version"] - 1,
    client_checksum=None,
)
print(sync)
```

下载索引：

```python
client.download_index("demo", "./demo.index.hypervec")
```

上传索引：

```python
client.upload_index(
    "demo",
    "./demo.index.hypervec",
    version=version["version"],
    checksum=version["index_checksum"],
)
```

server 会拒绝旧版本索引覆盖新版本索引。

## 11. 常见问题

### ModuleNotFoundError: hypervec.hypervec_scalar_store

说明 wheel 中缺少新增 server 模块。重新拉取最新代码后执行：

```bash
bash scripts/build_arm_pyhypervec_server.sh
```

### Empty reply from server

常见原因是连接到了不可达地址、server 进程已退出、或者 SSH tunnel 已断开。

先在 ARM 机器本机检查：

```bash
curl http://127.0.0.1:8080/health
```

如果本机正常但远程不通，使用 SSH tunnel：

```powershell
ssh -p 22222 -L 8080:127.0.0.1:8080 xyq@172.22.163.151
```

### 断开 SSH 后 server 停止

前台启动的 server 会随 SSH 会话退出。使用 `nohup`、`tmux`、`screen` 或 systemd user service 后台运行。
