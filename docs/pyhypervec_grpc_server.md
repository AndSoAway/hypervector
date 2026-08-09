# pyhypervec gRPC Server

本文档说明从 `feat/grpc-v2` 迁移到最新 `main` 的 gRPC 能力。原分支已有的15个 RPC 名称、消息名称和字段编号保持不变；最新 `main` 的 collection 列表描述、索引示例、bundle 和 purge 接口作为增量 RPC 加入。

## 架构

HTTP 和 gRPC 都是 `HypervecServerEngine` 的协议适配层：

```text
pyhypervec HTTP client ──► FastAPI adapter ─┐
                                            ├─► one HypervecServerEngine
pyhypervec gRPC client ──► gRPC adapter ────┘
```

同时启动两种协议时必须使用 `hypervec_dual_server.py`，保证同一进程内共享一个 Engine。不要启动两个独立进程直接操作同一个 `data_root`。

## 安装

客户端 HTTP-only 安装不需要 gRPC 依赖：

```bash
pip install pyhypervec
```

客户端需要 gRPC 时：

```bash
pip install "pyhypervec[grpc]"
```

HyperVector Python server wheel 可分别安装：

```bash
pip install "hypervec[grpc-server]"
pip install "hypervec[dual-server]"
```

## 启动

只启动 gRPC：

```bash
bash scripts/start_grpc_server.sh --data-root ./data --port 50051
```

在一个进程内同时启动 HTTP 和 gRPC：

```bash
bash scripts/start_all_servers.sh \
  --data-root ./data \
  --http-port 8080 \
  --grpc-port 50051
```

也可以调用安装后的模块：

```bash
python -m hypervec.hypervec_grpc_server --data-root ./data --port 50051
python -m hypervec.hypervec_dual_server --data-root ./data
```

## 客户端 URI

```python
from pyhypervec import HypervecClient

grpc_client = HypervecClient("tcp://127.0.0.1:50051")
grpc_client = HypervecClient("grpc://127.0.0.1:50051")
http_client = HypervecClient("http://127.0.0.1:8080")
```

裸 `host:port` 按 Milvus 风格解释为 gRPC：

```python
client = HypervecClient("127.0.0.1:50051")
```

HTTP 和 gRPC 使用相同的 `HypervecClient` 方法。建议使用上下文管理器及时关闭 gRPC channel：

```python
with HypervecClient("grpc://127.0.0.1:50051") as client:
    print(client.health())
```

## RPC 覆盖

从 `feat/grpc-v2` 保留：

```text
Health
ListCollections
HasCollection
DescribeCollection
CreateCollection
DropCollection
Insert
Flush
LoadCollection
CloseCollection
Search
GetVersion
SyncCheck
DownloadIndex
UploadIndex
```

为匹配最新 `main` 新增：

```text
DescribeCollections
Examples
DownloadCollectionBundle
UploadCollectionBundle
PurgeCollectionData
```

## 错误映射

| Python 异常 | gRPC status |
|---|---|
| `FileNotFoundError` | `NOT_FOUND` |
| `FileExistsError` | `ALREADY_EXISTS` |
| `ConflictError` | `FAILED_PRECONDITION` |
| `ValueError` / JSON 参数错误 | `INVALID_ARGUMENT` |
| 其他异常 | `INTERNAL` |

客户端抛出 `HypervecGrpcError`，通过 `status_code` 和 `message` 保留结构化错误信息。

## 消息大小

原分支使用 gRPC 默认4 MiB限制，真实索引很容易失败。当前客户端和服务端默认将单消息上限提高到256 MiB。

客户端可配置：

```bash
export HYPERVEC_GRPC_MAX_MESSAGE_MB=512
```

服务端可配置：

```bash
bash scripts/start_grpc_server.sh --max-message-mb 512
```

当前 index 和 bundle 仍为 unary bytes，会整块进入内存。数百 MiB以上数据应在后续版本改为 streaming RPC。

## Proto 维护

协议源文件：

```text
src/python/hypervec.proto
```

生成的 stub 同时随 `hypervec` server package 和 `pyhypervec` client package 发布。修改 proto 后必须重新生成两套 stub，并保持包内相对导入。

```bash
python scripts/generate_grpc_stubs.py
```

## 当前验证

基础测试覆盖：

- URI分流、HTTP默认端口和 IPv6；
- 原分支15个 RPC 主链路；
- 最新 main 的 describe-all、examples、bundle 和 purge；
- `ConflictError` 等结构化错误；
- HTTP 创建后可立即从 gRPC看到，证明双协议共享同一个 Engine；
- 原有 HTTP client 和 FastAPI server 回归。

Wheel 全新环境测试、ARM wheel矩阵、大文件边界测试和高并发一致性压力测试列为后续专项任务。
