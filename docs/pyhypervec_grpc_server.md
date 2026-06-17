# HyperVec gRPC Server

本文档描述 HyperVec gRPC server 的设计、使用方式和测试覆盖情况。

---

## 背景与动机

- 提升高频请求场景下的通信效率
- 为批量数据传输和较大请求体提供更稳的协议基础
- 为后续流式请求 / 流式响应能力预留技术路径
- 为未来更标准化的多语言 SDK 和服务化部署打基础

gRPC server 是 HTTP server 的增量能力，不是替换。两者共用同一个 `HypervecServerEngine` 实例，核心业务逻辑零重复。

---

## 新增文件

| 文件 | 说明 |
|------|------|
| `src/python/hypervec.proto` | Protocol Buffers 接口定义，15 个 RPC |
| `src/python/hypervec_pb2.py` | 由 proto 自动生成的消息类 |
| `src/python/hypervec_pb2_grpc.py` | 由 proto 自动生成的 stub / servicer |
| `src/python/hypervec_grpc_server.py` | gRPC servicer 实现 + CLI 入口 |
| `pyhypervec/pyhypervec/uri.py` | URI 解析与协议分流模块 |
| `pyhypervec/pyhypervec/_grpc_transport.py` | pyhypervec gRPC transport 实现 |

---

## gRPC 接口列表

| RPC | 说明 |
|-----|------|
| `Health` | 健康检查，返回 `{"status": "ok"}` |
| `ListCollections` | 列出所有集合名称 |
| `HasCollection` | 检查集合是否存在 |
| `DescribeCollection` | 获取集合元信息（dim、total、version 等） |
| `CreateCollection` | 创建集合，接受 schema JSON + index_params JSON |
| `DropCollection` | 删除集合 |
| `Insert` | 批量写入向量 + 标量字段，数据以 JSON 传输 |
| `Flush` | 持久化并构建索引，返回新版本号 |
| `LoadCollection` | 加载集合到内存以供搜索 |
| `CloseCollection` | 卸载集合，释放内存 |
| `Search` | KNN 向量搜索，返回结果列表 JSON |
| `GetVersion` | 获取集合当前版本号及 checksum |
| `SyncCheck` | 客户端版本检查，判断是否需要重新拉取索引 |
| `DownloadIndex` | 下载索引文件（bytes/unary） |
| `UploadIndex` | 上传索引文件（bytes/unary） |

所有 RPC 与 HTTP server 语义一致，同一份 `data_root` 下行为完全相同。

### Proto 设计说明

复杂类型（schema、data、results）通过 JSON string payload 传输，而非在 proto 中展开嵌套消息。这样做的好处：

- proto 定义保持简洁，易于维护
- 与 HTTP server 的 JSON 处理层共享序列化逻辑
- 后续如需严格类型化，可在不改变服务语义的前提下演进 proto

### 异常映射

| Python 异常 | gRPC status code |
|-------------|-----------------|
| `FileNotFoundError` | `NOT_FOUND` |
| `FileExistsError` | `ALREADY_EXISTS` |
| `ValueError` | `INVALID_ARGUMENT` |
| 其他 | `INTERNAL` |

---

## 启动方式

### 独立进程启动

```bash
python src/python/hypervec_grpc_server.py \
    --data-root /path/to/data \
    --port 50051 \
    --host 0.0.0.0 \
    --workers 10
```

### 程序内启动

```python
from hypervec_grpc_server import create_server

server = create_server(data_root="/path/to/data")
server.add_insecure_port("[::]:50051")
server.start()
server.wait_for_termination()
```

### 与 HTTP server 并行启动

```python
from hypervec_server_engine import HypervecServerEngine
from hypervec_grpc_server import create_server as create_grpc_server
from hypervec_http_server import create_app
import uvicorn, threading

engine = HypervecServerEngine("/path/to/data")

# gRPC server（独立线程）
grpc_server = create_grpc_server(engine=engine)
grpc_server.add_insecure_port("[::]:50051")
grpc_server.start()

# HTTP server
app = create_app(data_root="/path/to/data", engine=engine)
uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## pyhypervec URI 兼容能力

### 支持的 URI 格式

| URI 格式 | transport | 默认端口 | 路由到 |
|----------|-----------|----------|--------|
| `tcp://host:port` | `grpc` | 50051 | gRPC insecure channel |
| `grpc://host:port` | `grpc` | 50051 | gRPC insecure channel |
| `host:port`（裸格式） | `grpc` | 50051 | gRPC insecure channel |
| `http://host:port` | `http` | 8080 | HTTP REST |
| `https://host:port` | `https` | 443 | HTTPS REST |
| 其他 scheme | — | — | 抛 `ValueError` |

### 使用示例

```python
from pyhypervec import HypervecClient

# gRPC（tcp:// 是推荐写法，与 pymilvus 一致）
client = HypervecClient("tcp://localhost:50051")

# gRPC（等价写法）
client = HypervecClient("grpc://localhost:50051")
client = HypervecClient("localhost:50051")

# HTTP
client = HypervecClient("http://localhost:8080")

# HTTPS
client = HypervecClient("https://myhost:443")
```

`HypervecClient.__init__` 签名未变：`(uri, token=None, timeout=30.0, http2=False)`。

### 实现结构

```
pyhypervec/pyhypervec/
├── uri.py               # ParsedURI dataclass + parse_uri()
├── _grpc_transport.py   # GrpcTransport，封装 stub 调用
└── client.py            # __init__ 中根据 parse_uri() 结果分流
```

---

## 测试覆盖

### 本次新增测试

| 文件 | 测试数 | 覆盖内容 |
|------|--------|---------|
| `test_uri.py` | 11 | URI 解析、归一化、scheme 校验、client 分流 |
| `test_hypervec_grpc_server.py` | 19 | 全部 15 个 RPC + NOT_FOUND / ALREADY_EXISTS 错误 |
| `test_grpc_integration.py` | 14 | gRPC 主链路 + HTTP/gRPC parity（11 + 3） |

**合计：54 passed, 1 skipped, 2 warnings**

### 关于 4 个预存失败

以下 4 个测试在本 PR 之前的 `main` 分支也是失败状态，与本次改动无关：

| 测试 | 失败原因 |
|------|---------|
| `test_simd_dispatch::test_dispatch_function_exists` | `hypervec.SIMDConfig` 未在 `swighypervec.swig` 中声明（需 DD 构建） |
| `test_simd_dispatch::test_dispatch_with_env_var` | `hypervec.get_compile_options` 未在 SWIG 中声明 |
| `test_simd_dispatch::test_get_level_equals_get_dispatched_level` | 同上 |
| `test_omp_threads_py::test_openmp` | `hypervec.check_openmp` 未在 SWIG 中声明 |

这 4 个 C++ 函数在 `src/include/` 中已实现，但 `swighypervec.swig` 中尚未添加对应声明。修复方式：在 SWIG 接口文件中补充声明并重新构建，属于独立工作项。

### HTTP server 未被破坏的说明

`test_hypervec_http_server.py` 6 个测试全部通过。其中 `test_hypervec_http_server_sync_routes` 被标记为 `skipped`，原因是该测试依赖尚未实现的 `/sync` 路由（在 `main` 分支已是 skip 状态），非本 PR 引入。

---

## 后续工作

1. **Streaming RPC**：`DownloadIndex` / `UploadIndex` 目前是 bytes/unary，大索引文件传输效率有限，后续可改为 server-streaming RPC。

2. **TLS / mTLS 支持**：当前 gRPC 仅支持 insecure channel，生产部署需接入 `grpc.ssl_channel_credentials`。

3. **SWIG binding 补全**：修复 `SIMDConfig`、`check_openmp`、`get_compile_options` 的 4 个预存测试失败。

4. **stub 打包方式**：`_grpc_transport.py` 通过相对路径定位生成的 pb2 文件，适合 monorepo 开发，正式打包发布时需调整为标准包安装路径。

5. **标量过滤下推**：`search` 接口已透传 `filter` 字段，engine 侧暂未实现标量过滤逻辑。
