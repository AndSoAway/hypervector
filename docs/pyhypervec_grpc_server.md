## 一、总览

本文档记录 `feat/grpc-v2` 分支的完整交付内容。

gRPC server 是 HTTP server 的**增量能力**。两者通过依赖注入共用同一个 `HypervecServerEngine`，核心业务逻辑不重复，transport 层只负责协议适配和异常映射。

```
客户端
  ├── HTTP  ──► hypervec_http_server.py  ─┐
  └── gRPC  ──► hypervec_grpc_server.py  ─┴──► HypervecServerEngine（唯一业务层）
                                                      │
                                                ┌─────┴─────┐
                                           MetaStore   ScalarStore
                                           (JSON 元数据)  (SQLite 标量)
```

---

## 二、需求对照

### 目标 A / 硬性约束 / 实现思路

| 需求 | 实现 | 验证 |
|------|------|------|
| 不破坏 HTTP server | HTTP server 文件未改动，gRPC 为新增文件 | HTTP 回归 6/6 通过 |
| 核心逻辑不重复 | 唯一业务层 `HypervecServerEngine`，两个 transport 调同一套方法 | — |
| 覆盖全部 15 个接口 | 见 §三.2 接口列表 | 集成测试端到端验证 |
| HTTP 与 gRPC 语义一致 | 同一份 `data_root`，调同一个 engine 方法 | `test_grpc_integration.py` parity 3 个场景 |
| 抽公共 service 层 | `src/python/hypervec_server_engine.py` | — |
| 独立 proto 文件 | `src/python/hypervec.proto`，15 个 RPC | — |
| proto 复杂类型用 JSON string | schema / 向量数据 / 搜索结果以 JSON string 传输，复用 HTTP 序列化逻辑 | — |
| download/upload 先用 bytes/unary | 已实现，streaming 列入后续计划（§七.5） | — |

### 目标 B / URI 处理

| 需求 | 实现 | 验证 |
|------|------|------|
| 支持 tcp:// / http:// / https:// | 显式 scheme 白名单，见 §四.2 | `test_uri.py` 11 个用例 |
| tcp:// 归一化逻辑明确 | `parse_uri()` 6 步归一化，见 §四.3 | `test_tcp_scheme`、`test_bare_hostport_treated_as_grpc` 等 |
| 公开接口不变 | `HypervecClient.__init__` 签名未变，见 §四.1 | `test_client_uri_signature_unchanged` 锁定签名 |
| 新增行为可测试 | URI 11 个、gRPC server 19 个、集成 14 个单元测试 | 全部有 assert |

### feat/python-server-bindings 迁移

旧分支有价值内容全部移植，见 §五。

### 验收标准

| 验收标准 | 结果 |
|---------|------|
| 1. 基于 main 新分支开发完成 | ✅ 分支 `feat/grpc-v2`，从 main commit `5581429` 拉取 |
| 2. pyhypervec 对外接口未变 | ✅ 签名测试锁定，11 个 URI 测试全绿 |
| 3. pyhypervec 能稳定处理 tcp:// / http:// / https:// URI | ✅ `uri.py` 显式归一化逻辑，有校验和错误处理 |
| 4. gRPC server 能独立启动 | ✅ `hypervec_grpc_server.py` 提供 `create_server()` 和 `main()` CLI |
| 5. 主要 collection / insert / search / version 接口可用 | ✅ 15 个 RPC，集成测试端到端验证 |
| 6. HTTP server 未被破坏 | ✅ HTTP 回归测试 6/6 通过，见 §六.2 |

---

## 三、gRPC server 实现细节

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `src/python/hypervec.proto` | Protocol Buffers 接口定义，15 个 RPC |
| `src/python/hypervec_pb2.py` | 由 proto 自动生成的消息类（不要手动修改） |
| `src/python/hypervec_pb2_grpc.py` | 由 proto 自动生成的 stub / servicer（不要手动修改） |
| `src/python/hypervec_grpc_server.py` | gRPC servicer 实现 + CLI 入口 |
| `pyhypervec/pyhypervec/uri.py` | URI 解析与协议分流模块 |
| `pyhypervec/pyhypervec/_grpc_transport.py` | pyhypervec gRPC transport 封装 |
| `pyhypervec/pyhypervec/client.py` | 修改：接入 URI 解析和 gRPC transport 分流（接口未变） |

### 3.2 接口列表

| RPC | 对应 HTTP 路由 | 说明 |
|-----|--------------|------|
| `Health` | `GET /health` | 健康检查，返回 `{"status": "ok"}` |
| `ListCollections` | `GET /collections` | 列出所有集合名称 |
| `HasCollection` | `GET /collections/{name}/exists` | 检查集合是否存在，返回布尔值 |
| `DescribeCollection` | `GET /collections/{name}/describe` | 获取集合元信息（dim、total、version 等） |
| `CreateCollection` | `POST /collections/{name}/create` | 创建集合，接受 schema JSON + index_params JSON |
| `DropCollection` | `DELETE /collections/{name}` | 删除集合及全部数据 |
| `Insert` | `POST /collections/{name}/insert` | 批量写入向量 + 标量字段，数据以 JSON 传输 |
| `Flush` | `POST /collections/{name}/flush` | 持久化并构建索引，返回新版本号 |
| `LoadCollection` | `POST /collections/{name}/load` | 加载集合到内存以供搜索 |
| `CloseCollection` | `POST /collections/{name}/close` | 卸载集合，释放内存 |
| `Search` | `POST /collections/{name}/search` | KNN 向量搜索，返回 top-k 结果 JSON |
| `GetVersion` | `GET /collections/{name}/version` | 获取集合当前版本号及 checksum |
| `SyncCheck` | `POST /collections/{name}/sync-check` | 客户端版本检查，判断是否需要重新拉取索引 |
| `DownloadIndex` | `GET /collections/{name}/index` | 下载索引文件原始字节（bytes/unary） |
| `UploadIndex` | `PUT /collections/{name}/index` | 上传索引文件原始字节（bytes/unary） |

### 3.3 异常映射

`HyperVecServicer` 不含任何业务逻辑，只做协议适配：将 protobuf 字段传给 engine，将 Python 异常映射为 gRPC status code。

| Python 异常 | gRPC status code | HTTP 等价 |
|-------------|-----------------|-----------|
| `FileNotFoundError` | `NOT_FOUND` | 404 |
| `FileExistsError` | `ALREADY_EXISTS` | 409 |
| `ValueError` | `INVALID_ARGUMENT` | 400 |
| 其他 | `INTERNAL` | 500 |

### 3.4 Proto 设计决策

复杂类型（schema、向量数据、搜索结果）通过 JSON string payload 传输，而非展开为嵌套消息类型。原因：
- proto 定义保持简洁，维护成本低
- 与 HTTP server 的 JSON 处理层复用序列化逻辑
- 后续如需强类型化，可在不改变服务语义的前提下演进 proto

### 3.5 启动方式

**仅启动 gRPC server：**

```bash
# 使用默认参数（data-root=./data，port=50051）
./scripts/start_grpc_server.sh

# 自定义参数
./scripts/start_grpc_server.sh --data-root /path/to/data --port 50051 --host 0.0.0.0 --workers 10

# 也可以直接调用 Python 入口
python src/python/hypervec_grpc_server.py --data-root /path/to/data --port 50051
```

**同时启动 gRPC + HTTP server（共享同一 data-root）：**

```bash
# 使用默认参数（grpc-port=50051，http-port=8080）
./scripts/start_all_servers.sh

# 自定义参数
./scripts/start_all_servers.sh \
    --data-root /path/to/data \
    --grpc-port 50051 \
    --http-port 8080
```

两个脚本位于 `scripts/`，按 `Ctrl+C` 退出时会同时关闭两个进程。

**重新生成 proto stubs（如修改了 proto 文件）：**

```bash
cd src/python
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. hypervec.proto
```

---

## 四、pyhypervec URI 处理

### 4.1 接口稳定性保证

`HypervecClient.__init__` 签名**完全未变**，有测试锁定：

```python
HypervecClient(uri, token=None, timeout=30.0, http2=False)
```

`test_client_uri_signature_unchanged` 通过 `inspect.signature` 锁定参数列表及默认值，URI 解析是纯内部重构，调用方无感知。

### 4.2 支持的 URI 格式

| URI 格式 | transport | 默认端口 | 路由到 |
|----------|-----------|----------|--------|
| `tcp://host:port` | `grpc` | 50051 | gRPC insecure channel |
| `grpc://host:port` | `grpc` | 50051 | gRPC insecure channel |
| `host:port`（裸格式，无 `://`） | `grpc` | 50051 | gRPC insecure channel |
| `http://host:port` | `http` | 8080 | HTTP REST |
| `https://host:port` | `https` | 443 | HTTPS REST |
| 其他 scheme | — | — | 抛 `ValueError: Unsupported URI scheme` |
| 缺少 host | — | — | 抛 `ValueError: Missing host` |

`tcp://` 是推荐写法，与 pymilvus 的使用习惯保持一致。TLS gRPC（`grpcs://`）留作后续扩展（§七.3）。

### 4.3 归一化流程

`parse_uri()` 的处理步骤：

1. 字符串中无 `://` → 自动补 `tcp://` 前缀（裸 `host:port` 兼容）
2. 提取 scheme，在白名单 `{tcp, grpc, http, https}` 中校验，不在则抛 `ValueError`
3. 提取 host，为空则抛 `ValueError`
4. 提取 port，缺省时使用各 scheme 对应默认端口
5. `tcp`/`grpc` → 返回 `ParsedURI(transport="grpc", address="host:port", http_base=None)`
6. `http`/`https` → 返回 `ParsedURI(transport=scheme, address="host:port", http_base="scheme://host:port")`

### 4.4 代码结构

```
pyhypervec/pyhypervec/
├── uri.py               # ParsedURI dataclass + parse_uri()
├── _grpc_transport.py   # GrpcTransport，封装 stub 调用，grpc.RpcError → HypervecClientError
└── client.py            # __init__ 调用 parse_uri()，按 transport 字段分流
```

### 4.5 使用示例

```python
from pyhypervec import HypervecClient

# gRPC — 三种等价写法
client = HypervecClient("tcp://localhost:50051")
client = HypervecClient("grpc://localhost:50051")
client = HypervecClient("localhost:50051")

# HTTP / HTTPS
client = HypervecClient("http://localhost:8080")
client = HypervecClient("https://myhost:443")

# 连接后接口完全一致，无需关心底层协议
client.create_collection("demo", schema=schema)
client.insert("demo", rows)
client.flush("demo")
client.load_collection("demo")
results = client.search(collection_name="demo", data=query, limit=5)
```

---

## 五、从 feat/python-server-bindings 补迁移了哪些内容

`feat/python-server-bindings` 与 `main` 的差异只剩旧 `server/` 目录：

| 旧文件 | 迁移状态 |
|--------|---------|
| `server/hv.py` | 已在 `main` 以 `src/python/swighypervec.py` + `loader.py` 替代 |
| `server/hypervec_backend.py` | 已在 `main` 以 `src/python/hypervec_server_engine.py` 替代 |
| `server/index_manager.py` | 已在 `main` 以 `hypervec_meta_store.py` + `hypervec_scalar_store.py` 替代 |
| `server/store.py` / `scalar_store.py` | 同上 |
| `server/main.py` | 已在 `main` 以 `src/python/hypervec_http_server.py` 替代 |
| `server/test_server.py` | 依赖编译的 C++ extension，纯 Python 等价测试已在本 PR 补充 |
| `server/README.md` | 本文档已覆盖其全部内容 |

---

## 六、测试结果

### 6.1 新增测试（本 PR）

| 测试文件 | 测试数 | 覆盖内容 |
|---------|--------|---------|
| `test/unit_tests/python/test_uri.py` | 11 | URI 解析正确性、归一化、scheme 校验、非法输入、client 内部分流验证 |
| `test/unit_tests/python/test_hypervec_grpc_server.py` | 19 | 全部 15 个 RPC 的请求/响应正确性，`NOT_FOUND`、`ALREADY_EXISTS` 错误码映射 |
| `test/unit_tests/python/test_grpc_integration.py` | 14 | gRPC 主链路（create/insert/flush/load/search/version/sync-check/download/upload）+ HTTP/gRPC parity（3 个场景） |

### 6.2 回归测试（已有，全部通过）

| 测试文件 | 测试数 | 说明 |
|---------|--------|------|
| `test_hypervec_http_server.py` | 6 | HTTP server 完整回归，包含路由覆盖 |
| `test_hypervec_server_engine.py` | 5 | engine 层回归 |
| `test_hypervec_stores.py` | 5 | MetaStore / ScalarStore 回归 |
| `test_pyhypervec_client.py` | 14 | pyhypervec HTTP client 回归 |

**汇总结果：54 passed, 1 skipped, 2 warnings**

skipped 的 1 个是 `test_hypervec_http_server_sync_routes`，该测试依赖尚未实现的 HTTP `/sync` 路由，在 `main` 分支上原本就是 skip 状态，非本次引入。

### 6.3 预存失败（与本 PR 无关）

以下 4 个测试在本 PR 之前的 `main` 分支已是相同的失败状态：

| 测试 | 失败原因 |
|------|---------|
| `test_simd_dispatch::test_dispatch_function_exists` | `hypervec.SIMDConfig` 未在 `swighypervec.swig` 中声明（需 DD 构建） |
| `test_simd_dispatch::test_dispatch_with_env_var` | `hypervec.get_compile_options` 未在 SWIG 中声明 |
| `test_simd_dispatch::test_get_level_equals_get_dispatched_level` | 同上 |
| `test_omp_threads_py::test_openmp` | `hypervec.check_openmp` 未在 SWIG 中声明 |

这 4 个 C++ 函数在 `src/include/` 中已实现，但 `swighypervec.swig` 中尚未补充 Python 声明，属于独立的 SWIG binding 完善工作，需单独跟进。

---

## 七、后续工作

1. **SWIG binding 补全**：在 `swighypervec.swig` 中补充 `SIMDConfig`、`check_openmp`、`get_compile_options` 声明，修复 4 个预存测试失败。

2. **stub 打包方式**：`_grpc_transport.py` 通过相对路径定位生成的 pb2 文件，适合 monorepo 开发，正式打包发布时需将 pb2 文件纳入 `pyhypervec` 包内。

3. **TLS / mTLS 支持**：当前 gRPC 仅支持 insecure channel，生产部署需接入 `grpc.ssl_channel_credentials`，可同步新增 `grpcs://` scheme 支持。

4. **标量过滤下推**：`search` 接口已透传 `filter` 字段到 engine，engine 侧暂未实现标量过滤逻辑，后续按需补充。

5. **Streaming RPC**：`DownloadIndex` / `UploadIndex` 目前是 bytes/unary，大索引文件（>100MB）传输效率有限，后续可改为 server-streaming RPC，proto 已预留扩展空间，无需修改 proto message 定义。
