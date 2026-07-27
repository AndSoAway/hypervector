# HyperVector 当前运行参数入口梳理

> 文档性质：开发调研交付物
> 输入文档：`docs/runtime_parameter_entry.md`、`docs/parameter_inventory.md`
> 项目分支：`feature/config-refactor`

> 实现后状态：HTTP Server 现已通过 `HypervecConfig` 统一加载默认值、INI 和显式 CLI。新增配置还包括 `enable_http2`，以及只完成加载和访问的 `default_index_type`、`default_metric_type`。下文主体保留为实现前调研基线。

## 1. 结论摘要

HyperVector 当前没有统一的运行时配置对象，也没有面向 HTTP Server 启动的 INI/JSON/YAML/TOML 配置文件入口。当前参数入口分散在以下四条链路：

| 入口类型 | 主要位置 | 生效时机 | 主要作用 |
|---|---|---|---|
| HTTP Server CLI | `src/python/hypervec_http_server.py:243-302` | 服务进程启动 | 数据目录、绑定地址、端口、ASGI server、日志级别、TLS |
| Python 运行时环境变量 | `src/python/loader.py:62-103` | `hypervec` 包导入/扩展加载 | Python SWIG 扩展与 CPU 优化级别选择 |
| C++ DD 环境变量 | `src/utils/simd/simd_levels.cpp:65-77` | DD 库加载 | C++ SIMD 动态派发级别选择 |
| ARM 构建/启动脚本变量 | `scripts/build_arm_pyhypervec_server.sh:5-17` | 构建、安装及可选启动 | 工具链、依赖、构建模式和 server CLI 转发 |

其中，真正属于 HTTP Server 进程启动配置的入口只有 CLI。Python/C++ SIMD 环境变量属于库加载机制，构建脚本变量属于部署层，不应直接与 server CLI 归为同一优先级链。

## 2. 程序入口

### 2.1 HTTP Server 入口

主要可执行入口为：

```bash
python -m hypervec.hypervec_http_server
```

对应代码：

- `main()`：`src/python/hypervec_http_server.py:243`
- `argparse` 参数定义：`src/python/hypervec_http_server.py:244-256`
- 参数解析：`src/python/hypervec_http_server.py:257`
- FastAPI app 构建：`src/python/hypervec_http_server.py:26-240`
- Uvicorn/Hypercorn 启动：`src/python/hypervec_http_server.py:262-302`
- `__main__` 入口：`src/python/hypervec_http_server.py:305-306`

### 2.2 C++ Demo 入口

仓库内的 C++ demo 包含 `main()`，但当前都没有 `argc/argv` 参数解析，不构成运行参数入口：

- `test/examples/cpp/demo_hnsw_graph_search.cpp:22`
- `test/examples/cpp/demo_hnsw_indexing.cpp:22`
- `test/examples/cpp/demo_hnsw_lvq_indexing.cpp:22`
- `test/examples/cpp/demo_hnsw_pq_indexing.cpp:42`
- `test/examples/cpp/demo_ivf_indexing.cpp:42`
- `test/examples/cpp/demo_ivflvq_indexing.cpp:23`
- `test/examples/cpp/demo_ivfpq_indexing.cpp:44`
- `test/examples/cpp/demo_lvq_indexing.cpp:22`

### 2.3 测试入口

`test/unit_tests/python/test_simd_dispatch.py:129-130` 支持通过 `unittest.main()` 直接执行，它是测试入口，不是产品运行配置入口。

## 3. HTTP Server 启动链路

```text
python -m hypervec.hypervec_http_server
    |
    v
main()
    |
    +--> argparse.parse_args()
    |
    +--> logging.basicConfig(level=logging.INFO)
    |
    +--> create_app(data_root=args.data_root)
    |       |
    |       v
    |    HypervecServerEngine(data_root)
    |       |
    |       +--> <data_root>/collections.json
    |       +--> <data_root>/scalar.db
    |       +--> <data_root>/collections/<collection>/index.hypervec
    |
    +--> args.server == "uvicorn"
    |       |
    |       v
    |    uvicorn.run(host, port, log_level, TLS)
    |
    +--> args.server == "hypercorn"
            |
            v
         Hypercorn Config(bind, loglevel, TLS, ALPN)
```

当前 `argparse.Namespace` 是唯一的 server 启动参数容器。参数解析后被 `main()` 直接消费，中间没有项目级 `Config` 对象。

## 4. CLI 参数流向

| CLI 参数 | 流向 | 最终消费者 |
|---|---|---|
| `--data-root` | `args.data_root -> create_app() -> HypervecServerEngine()` | 元数据、SQLite 和索引文件路径 |
| `--host` | `args.host -> uvicorn.run()` 或 `Hypercorn Config.bind` | ASGI server |
| `--port` | `args.port -> uvicorn.run()` 或 `Hypercorn Config.bind` | ASGI server |
| `--server` | `args.server -> if/else` | Uvicorn 或 Hypercorn 启动分支 |
| `--log-level` | `args.log_level -> ASGI server` | Uvicorn/Hypercorn 日志级别 |
| `--certfile` | `args.certfile -> ASGI TLS config` | Uvicorn/Hypercorn TLS |
| `--keyfile` | `args.keyfile -> ASGI TLS config` | Uvicorn/Hypercorn TLS |

`--certfile` 和 `--keyfile` 必须成对提供，两个 server 分支都会校验该约束。

## 5. 环境变量入口

### 5.1 Python 扩展加载

```text
import hypervec
    |
    v
src/python/loader.py
    |
    +--> HYPERVEC_OPT_LEVEL
    +--> HYPERVEC_DISABLE_CPU_FEATURES
    |
    v
swighypervec_avx512_spr / avx512 / avx2 / sve / generic
```

这两个变量在 Python 扩展选择阶段生效，不经过 HTTP Server `argparse`。

### 5.2 C++ 动态派发

```text
C++ library load (HYPERVEC_ENABLE_DD)
    |
    v
SIMDConfig static initializer
    |
    +--> HYPERVEC_SIMD_LEVEL
    +--> unset: CPU auto-detection
    |
    v
SIMD dispatch wrappers
```

`HYPERVEC_SIMD_LEVEL` 只在 DD 构建中作为运行时覆盖；静态 SIMD 构建由编译期宏确定。

### 5.3 ARM 部署脚本

```text
build_arm_pyhypervec_server.sh
    |
    +--> 读取构建/安装环境变量
    +--> CMake 构建和 wheel 安装
    +--> START_SERVER=1 ?
            |
            v
         python -m hypervec.hypervec_http_server
           --data-root "$DATA_ROOT"
           --host "$SERVER_HOST"
           --port "$SERVER_PORT"
           --server "$SERVER_IMPL"
```

`DATA_ROOT`、`SERVER_HOST`、`SERVER_PORT`、`SERVER_IMPL` 不是 HTTP Server 直接读取的环境变量，而是由脚本转换为 CLI 参数。

## 6. 请求级参数入口

FastAPI 请求模型还接收 schema、`index_params`、`search_params`、`limit`、`output_fields`、`filter`、`consistency_level` 等请求级参数。这些值由每个 HTTP 请求提供，并流向 `HypervecServerEngine`的集合创建、索引构建或查询路径，不属于进程启动配置。

## 7. 日志参数现状

当前日志配置分为两条链路：

1. Python root logger 在 `main()` 中固定执行 `logging.basicConfig(level=logging.INFO)`。
2. `--log-level`默认值为 `info`，只传给 Uvicorn 或 Hypercorn。

`HypervecServerEngine` 默认使用 logger `hypervec.server`，Python loader 使用自身模块 logger，C++ 算法的 `verbose` 标志则直接输出到 stderr。这些日志控制目前没有统一配置入口。

## 8. 当前配置抽象与边界

仓库中存在一些参数对象，但它们不是统一运行配置：

- Hypercorn `Config`：只在 Hypercorn 启动分支中临时创建。
- `CollectionSchema` / `IndexParams`：客户端 API 和请求参数构建器。
- `KMeansParameters` / `PQParameters` / `LVQParameters`：C++ 算法局部参数。
- `SearchParametersHNSW` / `IVFSearchParameters`：单次搜索参数。
- `SIMDConfig`：仅负责 C++ SIMD 派发。

## 9. 最终判定

1. HTTP Server 的统一配置模块应从现有 7 个 CLI 参数和 Python 日志初始化入手。
2. Python/C++ SIMD 环境变量与构建脚本变量应保持独立，避免扩大首期配置模块范围。
3. HTTP 请求参数、客户端默认值和 C++ 算法参数不应与进程启动配置混合。
4. 引入配置文件后，必须保留现有 CLI 启动链路，并明确覆盖顺序。
