# HyperVector 现有 CLI、环境变量与默认值清单

> 文档性质：开发调研交付物  
> 输入文档：`docs/parameter_inventory.md`、`docs/runtime_parameter_entry.md`  
> 项目分支：`feature/config-refactor`

## 1. 清单范围

本文档将现有参数分为以下类别：

1. HTTP Server 进程直接解析的 CLI 参数。
2. Python/C++ 在运行时读取的环境变量。
3. ARM 构建和可选启动脚本读取的环境变量。
4. 影响服务、客户端、请求、索引算法或构建产物的硬编码默认值。

当前只有 CLI 是 HTTP Server 的直接启动参数入口。`DATA_ROOT`、`SERVER_HOST` 等脚本变量会被转换为 CLI；SIMD 相关变量由库加载器单独读取。

## 2. CLI 参数清单

定义位置：`src/python/hypervec_http_server.py:244-256`

| 参数 | 类型 | 默认值 | 作用 | 后续 Config 候选 |
|---|---|---|---|---|
| `--data-root` | string/path | 无，必填 | collection 数据根目录 | 是 |
| `--host` | string | `127.0.0.1` | ASGI server 绑定地址 | 是 |
| `--port` | int | `8080` | ASGI server 绑定端口 | 是 |
| `--server` | enum string | `hypercorn` | 选择 `hypercorn` 或 `uvicorn` | 是 |
| `--log-level` | string | `info` | 传给 Uvicorn/Hypercorn 的日志级别 | 是 |
| `--certfile` | optional path | `None` | TLS 证书文件 | 是 |
| `--keyfile` | optional path | `None` | TLS 私钥文件 | 是 |

约束：`--certfile` 和 `--keyfile` 必须同时提供或同时为空。

## 3. 环境变量清单

### 3.1 Python/C++ 运行时变量

| 变量 | 类型 | 默认值 | 作用 | 来源 |
|---|---|---|---|---|
| `HYPERVEC_DISABLE_CPU_FEATURES` | string list | `""` | 从 Python loader 检测到的 CPU feature 中排除指定项 | `src/python/loader.py:62-63,82-83` |
| `HYPERVEC_OPT_LEVEL` | string | 未设置，自动检测 | 覆盖 Python SWIG 扩展选择 | `src/python/loader.py:91-103` |
| `HYPERVEC_SIMD_LEVEL` | enum string | 未设置，自动检测 | DD 构建的 C++ SIMD 级别：`NONE/AVX2/AVX512/AVX512_SPR/ARM_NEON/ARM_SVE` | `src/utils/simd/simd_levels.cpp:65-77,296-317` |

### 3.2 ARM 构建/启动脚本变量

定义位置：`scripts/build_arm_pyhypervec_server.sh:5-17`

| 变量 | 类型 | 默认值 | 作用 | 是否影响 server 启动 |
|---|---|---|---|---|
| `BUILD_DIR` | path | `build-arm` | CMake 构建目录 | 否 |
| `VENV_DIR` | path | `.venv` | Python 虚拟环境目录 | 否 |
| `PYTHON_BIN` | path/command | `python3` | 创建虚拟环境的 Python | 否 |
| `HYPERVEC_OPT_LEVEL` | string | `generic` | 传给 CMake 的优化级别 | 间接，影响构建产物 |
| `CMAKE_GENERATOR` | string | `Ninja` | CMake generator | 否 |
| `INSTALL_PYHYPERVEC` | bool-like | `1` | 是否安装 Python client | 否 |
| `INSTALL_SERVER_DEPS` | bool-like | `1` | 是否安装 FastAPI/Uvicorn/Hypercorn/h2 | 间接 |
| `INSTALL_CMAKE` | bool-like | `1` | 是否在虚拟环境中安装 CMake | 否 |
| `DATA_ROOT` | path | `${HOME}/hypervec_data` | `START_SERVER=1` 时转为 `--data-root` | 是 |
| `SERVER_HOST` | string | `0.0.0.0` | 转为 `--host` | 是 |
| `SERVER_PORT` | int-like | `8080` | 转为 `--port` | 是 |
| `SERVER_IMPL` | enum string | `hypercorn` | 转为 `--server` | 是 |
| `START_SERVER` | bool-like | `0` | 构建完成后是否 `exec` HTTP Server | 是 |

### 3.3 构建工具链变量

| 变量 | 默认值 | 作用 | 来源 |
|---|---|---|---|
| `MKLROOT` | 未设置 | MKL 发现根目录 | `cmake/FindMKL.cmake:64,317-348` |
| `PATH` | 进程环境 | `build.sh` 使用 Miniconda 时前置工具目录 | `build.sh:131` |
| `CC` | 由脚本设置 | C 编译器 | `build.sh:132,183` |
| `CXX` | 由脚本设置 | C++ 编译器 | `build.sh:133,184` |

`HYPERVEC_OPT_LEVEL` 同时出现在 Python loader 和 ARM 构建脚本中，但两者语义不同：前者选择已存在的 Python 扩展，后者决定要构建哪种优化产物。

## 4. 现有配置/状态文件

| 文件 | 类型 | 用途 | 是否为启动配置 |
|---|---|---|---|
| `pyhypervec/pyproject.toml` | TOML | Python client 包元数据 | 否 |
| `<data_root>/collections.json` | JSON | server 生成的 collection 元数据 | 否，是运行状态 |
| `<data_root>/scalar.db` | SQLite | 标量字段和向量行存储 | 否，是运行数据 |
| `<data_root>/collections/<collection>/index.hypervec` | binary | 持久化向量索引 | 否，是运行数据 |

结论：仓库目前不支持用户通过配置文件设置 HTTP Server 启动参数。

## 5. HTTP Server 与应用默认值

| 名称 | 默认值 | 作用 | 来源 |
|---|---|---|---|
| FastAPI title | `HyperVec HTTP Server` | OpenAPI/app 标题 | `src/python/hypervec_http_server.py:62` |
| FastAPI version | `1` | OpenAPI/app 版本 | `src/python/hypervec_http_server.py:62` |
| Python root logging level | `logging.INFO` | `logging.basicConfig()` 固定级别 | `src/python/hypervec_http_server.py:259` |
| Hypercorn bind | `[f"{host}:{port}"]` | 由 CLI host/port 派生 | `src/python/hypervec_http_server.py:296-298` |
| Hypercorn ALPN | `["h2", "http/1.1"]` | HTTP/2 和 HTTP/1.1 协商 | `src/python/hypervec_http_server.py:301` |

## 6. Server 存储与元数据默认值

| 名称 | 默认值 | 来源 | Config 候选 |
|---|---|---|---|
| 索引文件名 | `index.hypervec` | `src/python/hypervec_server_engine.py:31` | 是 |
| collection 目录 | `<data_root>/collections` | `src/python/hypervec_server_engine.py:172-173` | 是 |
| 元数据文件 | `<data_root>/collections.json` | `src/python/hypervec_server_engine.py:178` | 是 |
| 标量数据库 | `<data_root>/scalar.db` | `src/python/hypervec_server_engine.py:179` | 是 |
| engine logger | `hypervec.server` | `src/python/hypervec_server_engine.py:174` | 是 |
| collection 名称最大长度 | `255` | `src/python/hypervec_server_engine.py:186-195` | 否 |
| 默认 ID 字段 | `id` | `src/python/hypervec_server_engine.py:224-230` | 是 |
| 默认向量字段 | `vector` | `src/python/hypervec_server_engine.py:232-237` | 是 |
| 默认文本字段 | `contents` | `src/python/hypervec_server_engine.py:239-245` | 是 |
| 默认 `index_params` | `{"indexes": []}` | `src/python/hypervec_http_server.py:36` | 是 |
| 新 collection 版本 | `1` | `src/python/hypervec_meta_store.py:112` | 否 |
| 新 collection 维度 | `None` | `src/python/hypervec_meta_store.py:118` | 否 |
| 新 collection 数据量 | `0` | `src/python/hypervec_meta_store.py:119` | 否 |

## 7. HTTP 请求模型默认值

| 对象 | 字段 | 默认值 | 来源 |
|---|---|---|---|
| CreateCollectionRequest | `index_params` | `{"indexes": []}` | `src/python/hypervec_http_server.py:34-37` |
| SearchRequest | `search_params` | `{}` | `src/python/hypervec_http_server.py:41-45` |
| SearchRequest | `output_fields` | `[]` | `src/python/hypervec_http_server.py:41-45` |
| SearchRequest | `filter` | `""` | `src/python/hypervec_http_server.py:46` |
| SearchRequest | `consistency_level` | `None` | `src/python/hypervec_http_server.py:47` |
| upload index query | `version` | `None` | `src/python/hypervec_http_server.py:213` |
| upload index query | `checksum` | `None` | `src/python/hypervec_http_server.py:214` |

## 8. Python Client 默认值

| 名称 | 默认值 | 来源 |
|---|---|---|
| `HypervecClient.token` | `None` | `pyhypervec/pyhypervec/client.py:18-25` |
| `HypervecClient.timeout` | `30.0` 秒 | `pyhypervec/pyhypervec/client.py:20` |
| `HypervecClient.http2` | `False` | `pyhypervec/pyhypervec/client.py:21` |
| h2c 缺省端口 | `80` | `pyhypervec/pyhypervec/client.py:208` |
| h2c socket recv size | `65535` | `pyhypervec/pyhypervec/client.py:240,271` |
| schema `auto_id` | `False` | `pyhypervec/pyhypervec/schema.py:17` |
| schema `enable_dynamic_field` | `True` | `pyhypervec/pyhypervec/schema.py:18` |
| schema `description` | `""` | `pyhypervec/pyhypervec/schema.py:19` |
| index `metric_type` | `L2` | `pyhypervec/pyhypervec/schema.py:44` |
| index `index_type` | `HNSWFlat` | `pyhypervec/pyhypervec/schema.py:45` |
| index `params` | `{}` | `pyhypervec/pyhypervec/schema.py:46-54` |

## 9. Server 索引与搜索默认值

| 索引/场景 | 参数 | 默认值 | 来源 |
|---|---|---|---|
| fallback | `metric_type` | `L2` | `src/python/hypervec_server_engine.py:247-275` |
| fallback | `index_type` | `HNSWFlat` | `src/python/hypervec_server_engine.py:247-275` |
| IVF/IVFFlat | `nlist` | `1024` | `src/python/hypervec_server_engine.py:300-302` |
| IVFLVQ | `nlist` | `1024` | `src/python/hypervec_server_engine.py:303-307` |
| IVFLVQ | `nlocal` | `16` | `src/python/hypervec_server_engine.py:303-307` |
| IVFLVQ | `nbits` | `8` | `src/python/hypervec_server_engine.py:303-307` |
| IVFPQ | `nlist` | `1024` | `src/python/hypervec_server_engine.py:308-313` |
| IVFPQ | `m_pq` | `8` | `src/python/hypervec_server_engine.py:308-313` |
| IVFPQ | `nbits` | `8` | `src/python/hypervec_server_engine.py:308-313` |
| HNSWFlat | `m_hnsw` | `32` | `src/python/hypervec_server_engine.py:314-316` |
| HNSWLVQ | `nlocal` | `16` | `src/python/hypervec_server_engine.py:317-321` |
| HNSWLVQ | `nbits` | `8` | `src/python/hypervec_server_engine.py:317-321` |
| HNSWLVQ | `m_hnsw` | `32` | `src/python/hypervec_server_engine.py:317-321` |
| HNSWPQ | `m_pq` | `8` | `src/python/hypervec_server_engine.py:322-327` |
| HNSWPQ | `nbits` | `8` | `src/python/hypervec_server_engine.py:322-327` |
| HNSWPQ | `m_hnsw` | `32` | `src/python/hypervec_server_engine.py:322-327` |
| HNSW search | `ef_search`/`ef` | server 无默认，未传时使用 C++ 默认 | `src/python/hypervec_server_engine.py:333-344` |
| filter 前候选数 | multiplier | `8` | `src/python/hypervec_server_engine.py:574-576` |

## 10. C++ 库级可调默认值

| 参数 | 类型 | 默认值 | 作用 |
|---|---|---|---|
| `distance_compute_blas_threshold` | int | `20` | query 数达到阈值时切换 BLAS |
| `distance_compute_blas_query_bs` | int | `4096` | BLAS query block size |
| `distance_compute_blas_database_bs` | int | `1024` | BLAS database block size |
| `distance_compute_min_k_reservoir` | int | `100` | reservoir/heap 结果收集切换阈值 |
| `visited_table_hashset_threshold` | size_t | `500000` | HNSW visited table 切换 hash set 的规模阈值 |
| `bucket_sort_verbose` | int | `0` | bucket sort 诊断输出 |
| `RandomGenerator.seed` | int64 | `1234` | 随机数生成器种子 |

来源：`src/include/utils/distances/distances.h:271-280`、`src/include/index/hnsw/visited_table.h:22-33`、`src/include/utils/structures/random.h:41`。

## 11. C++ 算法默认值

### 11.1 KMeans / PQ / LVQ

| 参数组 | 字段 | 默认值 |
|---|---|---|
| `KMeansParameters` | `niter` | `25` |
| `KMeansParameters` | `seed` | `1234` |
| `KMeansParameters` | `nredo` | `1` |
| `KMeansParameters` | `verbose` | `false` |
| `KMeansParameters` | `spherical` | `false` |
| `KMeansParameters` | `metric` | `kMetricL2` |
| `KMeansParameters` | `metric_arg` | `0.0f` |
| `PQParameters` | `niter` | `25` |
| `PQParameters` | `seed` | `1234` |
| `PQParameters` | `nredo` | `1` |
| `PQParameters` | `verbose` | `false` |
| PQ | `HYPERVEC_PQ_MAX_NBITS` | `16` |
| `LVQParameters` | `niter` | `25` |
| `LVQParameters` | `seed` | `1234` |
| `LVQParameters` | `nredo` | `1` |
| `LVQParameters` | `verbose` | `false` |
| LVQ | `HYPERVEC_LVQ_MAX_NBITS` | `16` |

来源：`src/include/utils/algo/kmeans/kmeans.h`、`src/include/quantization/pq/pq.h`、`src/include/quantization/lvq/lvq.h`。

### 11.2 IVF / HNSW

| 参数组 | 字段 | 默认值 |
|---|---|---|
| `IVFSearchParameters` | `nprobe` | `1` |
| `SearchParametersHNSW` | `ef_search` | `16` |
| `SearchParametersHNSW` | `check_relative_distance` | `true` |
| `SearchParametersHNSW` | `bounded_queue` | `true` |
| `HNSW` | `entry_point` | `-1` |
| `HNSW` | `max_level` | `-1` |
| `HNSW` | `ef_construction` | `40` |
| `HNSW` | `ef_search` | `16` |
| `HNSW` | `check_relative_distance` | `true` |
| `HNSW` | `search_bounded_queue` | `true` |
| `HNSW` | `is_panorama` | `false` |
| `IndexHNSW` | `init_level0` | `true` |
| `IndexHNSW` | `keep_max_size_level0` | `false` |
| `IndexHNSW` constructor | `M` | `32` |
| `IndexHNSW` constructor | metric | `kMetricL2` |

来源：`src/include/index/ivf/index_ivf.h:20-23`、`src/include/index/hnsw/hnsw.h:54-58,134-156`、`src/include/index/hnsw/index_hnsw.h:40-55`。

## 12. 构建时默认值

| 参数 | 默认值 | 作用 | 来源 |
|---|---|---|---|
| `HYPERVEC_OPT_LEVEL` | CMake 字面默认 `""`，有效路径为 generic | 选择 generic/AVX2/AVX512/AVX512-SPR/SVE/DD | `CMakeLists.txt:27-28` |
| `HYPERVEC_ENABLE_MKL` | `OFF` | MKL 支持 | `CMakeLists.txt:29` |
| `HYPERVEC_ENABLE_PYTHON` | `OFF` | Python binding | `CMakeLists.txt:30` |
| `HYPERVEC_ENABLE_C_API` | `OFF` | C API | `CMakeLists.txt:31` |
| `HYPERVEC_ENABLE_EXTRAS` | `OFF` | demo/benchmark | `CMakeLists.txt:32` |
| `HYPERVEC_USE_LTO` | `OFF` | Link-Time Optimization | `CMakeLists.txt:33` |
| `BUILD_TESTING` | CTest 默认 `ON` | unit test/benchmark | `CMakeLists.txt:45-51` |
| core `CMAKE_CXX_STANDARD` | `17` | core C++ 标准 | `CMakeLists.txt:23` |
| Python `CMAKE_CXX_STANDARD` | `20` | binding C++ 标准 | `src/python/CMakeLists.txt:13` |

## 13. 作为统一 Config 的优先级候选

首期配置模块建议只收敛进程启动与日志参数：

| 分组 | 候选项 |
|---|---|
| Server | `data_root`、`host`、`port`、`server`、`enable_http2`、`certfile`、`keyfile` |
| Defaults | `default_index_type`、`default_metric_type`（本期只加载和访问） |
| Logging | `enable_logging`、`log_level`、`log_to_stderr`、`log_to_file`、`log_file_path` |

SIMD 选择、构建选项、HTTP 请求参数、客户端默认值和 C++ 算法参数应保持原有边界，后续只在有明确需求时再单独设计。
