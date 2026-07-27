# Runtime Parameter Inventory

## Implementation Status After Configuration Module

This file originally inventoried the pre-implementation parameter sources. That baseline remains below for traceability. The current server configuration is defined by `CONFIG_OPTIONS` in `src/python/hypervec_config.py` and materialized as `HypervecConfig`.

Current precedence:

```text
built-in defaults < INI configuration < explicit CLI options
```

Environment variables are not part of this precedence. Build and ARM wrapper variables retain their existing behavior; when a wrapper passes their values as CLI arguments, they take effect as explicit CLI values.

Current CLI additions are:

| Option | Purpose |
|---|---|
| `--config PATH` | Read the explicit INI file; no implicit file search is performed. |
| `--export-sample-config PATH` | Create a commented sample and exit without starting the server. |
| `--enable-http2` / `--no-enable-http2` | Control Hypercorn HTTP/2 protocol advertisement. |
| `--default-index-type TYPE` | Override the reserved collection default index type. |
| `--default-metric-type TYPE` | Override the reserved collection default metric type. |
| `--enable-logging` / `--no-enable-logging` | Explicitly enable or disable HyperVector Python logging. |
| `--log-to-stderr` / `--no-log-to-stderr` | Explicitly control the stderr handler. |
| `--log-to-file` / `--no-log-to-file` | Explicitly control the file handler. |
| `--log-file-path PATH` | Set the file handler destination. |

The original seven CLI names remain supported. Their effective defaults now come from `CONFIG_OPTIONS`, not from `argparse`. The complete current option/default/validation table is maintained in `docs/configuration.md`; the generated golden sample is `configs/hypervec.ini.sample`. Configuration is startup-only and does not support hot reload.

## Pre-Implementation CLI Parameters

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `--data-root` | string/path | Required; no default | Collection data root. The engine creates metadata, scalar DB, and collection index files under this directory. | `src/python/hypervec_http_server.py:245` | Yes |
| `--host` | string | `"127.0.0.1"` | Bind host for the ASGI server. | `src/python/hypervec_http_server.py:246` | Yes |
| `--port` | int | `8080` | Bind port for the ASGI server. | `src/python/hypervec_http_server.py:247` | Yes |
| `--server` | enum string | `"hypercorn"` | ASGI server implementation. Choices are `hypercorn` and `uvicorn`; Hypercorn is default because it supports HTTP/2. | `src/python/hypervec_http_server.py:248-253` | Yes |
| `--log-level` | string | `"info"` | ASGI server log level passed to Hypercorn or Uvicorn. | `src/python/hypervec_http_server.py:254`, `src/python/hypervec_http_server.py:275`, `src/python/hypervec_http_server.py:280`, `src/python/hypervec_http_server.py:298` | Yes |
| `--certfile` | string/path or null | `None` | TLS certificate file for HTTP/2 over TLS. Must be supplied with `--keyfile`. | `src/python/hypervec_http_server.py:255`, `src/python/hypervec_http_server.py:268-277`, `src/python/hypervec_http_server.py:293-300` | Yes |
| `--keyfile` | string/path or null | `None` | TLS private key file for HTTP/2 over TLS. Must be supplied with `--certfile`. | `src/python/hypervec_http_server.py:256`, `src/python/hypervec_http_server.py:268-277`, `src/python/hypervec_http_server.py:293-300` | Yes |

# Environment Variables

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `HYPERVEC_DISABLE_CPU_FEATURES` | string list | `""` | Comma/whitespace-separated CPU feature names to remove from Python loader CPU-feature detection. Used before selecting optimized Python extension modules. | `src/python/loader.py:62-63`, `src/python/loader.py:82-83` | Yes |
| `HYPERVEC_OPT_LEVEL` | string | unset / `None` in Python loader | Forces Python loader optimization level instead of auto-detecting CPU features. Recognized values in messages include `avx512_spr`, `avx512`, `AVX2`, `SVE`; build scripts use lowercase values. | `src/python/loader.py:91-103`, `src/python/loader.py:165-193` | Yes |
| `HYPERVEC_SIMD_LEVEL` | string enum | unset => auto-detect | C++ dynamic-dispatch runtime SIMD override. Only active in builds compiled with `HYPERVEC_ENABLE_DD`. Valid strings are `NONE`, `AVX2`, `AVX512`, `AVX512_SPR`, `ARM_NEON`, `ARM_SVE`. | `src/utils/simd/simd_levels.cpp:65-77`, `src/utils/simd/simd_levels.cpp:296-317` | Yes |
| `BUILD_DIR` | string/path | `build-arm` | Build directory for ARM server build script. | `scripts/build_arm_pyhypervec_server.sh:5` | No |
| `VENV_DIR` | string/path | `.venv` | Python virtual environment path used by build script. | `scripts/build_arm_pyhypervec_server.sh:6` | No |
| `PYTHON_BIN` | string/path | `python3` | Python executable used to create the virtualenv. | `scripts/build_arm_pyhypervec_server.sh:7` | No |
| `HYPERVEC_OPT_LEVEL` | string | `generic` | Build optimization level passed to CMake as `-DHYPERVEC_OPT_LEVEL`. | `scripts/build_arm_pyhypervec_server.sh:8`, `scripts/build_arm_pyhypervec_server.sh:121` | No |
| `CMAKE_GENERATOR` | string | `Ninja` | CMake generator used by the ARM build/server script. | `scripts/build_arm_pyhypervec_server.sh:9`, `scripts/build_arm_pyhypervec_server.sh:116` | No |
| `INSTALL_PYHYPERVEC` | bool-like string | `1` | Whether to install the pure Python `pyhypervec` client package. | `scripts/build_arm_pyhypervec_server.sh:10`, `scripts/build_arm_pyhypervec_server.sh:142` | No |
| `INSTALL_SERVER_DEPS` | bool-like string | `1` | Whether to install `fastapi`, `uvicorn`, `hypercorn`, and `h2`. | `scripts/build_arm_pyhypervec_server.sh:11`, `scripts/build_arm_pyhypervec_server.sh:97-100` | No |
| `INSTALL_CMAKE` | bool-like string | `1` | Whether to install `cmake>=3.24` into the virtualenv. | `scripts/build_arm_pyhypervec_server.sh:12`, `scripts/build_arm_pyhypervec_server.sh:78-81` | No |
| `DATA_ROOT` | string/path | `${HOME}/hypervec_data` | Server data root used by the build script and passed to `--data-root` when starting the server. | `scripts/build_arm_pyhypervec_server.sh:13`, `scripts/build_arm_pyhypervec_server.sh:193-195` | Yes |
| `SERVER_HOST` | string | `0.0.0.0` | Server bind host passed to `--host` when `START_SERVER=1`. | `scripts/build_arm_pyhypervec_server.sh:14`, `scripts/build_arm_pyhypervec_server.sh:195` | Yes |
| `SERVER_PORT` | int-like string | `8080` | Server bind port passed to `--port` when `START_SERVER=1`. | `scripts/build_arm_pyhypervec_server.sh:15`, `scripts/build_arm_pyhypervec_server.sh:196` | Yes |
| `SERVER_IMPL` | enum string | `hypercorn` | ASGI server implementation passed to `--server` when `START_SERVER=1`. | `scripts/build_arm_pyhypervec_server.sh:16`, `scripts/build_arm_pyhypervec_server.sh:197` | Yes |
| `START_SERVER` | bool-like string | `0` | Whether the build script `exec`s the HTTP server after build/install verification. | `scripts/build_arm_pyhypervec_server.sh:17`, `scripts/build_arm_pyhypervec_server.sh:191` | No |
| `MKLROOT` | string/path | unset | Optional MKL installation root used by `cmake/FindMKL.cmake` when discovering MKL libraries. | `cmake/FindMKL.cmake:64`, `cmake/FindMKL.cmake:317-348` | No |
| `PATH` | string | process environment | `build.sh` prepends Miniconda bin path when it installs toolchain dependencies. | `build.sh:131` | No |
| `CC` | string/path | set by script | C compiler selected by `build.sh` when using Miniconda or system GCC. | `build.sh:132`, `build.sh:183` | No |
| `CXX` | string/path | set by script | C++ compiler selected by `build.sh` when using Miniconda or system G++. | `build.sh:133`, `build.sh:184` | No |

# Configuration Files

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `pyhypervec/pyproject.toml` | TOML | N/A | Package metadata for the pure Python HTTP client: name, version, dependencies, Python requirement, setuptools package discovery. It is not runtime server config. | `pyhypervec/pyproject.toml:1-16` | No |
| `<data_root>/collections.json` | JSON | generated if needed | Runtime metadata store generated and maintained by the server. Contains collection metadata such as schema, index params, version, index path, checksum, counts, and timestamps. It is state, not startup config. | `src/python/hypervec_server_engine.py:178`, `src/python/hypervec_meta_store.py:59-85` | No |
| `<data_root>/scalar.db` | SQLite | generated if needed | Runtime scalar/vector row store generated and maintained by the server. | `src/python/hypervec_server_engine.py:179`, `src/python/hypervec_scalar_store.py:19-31` | No |
| `<data_root>/collections/<collection>/index.hypervec` | binary index file | generated by flush/upload | Serialized HyperVec index generated by flush/upload paths. | `src/python/hypervec_server_engine.py:31`, `src/python/hypervec_server_engine.py:205-207`, `src/python/hypervec_server_engine.py:346-353` | No |

# Hard-coded Default Values

## HTTP Server and App Defaults

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| FastAPI title | string | `"HyperVec HTTP Server"` | App title. | `src/python/hypervec_http_server.py:62` | No |
| FastAPI version | string | `"1"` | App version exposed by FastAPI metadata. | `src/python/hypervec_http_server.py:62` | No |
| Root logging level | int/log level | `logging.INFO` | Root logging initialization. Independent from `--log-level`, which is passed to the ASGI server. | `src/python/hypervec_http_server.py:259` | Yes |
| Hypercorn bind | list string | `[f"{host}:{port}"]` | Derived from CLI host/port. | `src/python/hypervec_http_server.py:296-298` | Yes |
| Hypercorn ALPN protocols | list string | `["h2", "http/1.1"]` | Enables HTTP/2 and HTTP/1.1 when TLS is configured; Hypercorn also supports h2c. | `src/python/hypervec_http_server.py:301` | Yes |

## Server Engine Storage Defaults

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `INDEX_FILE` | string | `"index.hypervec"` | File name used for serialized collection index. | `src/python/hypervec_server_engine.py:31`, `src/python/hypervec_server_engine.py:205-207` | Yes |
| `collections_root` | path | `<data_root>/collections` | Directory for per-collection data. | `src/python/hypervec_server_engine.py:172-173` | Yes |
| metadata file | path | `<data_root>/collections.json` | Metadata store path. | `src/python/hypervec_server_engine.py:178` | Yes |
| scalar DB file | path | `<data_root>/scalar.db` | SQLite scalar/vector store path. | `src/python/hypervec_server_engine.py:179` | Yes |
| engine logger name | string | `"hypervec.server"` | Logger used when no logger is injected. | `src/python/hypervec_server_engine.py:174` | Yes |
| max collection name length | int | `255` | Collection names must be alphanumeric, underscore, hyphen, and at most this length. | `src/python/hypervec_server_engine.py:186-195` | No |
| default id field | string | `"id"` | Fallback primary ID field. | `src/python/hypervec_server_engine.py:224-230`, `src/python/hypervec_meta_store.py:44` | Yes |
| default vector field | string | `"vector"` | Fallback vector field. | `src/python/hypervec_server_engine.py:232-237`, `src/python/hypervec_meta_store.py:45` | Yes |
| default text field | string | `"contents"` | Fallback text field. | `src/python/hypervec_server_engine.py:239-245`, `src/python/hypervec_meta_store.py:46` | Yes |
| default metadata `index_params` | dict | `{"indexes": []}` | Used when no index params are supplied. | `src/python/hypervec_http_server.py:36`, `src/python/hypervec_server_engine.py:420-424`, `src/python/hypervec_meta_store.py:43` | Yes |
| new collection version | int | `1` | Initial collection metadata version. | `src/python/hypervec_meta_store.py:41`, `src/python/hypervec_meta_store.py:112` | No |
| new collection `dim` | int/null | `None` | Dimension unknown until first insert. | `src/python/hypervec_meta_store.py:118` | No |
| new collection `total` | int | `0` | Initial row count. | `src/python/hypervec_meta_store.py:48`, `src/python/hypervec_meta_store.py:119` | No |

## HTTP Request/Model Defaults

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| create request `index_params` | dict | `{"indexes": []}` | Pydantic request default when creating a collection. | `src/python/hypervec_http_server.py:34-37` | Yes |
| search `search_params` | dict | `{}` | Request default. | `src/python/hypervec_http_server.py:41-45` | No |
| search `output_fields` | list | `[]` | Request default. | `src/python/hypervec_http_server.py:41-45` | No |
| search `filter` | string | `""` | Request default. | `src/python/hypervec_http_server.py:46` | No |
| search `consistency_level` | string/null | `None` | Accepted but currently discarded by engine. | `src/python/hypervec_http_server.py:47`, `src/python/hypervec_server_engine.py:556-558` | No |
| upload `version` query | int/null | `None` | Optional uploaded index version. | `src/python/hypervec_http_server.py:213` | No |
| upload `checksum` query | string/null | `None` | Optional uploaded index checksum. | `src/python/hypervec_http_server.py:214` | No |

## Python Client Defaults

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `HypervecClient.token` | string/null | `None` | Optional bearer token. Server currently has no auth enforcement. | `pyhypervec/pyhypervec/client.py:18-25` | Yes |
| `HypervecClient.timeout` | float | `30.0` | URL, socket, and httpx timeout. | `pyhypervec/pyhypervec/client.py:20`, `pyhypervec/pyhypervec/client.py:79`, `pyhypervec/pyhypervec/client.py:176`, `pyhypervec/pyhypervec/client.py:219-220` | Yes |
| `HypervecClient.http2` | bool | `False` | Whether to use HTTP/2 paths. | `pyhypervec/pyhypervec/client.py:21`, `pyhypervec/pyhypervec/client.py:51`, `pyhypervec/pyhypervec/client.py:102` | Yes |
| HTTP/2 cleartext default port | int | `80` | Used when parsed URI has no port in h2c mode. | `pyhypervec/pyhypervec/client.py:208` | Yes |
| h2c socket recv size | int | `65535` | Receive buffer size during HTTP/2 frame loop. | `pyhypervec/pyhypervec/client.py:240`, `pyhypervec/pyhypervec/client.py:271` | No |
| schema `auto_id` | bool | `False` | Default collection schema value. | `pyhypervec/pyhypervec/schema.py:17`, `pyhypervec/pyhypervec/client.py:31` | No |
| schema `enable_dynamic_field` | bool | `True` | Default collection schema value. | `pyhypervec/pyhypervec/schema.py:18`, `pyhypervec/pyhypervec/client.py:32` | No |
| schema `description` | string | `""` | Default collection schema description. | `pyhypervec/pyhypervec/schema.py:19`, `pyhypervec/pyhypervec/client.py:33` | No |
| index params `metric_type` | string | `"L2"` | Default client index metric. | `pyhypervec/pyhypervec/schema.py:44` | Yes |
| index params `index_type` | string | `"HNSWFlat"` | Default client index type. | `pyhypervec/pyhypervec/schema.py:45` | Yes |
| index params `params` | dict | `{}` | Default client index params. | `pyhypervec/pyhypervec/schema.py:46-54` | Yes |

## Index and Search Defaults Exposed Through the Server

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| engine fallback metric type | string | `"L2"` | Used when an index config lacks `metric_type`; accepted aliases include IP, INNER_PRODUCT, COSINE, L2, EUCLIDEAN. | `src/python/hypervec_server_engine.py:247-256`, `src/python/hypervec_server_engine.py:265-275` | Yes |
| engine fallback index type | string | `"HNSWFlat"` | Used when no index config is supplied. | `src/python/hypervec_server_engine.py:247-256`, `src/python/hypervec_server_engine.py:275` | Yes |
| IVF/IVFFlat `nlist` | int | `1024` | Default cluster/list count in server index creation. | `src/python/hypervec_server_engine.py:300-302` | Yes |
| IVFLVQ `nlist` | int | `1024` | Default cluster/list count. | `src/python/hypervec_server_engine.py:303-307` | Yes |
| IVFLVQ `nlocal` | int | `16` | Default local quantization parameter. | `src/python/hypervec_server_engine.py:303-307` | Yes |
| IVFLVQ `nbits` | int | `8` | Default quantization bits. | `src/python/hypervec_server_engine.py:303-307` | Yes |
| IVFPQ `nlist` | int | `1024` | Default cluster/list count. | `src/python/hypervec_server_engine.py:308-313` | Yes |
| IVFPQ `m_pq` | int | `8` | Default number of product quantizer subquantizers. | `src/python/hypervec_server_engine.py:308-313` | Yes |
| IVFPQ `nbits` | int | `8` | Default code bits per subquantizer. | `src/python/hypervec_server_engine.py:308-313` | Yes |
| HNSWFlat `m_hnsw` | int | `32` | Default HNSW graph connection parameter. | `src/python/hypervec_server_engine.py:314-316` | Yes |
| HNSWLVQ `nlocal` | int | `16` | Default LVQ local count. | `src/python/hypervec_server_engine.py:317-321` | Yes |
| HNSWLVQ `nbits` | int | `8` | Default LVQ bits. | `src/python/hypervec_server_engine.py:317-321` | Yes |
| HNSWLVQ `m_hnsw` | int | `32` | Default HNSW graph connection parameter. | `src/python/hypervec_server_engine.py:317-321` | Yes |
| HNSWPQ `m_pq` | int | `8` | Default PQ subquantizers. | `src/python/hypervec_server_engine.py:322-327` | Yes |
| HNSWPQ `nbits` | int | `8` | Default PQ bits. | `src/python/hypervec_server_engine.py:322-327` | Yes |
| HNSWPQ `m_hnsw` | int | `32` | Default HNSW graph connection parameter. | `src/python/hypervec_server_engine.py:322-327` | Yes |
| search `ef_search` alias | int/null | no server default; C++ default applies | Server accepts `ef_search` or `ef` and calls `search_with_ef` when supported. | `src/python/hypervec_server_engine.py:333-344` | Yes |
| candidate multiplier | int | `8` | Server searches up to `max(limit, limit * 8)` candidates before filtering. | `src/python/hypervec_server_engine.py:574-576` | Yes |

## C++ Library Runtime-Tunable Defaults

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `distance_compute_blas_threshold` | int | `20` | Threshold on query count above which distance computation switches to BLAS. | `src/include/utils/distances/distances.h:271-272`, `src/utils/distances/distances.cpp:769-799` | Yes |
| `distance_compute_blas_query_bs` | int | `4096` | Query block size for BLAS distance computation. | `src/include/utils/distances/distances.h:274-276`, `src/utils/distances/distances.cpp:282-283`, `src/utils/distances/distances.cpp:796-798` | Yes |
| `distance_compute_blas_database_bs` | int | `1024` | Database block size for BLAS distance computation. | `src/include/utils/distances/distances.h:274-276`, `src/utils/distances/distances.cpp:282-283`, `src/utils/distances/distances.cpp:796-798` | Yes |
| `distance_compute_min_k_reservoir` | int | `100` | Result count threshold where reservoir collection is used instead of heap. | `src/include/utils/distances/distances.h:278-280`, `src/include/utils/common/result_handler.h:678-688`, `src/utils/distances/distances.cpp:799` | Yes |
| `visited_table_hashset_threshold` | size_t | `500000` | HNSW visited table switches to hash set when graph size is at or above this threshold unless explicitly overridden. | `src/include/index/hnsw/visited_table.h:22-33`, `src/index/hnsw/visited_table.cpp:15-24` | Yes |
| `bucket_sort_verbose` | int | `0` | Global verbosity flag for bucket sort diagnostics. | `src/include/utils/structures/sorting.h:28`, `src/utils/structures/sorting.cpp:204` | No |
| `RandomGenerator.seed` | int64 | `1234` | Default seed for random generator construction. | `src/include/utils/structures/random.h:41` | Yes |

## C++ Algorithm Defaults

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `KMeansParameters.niter` | int | `25` | Lloyd iterations per redo. | `src/include/utils/algo/kmeans/kmeans.h:21-29` | Yes |
| `KMeansParameters.seed` | int | `1234` | RNG seed for centroid initialization. | `src/include/utils/algo/kmeans/kmeans.h:16-32` | Yes |
| `KMeansParameters.nredo` | int | `1` | Number of random restarts. | `src/include/utils/algo/kmeans/kmeans.h:34-36` | Yes |
| `KMeansParameters.verbose` | bool | `false` | Print per-iteration objective and summaries to stderr. | `src/include/utils/algo/kmeans/kmeans.h:38-39` | Yes |
| `KMeansParameters.spherical` | bool | `false` | Normalize centroids after each iteration; forced for inner-product metric. | `src/include/utils/algo/kmeans/kmeans.h:41-44` | Yes |
| `KMeansParameters.metric` | enum | `kMetricL2` | Assignment/objective metric. | `src/include/utils/algo/kmeans/kmeans.h:46-58` | Yes |
| `KMeansParameters.metric_arg` | float | `0.0f` | Reserved for parameterized metrics. | `src/include/utils/algo/kmeans/kmeans.h:60-62` | No |
| `PQParameters.niter` | int | `25` | Lloyd iterations per subquantizer. | `src/include/quantization/pq/pq.h:25-46` | Yes |
| `PQParameters.seed` | int | `1234` | Base RNG seed for subquantizers. | `src/include/quantization/pq/pq.h:20-23`, `src/include/quantization/pq/pq.h:48-51` | Yes |
| `PQParameters.nredo` | int | `1` | Random restarts per subquantizer. | `src/include/quantization/pq/pq.h:30-55` | Yes |
| `PQParameters.verbose` | bool | `false` | Print per-subquantizer progress to stderr. | `src/include/quantization/pq/pq.h:57-58` | Yes |
| `HYPERVEC_PQ_MAX_NBITS` | int | `16` | Maximum supported PQ bits. | `src/include/quantization/pq/pq.h:36-39` | No |
| `LVQParameters.niter` | int | `25` | Lloyd iterations for LVQ training. | `src/include/quantization/lvq/lvq.h:18-25` | Yes |
| `LVQParameters.seed` | int | `1234` | RNG seed for LVQ training. | `src/include/quantization/lvq/lvq.h:18-26` | Yes |
| `LVQParameters.nredo` | int | `1` | Random restarts for LVQ training. | `src/include/quantization/lvq/lvq.h:20-26` | Yes |
| `LVQParameters.verbose` | bool | `false` | LVQ training verbosity. | `src/include/quantization/lvq/lvq.h:27` | Yes |
| `HYPERVEC_LVQ_MAX_NBITS` | int | `16` | Maximum supported LVQ bits. | `src/include/quantization/lvq/lvq.h:21` | No |
| `IVFSearchParameters.nprobe` | idx_t | `1` | Default number of IVF lists to probe. | `src/include/index/ivf/index_ivf.h:20-23`, `src/index/ivf/index_ivf.cpp:25-30` | Yes |
| `HNSW.SearchParametersHNSW.ef_search` | int | `16` | Search expansion factor override. | `src/include/index/hnsw/hnsw.h:54-58` | Yes |
| `HNSW.SearchParametersHNSW.check_relative_distance` | bool | `true` | Relative distance stopping behavior. | `src/include/index/hnsw/hnsw.h:54-58` | Yes |
| `HNSW.SearchParametersHNSW.bounded_queue` | bool | `true` | Use bounded queue during HNSW exploration. | `src/include/index/hnsw/hnsw.h:54-58` | Yes |
| `HNSW.entry_point` | int | `-1` | Initial graph entry point sentinel. | `src/include/index/hnsw/hnsw.h:134-136` | No |
| `HNSW.max_level` | int | `-1` | Initial graph max level sentinel. | `src/include/index/hnsw/hnsw.h:140-141` | No |
| `HNSW.ef_construction` | int | `40` | Expansion factor during construction. | `src/include/index/hnsw/hnsw.h:143-145` | Yes |
| `HNSW.ef_search` | int | `16` | Expansion factor during search. | `src/include/index/hnsw/hnsw.h:146-147` | Yes |
| `HNSW.check_relative_distance` | bool | `true` | Default relative distance check during search. | `src/include/index/hnsw/hnsw.h:149-151` | Yes |
| `HNSW.search_bounded_queue` | bool | `true` | Default bounded queue during search. | `src/include/index/hnsw/hnsw.h:153-154` | Yes |
| `HNSW.is_panorama` | bool | `false` | Internal mode flag. | `src/include/index/hnsw/hnsw.h:156` | No |
| `IndexHNSW.init_level0` | bool | `true` | Whether to initialize level 0 graph. | `src/include/index/hnsw/index_hnsw.h:40-43` | No |
| `IndexHNSW.keep_max_size_level0` | bool | `false` | Whether to fill all level 0 neighbor slots. | `src/include/index/hnsw/index_hnsw.h:45-49` | No |
| `IndexHNSW` constructor `M` | int | `32` | Default HNSW connection parameter. | `src/include/index/hnsw/index_hnsw.h:54-55` | Yes |
| `IndexHNSW` constructor metric | enum | `kMetricL2` | Default HNSW metric. | `src/include/index/hnsw/index_hnsw.h:54` | Yes |

## Build-Time Defaults That Affect Runtime Artifacts

| Name | Type | Default Value | Description | Source File | Recommended for future Config module |
|---|---|---|---|---|---|
| `HYPERVEC_OPT_LEVEL` CMake option | string | literal CMake default `""`; effective generic path | Selects generic, AVX2, AVX512, AVX512-SPR, SVE, or DD code paths and libraries. Values not matching specialized modes use the generic `hypervec` target. | `CMakeLists.txt:27-28`, `cmake/link_to_hypervec_lib.cmake:7-87`, `src/python/CMakeLists.txt:43-47`, `src/python/CMakeLists.txt:108-145` | No |
| `HYPERVEC_ENABLE_MKL` | bool | `OFF` | Build with MKL support. | `CMakeLists.txt:29` | No |
| `HYPERVEC_ENABLE_PYTHON` | bool | `OFF` | Build Python extension and HTTP server package files. | `CMakeLists.txt:30`, `CMakeLists.txt:37-39` | No |
| `HYPERVEC_ENABLE_C_API` | bool | `OFF` | Build C API. | `CMakeLists.txt:31` | No |
| `HYPERVEC_ENABLE_EXTRAS` | bool | `OFF` | Build demos/benchmarks extras. | `CMakeLists.txt:32`, `CMakeLists.txt:41-43` | No |
| `HYPERVEC_USE_LTO` | bool | `OFF` | Link-time optimization switch. | `CMakeLists.txt:33` | No |
| `BUILD_TESTING` | bool | CTest default `ON` after `include(CTest)` | Enables unit tests and benchmarks unless disabled. | `CMakeLists.txt:45-51` | No |
| `CMAKE_CXX_STANDARD` top-level | int | `17` | C++ standard for core build. | `CMakeLists.txt:23` | No |
| `CMAKE_CXX_STANDARD` Python subdir | int | `20` | C++ standard for Python binding build. | `src/python/CMakeLists.txt:13` | No |
