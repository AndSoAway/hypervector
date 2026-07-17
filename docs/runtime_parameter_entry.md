# Runtime Parameter Entry

Investigation source: split from the completed repository investigation. No new repository scan was performed for this reorganization.

## Implementation Status After Configuration Module

The investigation below records the pre-implementation startup path and is retained as historical input. The current HTTP Server flow is now:

```text
build_argument_parser()
    -> argparse Namespace with only explicit business CLI values
    -> resolve_config(defaults, optional INI, CLI overrides)
    -> HypervecConfig(ServerConfig, IndexDefaultsConfig, LoggingConfig)
    -> configure_logging()
    -> run_server()
    -> create_app(data_root=config.server.data_root)
    -> Uvicorn or Hypercorn
```

Current implementation points:

- `src/python/hypervec_config.py` owns `CONFIG_OPTIONS`, typed config objects, INI parsing, validation, logging setup, and sample export.
- `src/python/hypervec_http_server.py` provides `build_argument_parser()`, `run_server()`, and testable `main(argv=None)`.
- Startup precedence is `built-in defaults < INI configuration < explicit CLI options`.
- Environment variables are not a fourth configuration source. The ARM wrapper may translate its environment values into explicit CLI arguments.
- Configuration is loaded once during startup. Hot reload, SIGHUP reload, and file watching are not supported.
- `enable_http2` controls Hypercorn protocol advertisement; `default_index_type` and `default_metric_type` are validated reserved values exposed through `HypervecConfig.defaults`.

The old statements such as “there is no unified runtime Config module” and the old source line references in the remaining sections describe the investigation baseline, not the current implementation.

## Program Entry Points

### HTTP Server Entry Point

The main executable entry point is `src/python/hypervec_http_server.py`.

- `main()` is defined at `src/python/hypervec_http_server.py:243`.
- CLI parsing uses Python `argparse` at `src/python/hypervec_http_server.py:244-257`.
- `if __name__ == "__main__": main()` is at `src/python/hypervec_http_server.py:305-306`.
- The FastAPI app is built by `create_app(data_root=...)` at `src/python/hypervec_http_server.py:26-240`.
- The ASGI server is selected by `--server` and initialized at `src/python/hypervec_http_server.py:262-302`.

### C++ Example Entry Points

The repository has C++ demo executables with `main()` but no runtime CLI parsing:

- `test/examples/cpp/demo_ivflvq_indexing.cpp:23`
- `test/examples/cpp/demo_hnsw_lvq_indexing.cpp:22`
- `test/examples/cpp/demo_lvq_indexing.cpp:22`
- `test/examples/cpp/demo_hnsw_pq_indexing.cpp:42`
- `test/examples/cpp/demo_ivf_indexing.cpp:42`
- `test/examples/cpp/demo_hnsw_graph_search.cpp:22`
- `test/examples/cpp/demo_ivfpq_indexing.cpp:44`
- `test/examples/cpp/demo_hnsw_indexing.cpp:22`

### Python Test Entry Point

`test/unit_tests/python/test_simd_dispatch.py:129-130` has a local `unittest.main()` entry point for direct test execution.

## Startup Flow

Primary server startup path:

```text
python -m hypervec.hypervec_http_server
    |
    v
src/python/hypervec_http_server.py:main()
    |
    v
argparse parses CLI arguments
    |
    v
logging.basicConfig(level=logging.INFO)
    |
    v
create_app(data_root=args.data_root)
    |
    v
HypervecServerEngine(data_root)
    |
    v
data_root/
  collections.json
  scalar.db
  collections/<collection>/index.hypervec
    |
    v
ASGI server initialization
  hypercorn Config or uvicorn.run()
```

ARM deployment script startup path:

```text
bash scripts/build_arm_pyhypervec_server.sh
    |
    v
environment variables read by shell script
    |
    v
virtualenv / dependency / CMake build setup
    |
    v
optional START_SERVER=1
    |
    v
exec python -m hypervec.hypervec_http_server
  --data-root "$DATA_ROOT"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --server "$SERVER_IMPL"
```

Python import/runtime SIMD path:

```text
import hypervec
    |
    v
src/python/loader.py
    |
    v
HYPERVEC_OPT_LEVEL or CPU feature auto-detection
    |
    v
load swighypervec_avx512_spr / avx512 / avx2 / sve / generic module
```

C++ dynamic dispatch runtime path, only when built with `HYPERVEC_ENABLE_DD`:

```text
library load
    |
    v
static SIMDConfig initializer
    |
    v
HYPERVEC_SIMD_LEVEL or CPU SIMD auto-detection
    |
    v
SIMD dispatch level used by dispatch wrappers
```

## CLI Parsing Flow

Runtime CLI parsing is currently concentrated in `src/python/hypervec_http_server.py:243-257`.

```text
main()
    |
    v
argparse.ArgumentParser(description="Run the HyperVec HTTP server.")
    |
    +--> --data-root, required
    +--> --host, default "127.0.0.1"
    +--> --port, default 8080
    +--> --server, choices ("hypercorn", "uvicorn")
    +--> --log-level, default "info"
    +--> --certfile, default None
    +--> --keyfile, default None
    |
    v
args = parser.parse_args()
```

The parsed values are used directly inside `main()`; there is no intermediate project-level config object.

## Runtime Configuration Flow

1. `argparse` creates `args` in `src/python/hypervec_http_server.py:244-257`.
2. Logging is initialized immediately with `logging.basicConfig(level=logging.INFO)` at `src/python/hypervec_http_server.py:259`.
3. `args.data_root` flows into `create_app(data_root=args.data_root)` at `src/python/hypervec_http_server.py:260`.
4. `create_app()` constructs `HypervecServerEngine(data_root)` if no engine is injected at `src/python/hypervec_http_server.py:26-32`.
5. `HypervecServerEngine.__init__()` expands and creates storage paths under `data_root` at `src/python/hypervec_server_engine.py:161-182`.
6. HTTP request payload defaults are defined in Pydantic models inside `create_app()` at `src/python/hypervec_http_server.py:34-51`.
7. Collection creation passes request schema and index params into `engine.create_collection()` at `src/python/hypervec_http_server.py:104-114`.
8. `engine.create_collection()` persists `index_params`, derives field names, and writes metadata through `MetaStore` at `src/python/hypervec_server_engine.py:407-430`.
9. `engine.flush()` materializes the configured index using `_make_index()` and writes `index.hypervec` at `src/python/hypervec_server_engine.py:489-521`.
10. Search request options flow into `engine.search()` at `src/python/hypervec_http_server.py:153-166`.
11. `engine.search()` uses `search_params` only for HNSW `ef_search` / `ef` at `src/python/hypervec_server_engine.py:333-344`, then filters and truncates results at `src/python/hypervec_server_engine.py:574-607`.
12. Server network options flow to either `uvicorn.run()` at `src/python/hypervec_http_server.py:262-281` or Hypercorn `Config` at `src/python/hypervec_http_server.py:283-302`.

## HTTP Server Initialization

Uvicorn path:

```text
if args.server == "uvicorn":
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, ...)
```

Source: `src/python/hypervec_http_server.py:262-281`.

Hypercorn path:

```text
config = Config()
config.bind = [f"{args.host}:{args.port}"]
config.loglevel = args.log_level
config.certfile = args.certfile
config.keyfile = args.keyfile
config.alpn_protocols = ["h2", "http/1.1"]
asyncio.run(serve(app, config))
```

Source: `src/python/hypervec_http_server.py:283-302`.

TLS validation requires `--certfile` and `--keyfile` to be supplied together for both Uvicorn and Hypercorn (`src/python/hypervec_http_server.py:268-270`, `src/python/hypervec_http_server.py:293-294`).

## Logging Initialization

Logging is currently split:

- Python root logging is initialized with `logging.basicConfig(level=logging.INFO)` at `src/python/hypervec_http_server.py:259`.
- ASGI server logging is controlled by CLI `--log-level`, default `"info"`, and passed to Uvicorn or Hypercorn at `src/python/hypervec_http_server.py:254`, `src/python/hypervec_http_server.py:275`, `src/python/hypervec_http_server.py:280`, and `src/python/hypervec_http_server.py:298`.
- The server engine uses logger name `"hypervec.server"` unless a logger is injected (`src/python/hypervec_server_engine.py:165`, `src/python/hypervec_server_engine.py:174`).
- The Python SIMD loader uses `logging.getLogger(__name__)` and emits debug/info/error logs during module selection (`src/python/loader.py:87-100`, `src/python/loader.py:108-196`).
- C++ algorithm verbosity is mostly independent of Python logging; for example k-means/PQ/LVQ verbose flags print progress to stderr.

## Existing Configuration Abstraction

There is no unified runtime Config module today.

Existing partial abstractions:

- `argparse.Namespace` in `hypervec_http_server.main()` is the only server startup config container.
- Hypercorn has its own `hypercorn.config.Config`, but it is created locally and not exposed as a project-level abstraction.
- `HypervecServerEngine.__init__()` accepts dependency injection for `logger`, `hypervec_module`, `meta_store`, and `scalar_store`, but not a formal config object.
- `CollectionSchema` and `IndexParams` in `pyhypervec/pyhypervec/schema.py` are request/client-side API builders, not process configuration.
- `KMeansParameters`, `PQParameters`, `LVQParameters`, `SearchParametersHNSW`, and `IVFSearchParameters` are C++ algorithm parameter structs, not application-level config.
- `SIMDConfig` in `src/include/utils/simd/simd_levels.h` / `src/utils/simd/simd_levels.cpp` controls C++ dynamic-dispatch runtime SIMD selection, but only for DD builds and only for SIMD behavior.

## Overall Architecture Diagram

```text
main()
  src/python/hypervec_http_server.py:243
    |
    v
CLI Parsing
  argparse definitions at lines 244-257
    |
    +--> data_root
    |      |
    |      v
    |   create_app(data_root)
    |      |
    |      v
    |   HypervecServerEngine(data_root)
    |      |
    |      +--> collections_root = data_root / "collections"
    |      +--> MetaStore(data_root / "collections.json")
    |      +--> ScalarStore(data_root / "scalar.db")
    |      +--> index path = collections/<collection>/index.hypervec
    |
    +--> host / port / server / log_level / certfile / keyfile
           |
           v
        HTTP Server Initialization
           |
           +--> uvicorn.run(app, host, port, log_level, ssl...)
           |
           +--> hypercorn Config(bind, loglevel, certfile, keyfile, alpn)

import hypervec
    |
    v
Python loader
    |
    +--> HYPERVEC_OPT_LEVEL
    +--> HYPERVEC_DISABLE_CPU_FEATURES
    |
    v
selected Python extension module

C++ library load in DD builds
    |
    v
SIMDConfig static initializer
    |
    +--> HYPERVEC_SIMD_LEVEL
    +--> CPU auto-detect
    |
    v
SIMD dispatch wrappers
```
