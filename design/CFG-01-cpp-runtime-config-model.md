# CFG-01: C++ typed runtime config model and metadata

Status: implemented in PR #44. Follow-up items tracked in issue #106.

## Goal

Establish the C++ config model and its centralized metadata table. This stage
does not read configuration files, does not talk to Python and does not change
the server startup chain.

## Scope

The config module owns **system-level configuration** only: settings bound to
the process environment.

In scope:

- `LoggingConfig`
- `ServerConfig`
- `HypervecConfig`

Out of scope:

- No INI parsing (CFG-02).
- No CLI override (CFG-02/CFG-06).
- No environment-variable layer, no runtime reload.
- No Python binding (CFG-04).
- No index type, distance metric or per-instance algorithm parameters.

### Why index type and metric are excluded

Index type and distance metric are **algorithm choices bound to an index
instance**, not process environment. They travel through the index construction
API, which is already the established pattern (`_make_index` reads `index_type`
and `metric_type` per request). Keeping a second, global default in the config
module would create two sources of truth, so the startup chain resolves missing
values from the engine's internal constants instead (HNSWFlat/L2).

Consequently `IndexDefaultsConfig`, the `IndexType` enum and the
`ParseIndexType`/`IndexTypeName`/`ParseMetricType`/`MetricTypeName` helpers are
deliberately **not** part of this module.

`server_mode` and `enable_http2` are likewise excluded: the legacy Python
`server` section has no such keys, and HTTP/gRPC/dual are selected by three
separate entry scripts (CFG-06).

## Model

### LoggingConfig

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enable_logging` | bool | `true` | Master switch for runtime logging. |
| `log_level` | string | `"info"` | Minimum level emitted globally. |
| `log_to_stderr` | bool | `true` | Write records to standard error. |
| `log_to_file` | bool | `false` | Write records to the file sink. |
| `log_file_path` | string | `"logs/hypervec.log"` | File sink target path. |

### ServerConfig

| Field | Type | Default | Meaning |
|---|---|---|---|
| `data_root` | string | `"./data"` | Root directory for collection data. |
| `host` | string | `"127.0.0.1"` | Server bind host. |
| `port` | int | `8080` | Server bind port. |

### HypervecConfig

A startup-time snapshot holding `logging` and `server`.

## Metadata table

`GetConfigOptions()` returns an ordered, immutable table of eight
`ConfigOption` entries:

- `section` / `key`: INI coordinates.
- `type`: `kBool`, `kInt` or `kString`.
- `default_value`: rendered from the matching `HypervecConfig` field, which is
  the single source of truth. The table never carries a hand-maintained copy.
- `description`: one-line help emitted into the sample config.
- `optional`: the key may be unset or left empty.

### `optional` semantics

`optional = true` marks a key whose absence is meaningful: the parser maps it to
"unset" rather than to an empty string. It does **not** weaken cross-field
validation. `logging.log_file_path` is the only optional key today; the rule
"required when `log_to_file = true`" is enforced by validation in CFG-02, not
by the metadata table.

## Sample config format contract

`RenderSampleConfig()` is generated from the metadata table and must satisfy:

- Sections appear in table order, one header `[section]` per section.
- Every key is preceded by a `# <description>` comment line.
- Exactly one blank line separates two sections, and no blank line precedes the
  first section or trails the document.
- All entries of one section are contiguous; the renderer emits a header only
  when the section changes, so splitting a section would silently produce a
  duplicate header.

The unit tests in `test/unit_tests/cpp/test_runtime_config.cpp` pin this
contract down.

## Default value basis (legacy Python to C++ model)

The model consolidates defaults from the three legacy entry points. Where they
disagree, the config layer keeps one value and the transport keeps its own
default until CFG-06 consumes the snapshot.

| key | C++ model | http_server | grpc_server | dual_server |
|---|---|---|---|---|
| `data_root` | `"./data"` | required | required | required |
| `host` | `"127.0.0.1"` | `127.0.0.1` | `[::]` | `0.0.0.0` |
| `port` | `8080` | `8080` | `50051` | HTTP `8080` / gRPC `50051` |
| `log_level` | `"info"` | `"info"` | not exposed | `"info"` |

### `data_root`

Legacy entry points made `--data-root` a required argument, so `"./data"` is a
**new model-level default**, not a value inherited from the legacy CLI. It is a
placeholder for the typed snapshot; startup still treats `data_root` as
required and validates it as non-empty.

Relative-path resolution is defined by the layer that consumes the value:

- INI file value: resolved against the directory containing the config file.
- CLI value: resolved against the current working directory.

Path resolution itself is implemented in CFG-02/CFG-06, not in this stage.

### Single `server.host` / `server.port`

The model carries one listener because that is the only listener that exists in
the config layer at this stage. `127.0.0.1:8080` matches the HTTP entry point,
which is the only transport with a `server` section in the legacy config.

gRPC and dual keep their own transport defaults at the adapter layer:

- With no config file, gRPC keeps `[::]:50051` and dual keeps `0.0.0.0:8080`
  plus gRPC `0.0.0.0:50051`.
- Once a config file supplies `server.host`/`server.port`, the adapters use the
  snapshot and explicit CLI values still override it.

If the gRPC listener needs to become separately configurable, that belongs to
CFG-06 and will be added as additional model fields, not as ad-hoc adapter
defaults.

### Logging keys

The five logging keys come from the 2026-08-11 requirement that the C++ core
own logging policy, and match the field set already carried by the logging
work (`enable_logging`, `log_level`, `log_to_stderr`, `log_to_file`,
`log_file_path`).

- `enable_logging = true` and `log_to_stderr = true` preserve the current
  always-on stderr behavior.
- `log_to_file = false` preserves the current absence of a file sink.
- `log_level = "info"` matches the HTTP and dual entry points; the gRPC entry
  point did not expose a level and hardcoded INFO, so the model default is
  consistent with it.
- `log_file_path` is the one place where the C++ model deliberately differs
  from the legacy Python metadata, whose default was `None`. The C++ model
  uses `"logs/hypervec.log"`, the value carried by the reference sample INI.
  Because the key is `optional` and only consumed when `log_to_file = true`,
  the difference is not observable while the file sink is off; CFG-02 defines
  the exact interaction (an explicit empty value means "unset").

This stage only stores the strings. Level validation against
`trace/debug/info/warn/error/fatal` lands with the C++ logger in LOG-01/LOG-03.

## ABI note

`GetConfigOptions()` returns `const std::vector<ConfigOption>&`. This is safe
for the current delivery shape: `hypervec` is a static library and the SWIG
bindings link it into their own shared module, so the container never crosses a
module boundary. Should the core be shipped as a shared library, the signature
must change to an out-parameter or `std::span<const ConfigOption>`.

## Verification

- Core C++ compile check.
- `test/unit_tests/cpp/test_runtime_config.cpp`: metadata/default consistency,
  value-type literals, metadata uniqueness, section contiguity and the sample
  format contract.
