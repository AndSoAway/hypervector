# HyperVec performance evaluation tool

`hypervec_eval` measures HyperVec index build and search performance with a
dense float32 HDF5 dataset. It is adapted from the evaluation concepts used by
VSAG while using HyperVec's public `IndexConfig`, `Engine`, registry,
`Index::Build`, and `Index::Search` interfaces. No HyperVec index
implementation is changed by this tool.

## Dataset contract

The input HDF5 file must contain four rank-two datasets:

| Dataset | Type | Shape |
| --- | --- | --- |
| `train` | float32 | `(base_count, dimension)` |
| `test` | float32 | `(query_count, dimension)` |
| `neighbors` | int64-compatible integer | `(query_count, ground_truth_k)` |
| `distances` | float32 | `(query_count, ground_truth_k)` |

## Build

```bash
cmake -S . -B build \
  -DHYPERVEC_BUILD_EVAL=ON \
  -DHYPERVEC_FETCH_DEPS=ON \
  -DBUILD_TESTING=ON
cmake --build build --target hypervec_eval hypervec_eval_test -j
ctest --test-dir build -R hypervec_eval_test --output-on-failure
```

The Eval target follows the project-wide dependency policy. Missing yaml-cpp,
nlohmann-json, tabulate, and HDF5 dependencies are fetched only when
`HYPERVEC_FETCH_DEPS=ON`; otherwise they must already be installed.

## Run

CLI mode:

```bash
./build/tools/eval/hypervec_eval \
  --index-config tools/eval/configs/hnsw_flat.json \
  --search-config tools/eval/configs/hnsw_search.json \
  --dataset /path/to/sift-128-euclidean.hdf5 \
  --topk 10 --num_threads 4
```

YAML mode:

```bash
./build/tools/eval/hypervec_eval --config tools/eval/eval_template.yaml
```

Index construction and algorithm parameters live in the same strict JSON
configuration accepted by HyperVec's public factory. Per-search overrides use
the public search JSON format. Use `--help` for the full option list and
registered index types.

## Metrics

Build output contains `duration(s)`, `tps`, and `index_memory(B)`. Search output
contains QPS, average latency, latency p50/p80/p90/p95/p99, average recall, and
recall p0/p10/p30/p50/p70/p90. Distance recall uses the ground-truth distance
at `top_k - 1` as the threshold with a `2e-6` tolerance. `--id_recall` selects
ID-intersection recall instead.

The public factory currently has no generic finalize/freeze operation or
public build/search statistics interface. Eval therefore does not downcast
concrete index types to call private lifecycle methods and reports only the
tool-level timing, throughput, memory, latency, and recall metrics above.
