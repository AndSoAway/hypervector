# HyperVec performance evaluation tool

`hypervec_eval` measures HyperVec index build and search performance on dense
float32 vectors stored in the standard ANN benchmark `.fvecs` / `.ivecs`
format. It is adapted from the evaluation concepts used by VSAG while using
HyperVec's public `IndexConfig`, `Engine`, registry, `Index::Build`, and
`Index::Search` interfaces. No HyperVec index implementation is changed by this
tool.

## Dataset contract

Three files are required:

| File | Format | Shape |
| --- | --- | --- |
| base vectors | `.fvecs` | `base_count` x `dimension` float32 |
| query vectors | `.fvecs` | `query_count` x `dimension` float32 |
| ground truth | `.ivecs` | `query_count` x `ground_truth_k` int32 neighbor ids |

Each `.fvecs` record is `[int32 dimension][dimension float32]`; each `.ivecs`
record is `[int32 k][k int32]`. Ground truth provides neighbor ids only, so
recall is computed by id intersection.

## Build

```bash
cmake -S . -B build \
  -DHYPERVEC_BUILD_EVAL=ON \
  -DHYPERVEC_FETCH_DEPS=ON \
  -DBUILD_TESTING=ON
cmake --build build --target hypervec_eval hypervec_eval_test -j
ctest --test-dir build -R hypervec_eval_test --output-on-failure
```

The Eval target follows the project-wide dependency policy. Missing
nlohmann-json and tabulate dependencies are fetched only when
`HYPERVEC_FETCH_DEPS=ON`; otherwise they must already be installed.

## Run

```bash
./build/tools/eval/hypervec_eval \
  --base /path/to/base.fvecs \
  --query /path/to/query.fvecs \
  --groundtruth /path/to/groundtruth.ivecs \
  --index-config tools/eval/configs/ivf_flat.json \
  --search-config tools/eval/configs/ivf_search.json \
  --topk 10 --num_threads 4
```

Index construction and algorithm parameters live in the same strict JSON
configuration accepted by HyperVec's public factory. Per-search overrides use
the public search JSON format. Use `--help` for the full option list and
registered index types.

## Metrics

Build output contains `duration(s)`, `tps`, and `index_memory(B)`. Search output
contains QPS, average latency, latency p50/p80/p90/p95/p99, average recall, and
recall p0/p10/p30/p50/p70/p90. Recall is computed by id intersection against
the `.ivecs` ground-truth neighbors.

The public factory currently has no generic finalize/freeze operation or
public build/search statistics interface. Eval therefore does not downcast
concrete index types to call private lifecycle methods and reports only the
tool-level timing, throughput, memory, latency, and recall metrics above.
