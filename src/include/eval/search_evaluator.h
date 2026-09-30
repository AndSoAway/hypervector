/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstddef>

namespace hypervec {

/** Non-owning matrix of neighbor labels in row-major query order. */
struct NeighborLabelsView {
  const idx_t* labels = nullptr;
  idx_t query_count = 0;
  idx_t neighbors_per_query = 0;
};

/** Compute set-based recall@k while ignoring duplicate result labels.
 *
 * Ground-truth rows must contain at least k unique, non-negative labels.
 * Negative result labels are treated as missing results.
 */
double ComputeRecallAtK(const NeighborLabelsView& results,
                        const NeighborLabelsView& ground_truth, idx_t k);

/** Non-owning inputs for one search evaluation. */
struct SearchEvaluationInput {
  const float* queries = nullptr;
  idx_t query_count = 0;
  NeighborLabelsView ground_truth;
};

/** Controls warmup and repeated measured searches over the complete batch. */
struct SearchEvaluationOptions {
  idx_t k = 10;
  size_t warmup_runs = 0;
  size_t measured_runs = 1;
  /** Queries per timed Index::Search call; zero uses the complete workload. */
  idx_t query_batch_size = 0;
  /** Independent concurrent callers. With concurrency > 1 and no explicit
   * batch size, each call searches one query. Requires a read-only index. */
  size_t concurrency = 1;
  /** Asserts the caller has verified the index performs no mutable work during
   * Search. Concurrency above one shares one Index across threads, and the
   * persistence layer exposes no read-only load mode, so the guarantee cannot
   * be derived from the Index itself and must be asserted here. EvaluateSearch
   * rejects concurrency above one while this is false. */
  bool index_is_read_only = false;
};

/** Aggregate batch-search quality and timing metrics.
 * With concurrency > 1, mean_latency_ms is wall time divided by the total
 * query count (inverse throughput), NOT mean individual query latency. For
 * per-query latency percentiles, use query_batch_size=1.
 */
struct SearchEvaluationResult {
  idx_t query_count = 0;
  idx_t k = 0;
  size_t measured_runs = 0;
  double recall_at_k = 0.0;
  double elapsed_seconds = 0.0;
  double mean_latency_ms = 0.0;
  double queries_per_second = 0.0;
  idx_t query_batch_size = 0;
  size_t concurrency = 1;
  size_t latency_sample_count = 0;
  double batch_latency_p50_ms = 0.0;
  double batch_latency_p95_ms = 0.0;
  double batch_latency_p99_ms = 0.0;
};

/** Evaluate Index::Search against caller-provided ground truth.
 *
 * Warmup runs are not timed. Recall is computed from the final measured run;
 * latency and QPS cover every measured run. Batch latency uses nearest-rank
 * percentiles over individual Index::Search calls. Concurrent evaluation
 * shares a read-only Index and SearchParameters across independent callers;
 * callers must not mutate either while evaluation runs. It does not promise
 * thread safety for arbitrary user-provided Index implementations. Each
 * concurrent worker limits nested OpenMP search parallelism to one thread.
 * Search exceptions propagate after all workers finish.
 */
SearchEvaluationResult EvaluateSearch(
    const Index& index, const SearchEvaluationInput& input,
    const SearchEvaluationOptions& options = {},
    const SearchParameters* parameters = nullptr);

}  // namespace hypervec
