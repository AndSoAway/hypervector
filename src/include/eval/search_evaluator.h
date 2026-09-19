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
};

/** Aggregate batch-search quality and timing metrics. */
struct SearchEvaluationResult {
  idx_t query_count = 0;
  idx_t k = 0;
  size_t measured_runs = 0;
  double recall_at_k = 0.0;
  double elapsed_seconds = 0.0;
  double mean_latency_ms = 0.0;
  double queries_per_second = 0.0;
};

/** Evaluate Index::Search against caller-provided ground truth.
 *
 * Warmup runs are not timed. Recall is computed from the final measured run;
 * latency and QPS cover every measured run. Search exceptions propagate.
 */
SearchEvaluationResult EvaluateSearch(
    const Index& index, const SearchEvaluationInput& input,
    const SearchEvaluationOptions& options = {},
    const SearchParameters* parameters = nullptr);

}  // namespace hypervec
