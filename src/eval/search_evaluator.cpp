/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/search_evaluator.h>
#include <utils/log/assert.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

namespace hypervec {

namespace {

size_t CheckedCount(idx_t rows, idx_t columns, const char* context) {
  HYPERVEC_THROW_IF_NOT_FMT(rows > 0, "%s rows must be positive", context);
  HYPERVEC_THROW_IF_NOT_FMT(columns > 0, "%s columns must be positive",
                            context);
  const uint64_t unsigned_rows = static_cast<uint64_t>(rows);
  const uint64_t unsigned_columns = static_cast<uint64_t>(columns);
  HYPERVEC_THROW_IF_NOT_FMT(
      unsigned_rows <= (std::numeric_limits<size_t>::max)() / unsigned_columns,
      "%s size exceeds addressable memory", context);
  return static_cast<size_t>(unsigned_rows * unsigned_columns);
}

void ValidateLabelsView(const NeighborLabelsView& view, const char* context) {
  HYPERVEC_THROW_IF_NOT_FMT(view.labels != nullptr,
                            "%s labels must not be null", context);
  (void)CheckedCount(view.query_count, view.neighbors_per_query, context);
}

void ValidateGroundTruth(const NeighborLabelsView& ground_truth, idx_t k) {
  for (idx_t query = 0; query < ground_truth.query_count; ++query) {
    std::unordered_set<idx_t> unique;
    unique.reserve(static_cast<size_t>(k));
    const size_t offset =
        static_cast<size_t>(query) * ground_truth.neighbors_per_query;
    for (idx_t neighbor = 0; neighbor < k; ++neighbor) {
      const idx_t label = ground_truth.labels[offset + neighbor];
      HYPERVEC_THROW_IF_NOT_MSG(
          label >= 0, "ground-truth labels within k must be non-negative");
      HYPERVEC_THROW_IF_NOT_MSG(
          unique.insert(label).second,
          "ground-truth labels within one query must be unique");
    }
  }
}

}  // namespace

double ComputeRecallAtK(const NeighborLabelsView& results,
                        const NeighborLabelsView& ground_truth, idx_t k) {
  ValidateLabelsView(results, "result");
  ValidateLabelsView(ground_truth, "ground-truth");
  HYPERVEC_THROW_IF_NOT_MSG(k > 0, "recall k must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(results.query_count == ground_truth.query_count,
                            "result and ground-truth query counts must match");
  HYPERVEC_THROW_IF_NOT_MSG(results.neighbors_per_query >= k,
                            "result rows must contain at least k labels");
  HYPERVEC_THROW_IF_NOT_MSG(ground_truth.neighbors_per_query >= k,
                            "ground-truth rows must contain at least k labels");
  ValidateGroundTruth(ground_truth, k);

  size_t matches = 0;
  for (idx_t query = 0; query < results.query_count; ++query) {
    const size_t result_offset =
        static_cast<size_t>(query) * results.neighbors_per_query;
    const size_t ground_truth_offset =
        static_cast<size_t>(query) * ground_truth.neighbors_per_query;
    std::unordered_set<idx_t> expected;
    expected.reserve(static_cast<size_t>(k));
    for (idx_t neighbor = 0; neighbor < k; ++neighbor) {
      expected.insert(ground_truth.labels[ground_truth_offset + neighbor]);
    }

    std::unordered_set<idx_t> observed;
    observed.reserve(static_cast<size_t>(k));
    for (idx_t neighbor = 0; neighbor < k; ++neighbor) {
      const idx_t label = results.labels[result_offset + neighbor];
      if (label >= 0 && observed.insert(label).second &&
          expected.contains(label)) {
        ++matches;
      }
    }
  }

  const size_t denominator = CheckedCount(results.query_count, k, "recall");
  return static_cast<double>(matches) / static_cast<double>(denominator);
}

SearchEvaluationResult EvaluateSearch(const Index& index,
                                      const SearchEvaluationInput& input,
                                      const SearchEvaluationOptions& options,
                                      const SearchParameters* parameters) {
  HYPERVEC_THROW_IF_NOT_MSG(input.queries != nullptr,
                            "evaluation queries must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(input.query_count > 0,
                            "evaluation query count must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(options.k > 0, "evaluation k must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(options.measured_runs > 0,
                            "evaluation measured_runs must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      input.ground_truth.query_count == input.query_count,
      "evaluation and ground-truth query counts must match");
  HYPERVEC_THROW_IF_NOT_MSG(
      input.ground_truth.neighbors_per_query >= options.k,
      "ground-truth rows must contain at least evaluation k labels");
  ValidateLabelsView(input.ground_truth, "ground-truth");
  ValidateGroundTruth(input.ground_truth, options.k);

  const size_t output_count =
      CheckedCount(input.query_count, options.k, "evaluation output");
  HYPERVEC_THROW_IF_NOT_MSG(
      options.measured_runs <= (std::numeric_limits<size_t>::max)() /
                                   static_cast<size_t>(input.query_count),
      "evaluation measured query count exceeds size_t");
  std::vector<float> distances(output_count);
  std::vector<idx_t> labels(output_count);

  for (size_t run = 0; run < options.warmup_runs; ++run) {
    index.Search(input.query_count, input.queries, options.k, distances.data(),
                 labels.data(), parameters);
  }

  const auto start = std::chrono::steady_clock::now();
  for (size_t run = 0; run < options.measured_runs; ++run) {
    index.Search(input.query_count, input.queries, options.k, distances.data(),
                 labels.data(), parameters);
  }
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_seconds =
      std::chrono::duration<double>(end - start).count();
  const size_t measured_queries =
      options.measured_runs * static_cast<size_t>(input.query_count);

  const NeighborLabelsView results{labels.data(), input.query_count, options.k};
  SearchEvaluationResult evaluation;
  evaluation.query_count = input.query_count;
  evaluation.k = options.k;
  evaluation.measured_runs = options.measured_runs;
  evaluation.recall_at_k =
      ComputeRecallAtK(results, input.ground_truth, options.k);
  evaluation.elapsed_seconds = elapsed_seconds;
  evaluation.mean_latency_ms =
      elapsed_seconds * 1000.0 / static_cast<double>(measured_queries);
  evaluation.queries_per_second =
      elapsed_seconds > 0.0
          ? static_cast<double>(measured_queries) / elapsed_seconds
          : (std::numeric_limits<double>::infinity)();
  return evaluation;
}

}  // namespace hypervec
