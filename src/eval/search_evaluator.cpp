/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/search_evaluator.h>
#include <omp.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
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

void SearchBatches(const Index& index, const SearchEvaluationInput& input,
                   idx_t k, idx_t batch_size, float* distances, idx_t* labels,
                   const SearchParameters* parameters,
                   std::vector<double>* latency_samples_ms) {
  for (idx_t first = 0; first < input.query_count;) {
    const idx_t count = std::min(batch_size, input.query_count - first);
    const size_t query_offset =
        static_cast<size_t>(first) * static_cast<size_t>(index.d);
    const size_t result_offset =
        static_cast<size_t>(first) * static_cast<size_t>(k);
    const auto start = std::chrono::steady_clock::now();
    index.Search(count, input.queries + query_offset, k,
                 distances + result_offset, labels + result_offset, parameters);
    if (latency_samples_ms != nullptr) {
      const auto end = std::chrono::steady_clock::now();
      latency_samples_ms->push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }
    first += count;
  }
}

// Each worker owns disjoint result rows for every run; no worker writes to
// another's rows, and the index is read-only throughout evaluation.
double SearchConcurrent(const Index& index, const SearchEvaluationInput& input,
                        idx_t k, idx_t batch_size, size_t runs, size_t workers,
                        float* distances, idx_t* labels,
                        const SearchParameters* parameters,
                        std::vector<double>* latency_samples_ms) {
  const size_t batches =
      1U + static_cast<size_t>((input.query_count - 1) / batch_size);
  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> failed{false};
  std::exception_ptr error;
  std::mutex error_mutex;
  std::vector<std::jthread> threads;
  threads.reserve(workers);
  try {
    for (size_t worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&, worker] {
        // Avoid multiplying worker count by an index's own OpenMP query team.
        omp_set_num_threads(1);
        ready.fetch_add(1);
        ready.notify_one();
        start.wait(false);
        try {
          for (size_t run = 0; run < runs && !failed.load(); ++run) {
            for (size_t batch = worker; batch < batches && !failed.load();
                 batch += workers) {
              const idx_t first = static_cast<idx_t>(batch * batch_size);
              const idx_t count =
                  std::min(batch_size, input.query_count - first);
              const size_t query_offset =
                  static_cast<size_t>(first) * static_cast<size_t>(index.d);
              const size_t result_offset =
                  static_cast<size_t>(first) * static_cast<size_t>(k);
              const auto begin = std::chrono::steady_clock::now();
              index.Search(count, input.queries + query_offset, k,
                           distances + result_offset, labels + result_offset,
                           parameters);
              if (latency_samples_ms != nullptr) {
                const auto end = std::chrono::steady_clock::now();
                (*latency_samples_ms)[run * batches + batch] =
                    std::chrono::duration<double, std::milli>(end - begin)
                        .count();
              }
            }
          }
        } catch (...) {
          std::lock_guard<std::mutex> guard(error_mutex);
          if (error == nullptr) {
            error = std::current_exception();
          }
          failed.store(true);
        }
      });
    }
  } catch (...) {
    start.store(true);
    start.notify_all();
    throw;
  }
  for (;;) {
    const size_t observed = ready.load();
    if (observed == workers) {
      break;
    }
    ready.wait(observed);
  }
  const auto begin = std::chrono::steady_clock::now();
  start.store(true);
  start.notify_all();
  threads.clear();  // jthread joins before we read results or exceptions.
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  if (error != nullptr) {
    std::rethrow_exception(error);
  }
  return elapsed;
}

double NearestRankPercentile(const std::vector<double>& sorted_values,
                             double percentile) {
  const long double rank =
      std::ceil(static_cast<long double>(percentile) * sorted_values.size());
  const size_t offset = std::max<size_t>(1, static_cast<size_t>(rank)) - 1U;
  return sorted_values[offset];
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
  HYPERVEC_THROW_IF_NOT_MSG(options.query_batch_size >= 0,
                            "evaluation query_batch_size must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(options.concurrency > 0,
                            "evaluation concurrency must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      input.ground_truth.query_count == input.query_count,
      "evaluation and ground-truth query counts must match");
  HYPERVEC_THROW_IF_NOT_MSG(
      input.ground_truth.neighbors_per_query >= options.k,
      "ground-truth rows must contain at least evaluation k labels");
  ValidateLabelsView(input.ground_truth, "ground-truth");
  ValidateGroundTruth(input.ground_truth, options.k);
  (void)CheckedCount(input.query_count, index.d, "evaluation query");

  const size_t output_count =
      CheckedCount(input.query_count, options.k, "evaluation output");
  HYPERVEC_THROW_IF_NOT_MSG(
      options.measured_runs <= (std::numeric_limits<size_t>::max)() /
                                   static_cast<size_t>(input.query_count),
      "evaluation measured query count exceeds size_t");
  const idx_t batch_size =
      options.query_batch_size == 0
          ? (options.concurrency == 1 ? input.query_count : 1)
          : std::min(options.query_batch_size, input.query_count);
  const size_t batches_per_run =
      1U + static_cast<size_t>((input.query_count - 1) / batch_size);
  HYPERVEC_THROW_IF_NOT_MSG(
      options.measured_runs <=
          (std::numeric_limits<size_t>::max)() / batches_per_run,
      "evaluation latency sample count exceeds size_t");
  std::vector<float> distances(output_count);
  std::vector<idx_t> labels(output_count);
  const size_t workers = std::min(options.concurrency, batches_per_run);

  std::vector<double> latency_samples_ms;
  double elapsed_seconds = 0.0;
  if (workers == 1) {
    for (size_t run = 0; run < options.warmup_runs; ++run) {
      SearchBatches(index, input, options.k, batch_size, distances.data(),
                    labels.data(), parameters, nullptr);
    }
    latency_samples_ms.reserve(options.measured_runs * batches_per_run);
    const auto start = std::chrono::steady_clock::now();
    for (size_t run = 0; run < options.measured_runs; ++run) {
      SearchBatches(index, input, options.k, batch_size, distances.data(),
                    labels.data(), parameters, &latency_samples_ms);
    }
    elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
  } else {
    if (options.warmup_runs > 0) {
      SearchConcurrent(index, input, options.k, batch_size, options.warmup_runs,
                       workers, distances.data(), labels.data(), parameters,
                       nullptr);
    }
    latency_samples_ms.resize(options.measured_runs * batches_per_run);
    elapsed_seconds = SearchConcurrent(
        index, input, options.k, batch_size, options.measured_runs, workers,
        distances.data(), labels.data(), parameters, &latency_samples_ms);
  }
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
  std::sort(latency_samples_ms.begin(), latency_samples_ms.end());
  evaluation.query_batch_size = batch_size;
  evaluation.concurrency = workers;
  evaluation.latency_sample_count = latency_samples_ms.size();
  evaluation.batch_latency_p50_ms =
      NearestRankPercentile(latency_samples_ms, 0.50);
  evaluation.batch_latency_p95_ms =
      NearestRankPercentile(latency_samples_ms, 0.95);
  evaluation.batch_latency_p99_ms =
      NearestRankPercentile(latency_samples_ms, 0.99);
  return evaluation;
}

}  // namespace hypervec
