/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/search_evaluator.h>
#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <utils/log/exception.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

class ScriptedIndex final : public hypervec::Index {
 public:
  ScriptedIndex() : Index(2, hypervec::kMetricL2) {}

  mutable size_t search_count = 0;
  mutable const hypervec::SearchParameters* received_parameters = nullptr;

  void Add(hypervec::idx_t n, const float*) final { n_total += n; }

  void Search(hypervec::idx_t n, const float* queries, hypervec::idx_t k,
              float* distances, hypervec::idx_t* labels,
              const hypervec::SearchParameters* parameters) const final {
    ++search_count;
    received_parameters = parameters;
    ASSERT_EQ(k, 2);
    for (hypervec::idx_t query = 0; query < n; ++query) {
      const bool first_query = queries[query * d] == 0.0F;
      distances[query * k] = 0.0F;
      distances[query * k + 1] = 1.0F;
      labels[query * k] = first_query ? 2 : 4;
      labels[query * k + 1] = first_query ? 1 : 4;
    }
  }

  void Reset() final { n_total = 0; }
};

class ConcurrentIndex final : public hypervec::Index {
 public:
  ConcurrentIndex() : Index(2, hypervec::kMetricL2) {}

  mutable std::atomic<size_t> calls{0};
  mutable std::mutex mutex;
  mutable std::set<std::thread::id> callers;
  bool fail = false;

  void Add(hypervec::idx_t n, const float*) final { n_total += n; }
  void Reset() final { n_total = 0; }

  void Search(hypervec::idx_t n, const float* queries, hypervec::idx_t k,
              float* distances, hypervec::idx_t* labels,
              const hypervec::SearchParameters*) const final {
    ++calls;
    {
      std::lock_guard<std::mutex> guard(mutex);
      callers.insert(std::this_thread::get_id());
    }
    if (fail && queries[0] == 1.0F) {
      throw std::runtime_error("concurrent search failure");
    }
    for (hypervec::idx_t query = 0; query < n; ++query) {
      distances[query * k] = 0.0F;
      distances[query * k + 1] = 1.0F;
      labels[query * k] = queries[query * d] == 0.0F ? 2 : 4;
      labels[query * k + 1] = queries[query * d] == 0.0F ? 1 : 4;
    }
  }
};

}  // namespace

TEST(SearchEvaluator, ComputesSetBasedRecallWithoutDuplicateCredit) {
  const std::array<hypervec::idx_t, 6> results = {2, 1, -1, 3, 3, 8};
  const std::array<hypervec::idx_t, 8> ground_truth = {1, 2, 7, 9, 3, 4, 5, 6};

  const double recall = hypervec::ComputeRecallAtK(
      {results.data(), 2, 3}, {ground_truth.data(), 2, 4}, 3);

  EXPECT_DOUBLE_EQ(recall, 0.5);
}

TEST(SearchEvaluator, MeasuresRepeatedSearchAndForwardsParameters) {
  ScriptedIndex index;
  const std::array<float, 4> queries = {0.0F, 0.0F, 1.0F, 1.0F};
  const std::array<hypervec::idx_t, 4> ground_truth = {1, 2, 3, 4};
  hypervec::SearchParameters parameters;
  hypervec::SearchEvaluationInput input{
      queries.data(), 2, {ground_truth.data(), 2, 2}};
  hypervec::SearchEvaluationOptions options;
  options.k = 2;
  options.warmup_runs = 2;
  options.measured_runs = 3;

  const auto result =
      hypervec::EvaluateSearch(index, input, options, &parameters);

  EXPECT_EQ(index.search_count, 5U);
  EXPECT_EQ(index.received_parameters, &parameters);
  EXPECT_EQ(result.query_count, 2);
  EXPECT_EQ(result.k, 2);
  EXPECT_EQ(result.measured_runs, 3U);
  EXPECT_DOUBLE_EQ(result.recall_at_k, 0.75);
  EXPECT_GE(result.elapsed_seconds, 0.0);
  EXPECT_GE(result.mean_latency_ms, 0.0);
  EXPECT_TRUE(result.queries_per_second > 0.0 ||
              std::isinf(result.queries_per_second));
  EXPECT_EQ(result.query_batch_size, 2);
  EXPECT_EQ(result.latency_sample_count, 3U);
  EXPECT_LE(result.batch_latency_p50_ms, result.batch_latency_p95_ms);
  EXPECT_LE(result.batch_latency_p95_ms, result.batch_latency_p99_ms);
}

TEST(SearchEvaluator, BatchesQueriesAndReportsOneSamplePerSearchCall) {
  ScriptedIndex index;
  const std::array<float, 4> queries = {0.0F, 0.0F, 1.0F, 1.0F};
  const std::array<hypervec::idx_t, 4> ground_truth = {1, 2, 3, 4};
  const hypervec::SearchEvaluationInput input{
      queries.data(), 2, {ground_truth.data(), 2, 2}};
  hypervec::SearchEvaluationOptions options;
  options.k = 2;
  options.warmup_runs = 2;
  options.measured_runs = 3;
  options.query_batch_size = 1;

  const auto result = hypervec::EvaluateSearch(index, input, options);

  EXPECT_EQ(index.search_count, 10U);
  EXPECT_EQ(result.query_batch_size, 1);
  EXPECT_EQ(result.latency_sample_count, 6U);
  EXPECT_DOUBLE_EQ(result.recall_at_k, 0.75);
  EXPECT_LE(result.batch_latency_p50_ms, result.batch_latency_p95_ms);
  EXPECT_LE(result.batch_latency_p95_ms, result.batch_latency_p99_ms);
}

TEST(SearchEvaluator, ConcurrentCallersShareOnlyReadOnlyIndex) {
  ConcurrentIndex index;
  const std::array<float, 8> queries = {0, 0, 1, 1, 0, 0, 1, 1};
  const std::array<hypervec::idx_t, 8> truth = {1, 2, 3, 4, 1, 2, 3, 4};
  const hypervec::SearchEvaluationInput input{
      queries.data(), 4, {truth.data(), 4, 2}};
  hypervec::SearchEvaluationOptions options;
  options.k = 2;
  options.warmup_runs = 1;
  options.measured_runs = 2;
  options.concurrency = 4;
  options.index_is_read_only = true;

  const auto result = hypervec::EvaluateSearch(index, input, options);
  EXPECT_EQ(result.concurrency, 4U);
  EXPECT_EQ(result.query_batch_size, 1);
  EXPECT_EQ(result.latency_sample_count, 8U);
  EXPECT_DOUBLE_EQ(result.recall_at_k, 0.75);
  EXPECT_EQ(index.calls.load(), 12U);
  EXPECT_EQ(index.callers.size(), 4U);

  options.query_batch_size = 4;
  const auto single_batch = hypervec::EvaluateSearch(index, input, options);
  EXPECT_EQ(single_batch.concurrency, 1U);
  EXPECT_EQ(single_batch.query_batch_size, 4);

  index.fail = true;
  options.warmup_runs = 0;
  options.query_batch_size = 1;
  EXPECT_THROW(hypervec::EvaluateSearch(index, input, options),
               std::runtime_error);
}

TEST(SearchEvaluator, ConcurrentHNSWSearchMatchesSequentialSearch) {
  hypervec::IndexHNSWFlat index(2, 4);
  const std::vector<float> database = {0, 0, 1, 0, 2, 0, 3, 0,
                                       4, 0, 5, 0, 6, 0, 7, 0};
  const std::vector<float> queries = database;
  index.Add(8, database.data());
  std::vector<float> distances(8 * 2);
  std::vector<hypervec::idx_t> truth(8 * 2);
  index.Search(8, queries.data(), 2, distances.data(), truth.data());
  const hypervec::SearchEvaluationInput input{
      queries.data(), 8, {truth.data(), 8, 2}};
  hypervec::SearchEvaluationOptions options;
  options.k = 2;
  options.warmup_runs = 1;
  options.measured_runs = 2;
  options.concurrency = 4;
  options.index_is_read_only = true;
  const auto result = hypervec::EvaluateSearch(index, input, options);
  EXPECT_EQ(result.concurrency, 4U);
  EXPECT_EQ(result.latency_sample_count, 16U);
  EXPECT_DOUBLE_EQ(result.recall_at_k, 1.0);
}

TEST(SearchEvaluator, ReportsExactFlatRecall) {
  hypervec::IndexFlatL2 index(2);
  const std::vector<float> database = {0.0F, 0.0F, 1.0F, 0.0F,
                                       2.0F, 0.0F, 3.0F, 0.0F};
  const std::vector<float> queries = {0.1F, 0.0F, 2.9F, 0.0F};
  const std::array<hypervec::idx_t, 4> ground_truth = {0, 1, 3, 2};
  index.Add(4, database.data());

  const hypervec::SearchEvaluationInput input{
      queries.data(), 2, {ground_truth.data(), 2, 2}};
  hypervec::SearchEvaluationOptions options;
  options.k = 2;
  const auto result = hypervec::EvaluateSearch(index, input, options);

  EXPECT_DOUBLE_EQ(result.recall_at_k, 1.0);
  EXPECT_EQ(result.query_count, 2);
}

TEST(SearchEvaluator, RejectsMalformedInputsBeforeSearching) {
  ScriptedIndex index;
  const std::array<float, 4> queries = {0.0F, 0.0F, 1.0F, 1.0F};
  const std::array<hypervec::idx_t, 4> valid_ground_truth = {1, 2, 3, 4};
  const std::array<hypervec::idx_t, 4> duplicate_ground_truth = {1, 1, 3, 4};
  hypervec::SearchEvaluationOptions options;
  options.k = 2;

  hypervec::SearchEvaluationInput input{
      queries.data(), 2, {valid_ground_truth.data(), 2, 2}};
  options.measured_runs = 0;
  EXPECT_THROW(hypervec::EvaluateSearch(index, input, options),
               hypervec::HypervecException);
  EXPECT_EQ(index.search_count, 0U);

  options.measured_runs = 1;
  input.ground_truth.labels = duplicate_ground_truth.data();
  EXPECT_THROW(hypervec::EvaluateSearch(index, input, options),
               hypervec::HypervecException);
  EXPECT_EQ(index.search_count, 0U);

  input.ground_truth.labels = valid_ground_truth.data();
  input.ground_truth.neighbors_per_query = 1;
  EXPECT_THROW(hypervec::EvaluateSearch(index, input, options),
               hypervec::HypervecException);
  EXPECT_EQ(index.search_count, 0U);

  input.ground_truth.neighbors_per_query = 2;
  options.query_batch_size = -1;
  EXPECT_THROW(hypervec::EvaluateSearch(index, input, options),
               hypervec::HypervecException);
  EXPECT_EQ(index.search_count, 0U);

  options.query_batch_size = 0;
  options.concurrency = 0;
  EXPECT_THROW(hypervec::EvaluateSearch(index, input, options),
               hypervec::HypervecException);
  EXPECT_EQ(index.search_count, 0U);
}

TEST(SearchEvaluator, RejectsInvalidRecallViews) {
  const std::array<hypervec::idx_t, 2> labels = {1, 2};
  const std::array<hypervec::idx_t, 2> negative_ground_truth = {1, -1};

  EXPECT_THROW(
      hypervec::ComputeRecallAtK({labels.data(), 1, 2},
                                 {negative_ground_truth.data(), 1, 2}, 2),
      hypervec::HypervecException);
  EXPECT_THROW(hypervec::ComputeRecallAtK({labels.data(), 1, 1},
                                          {labels.data(), 1, 2}, 2),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::ComputeRecallAtK({labels.data(), 1, 2},
                                          {labels.data(), 2, 1}, 1),
               hypervec::HypervecException);
}
