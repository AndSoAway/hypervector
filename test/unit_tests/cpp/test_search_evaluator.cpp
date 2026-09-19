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
#include <utils/log/exception.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

class ScriptedIndex final : public hypervec::Index {
 public:
  ScriptedIndex() : Index(2, hypervec::kMetricL2) {}

  mutable size_t search_count = 0;
  mutable const hypervec::SearchParameters* received_parameters = nullptr;

  void Add(hypervec::idx_t n, const float*) final { n_total += n; }

  void Search(hypervec::idx_t n, const float*, hypervec::idx_t k,
              float* distances, hypervec::idx_t* labels,
              const hypervec::SearchParameters* parameters) const final {
    ++search_count;
    received_parameters = parameters;
    ASSERT_EQ(n, 2);
    ASSERT_EQ(k, 2);
    const std::array<hypervec::idx_t, 4> scripted_labels = {2, 1, 4, 4};
    for (size_t i = 0; i < scripted_labels.size(); ++i) {
      distances[i] = static_cast<float>(i);
      labels[i] = scripted_labels[i];
    }
  }

  void Reset() final { n_total = 0; }
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
