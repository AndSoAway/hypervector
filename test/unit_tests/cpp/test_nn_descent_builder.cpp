/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_validation.h>
#include <index/graph/nn_descent_builder.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

class ScalarDistanceComputer final : public hypervec::DistanceComputer {
 public:
  explicit ScalarDistanceComputer(std::vector<float> values)
      : values_(std::move(values)) {}

  void SetQuery(const float* query) override { query_ = *query; }

  float operator()(hypervec::idx_t index) override {
    const float difference = values_.at(static_cast<size_t>(index)) - query_;
    return difference * difference;
  }

  float symmetric_dis(hypervec::idx_t lhs, hypervec::idx_t rhs) override {
    const float difference = values_.at(static_cast<size_t>(lhs)) -
                             values_.at(static_cast<size_t>(rhs));
    return difference * difference;
  }

 private:
  std::vector<float> values_;
  float query_ = 0.0F;
};

class NaNDistanceComputer final : public hypervec::DistanceComputer {
 public:
  void SetQuery(const float*) override {}
  float operator()(hypervec::idx_t) override {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float symmetric_dis(hypervec::idx_t, hypervec::idx_t) override {
    return std::numeric_limits<float>::quiet_NaN();
  }
};

std::vector<hypervec::GraphId> CopyNeighbors(
    const hypervec::GraphStorage& graph, hypervec::GraphId node) {
  const auto neighbors = graph.Neighbors(node);
  return {neighbors.begin(), neighbors.end()};
}

std::vector<hypervec::GraphId> ExactNeighbors(const std::vector<float>& values,
                                              size_t node, size_t degree) {
  std::vector<std::pair<float, hypervec::GraphId>> candidates;
  candidates.reserve(values.size() - 1);
  for (size_t other = 0; other < values.size(); ++other) {
    if (other == node) {
      continue;
    }
    const float difference = values[node] - values[other];
    candidates.emplace_back(difference * difference,
                            static_cast<hypervec::GraphId>(other));
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.resize(std::min(degree, candidates.size()));
  std::vector<hypervec::GraphId> result;
  result.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    result.push_back(candidate.second);
  }
  return result;
}

double Recall(const hypervec::GraphStorage& graph,
              const std::vector<float>& values, size_t degree) {
  size_t matches = 0;
  size_t expected_count = 0;
  for (size_t node = 0; node < values.size(); ++node) {
    const std::vector<hypervec::GraphId> expected =
        ExactNeighbors(values, node, degree);
    const std::vector<hypervec::GraphId> actual =
        CopyNeighbors(graph, static_cast<hypervec::GraphId>(node));
    expected_count += expected.size();
    for (hypervec::GraphId neighbor : actual) {
      matches += static_cast<size_t>(std::find(expected.begin(), expected.end(),
                                               neighbor) != expected.end());
    }
  }
  return expected_count == 0 ? 1.0
                             : static_cast<double>(matches) /
                                   static_cast<double>(expected_count);
}

}  // namespace

TEST(NNDescentBuilder, BuildsExactGraphWhenDegreeCoversDataset) {
  const std::vector<float> values = {0.0F, 10.0F, 3.0F, 4.0F, 20.0F};
  ScalarDistanceComputer distance(values);
  const hypervec::NNDescentBuilder builder(
      hypervec::NNDescentOptions{4, 4, 0.0, 7});
  hypervec::NNDescentStats stats;

  const hypervec::MutableBoundedGraph graph =
      builder.Build(distance, values.size(), &stats);

  EXPECT_EQ(graph.NodeCount(), values.size());
  EXPECT_EQ(graph.MaxDegree(), 4U);
  for (size_t node = 0; node < values.size(); ++node) {
    EXPECT_EQ(CopyNeighbors(graph, static_cast<hypervec::GraphId>(node)),
              ExactNeighbors(values, node, 4));
  }
  const auto report = hypervec::ValidateGraph(graph);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(stats.initial_distance_computations, 20U);
  EXPECT_EQ(stats.iterations, 1U);
  EXPECT_EQ(stats.neighbor_updates, 0U);
  EXPECT_TRUE(stats.converged);
}

TEST(NNDescentBuilder, RefinesRandomGraphWithDeterministicRecall) {
  constexpr size_t kCount = 128;
  constexpr size_t kDegree = 10;
  std::vector<float> values(kCount);
  for (size_t node = 0; node < kCount; ++node) {
    values[node] = static_cast<float>((node * 37) % kCount);
  }
  ScalarDistanceComputer distance(values);
  const hypervec::NNDescentBuilder builder(
      hypervec::NNDescentOptions{kDegree, 15, 0.001, 42});
  hypervec::NNDescentStats first_stats;
  const hypervec::MutableBoundedGraph first =
      builder.Build(distance, values.size(), &first_stats);

  ScalarDistanceComputer repeated_distance(values);
  hypervec::NNDescentStats repeated_stats;
  const hypervec::MutableBoundedGraph repeated =
      builder.Build(repeated_distance, values.size(), &repeated_stats);

  EXPECT_TRUE(hypervec::ValidateGraph(first).IsStructurallyValid());
  EXPECT_GE(Recall(first, values, kDegree), 0.90);
  EXPECT_GT(first_stats.neighbor_updates, 0U);
  EXPECT_GT(first_stats.refinement_distance_computations, 0U);
  EXPECT_LE(first_stats.iterations, 15U);
  EXPECT_TRUE(first_stats.converged);
  EXPECT_EQ(first_stats.iterations, repeated_stats.iterations);
  EXPECT_EQ(first_stats.neighbor_updates, repeated_stats.neighbor_updates);
  for (size_t node = 0; node < values.size(); ++node) {
    EXPECT_EQ(CopyNeighbors(first, static_cast<hypervec::GraphId>(node)),
              CopyNeighbors(repeated, static_cast<hypervec::GraphId>(node)));
  }
}

TEST(NNDescentBuilder, ParallelRefinementMatchesSequentialGraphAndStats) {
  constexpr size_t kCount = 128;
  std::vector<float> values(kCount);
  for (size_t node = 0; node < kCount; ++node) {
    values[node] = static_cast<float>((node * 37) % kCount);
  }
  hypervec::NNDescentOptions options{10, 15, 0.001, 42};
  ScalarDistanceComputer sequential_distance(values);
  hypervec::NNDescentStats baseline_stats;
  const auto baseline = hypervec::NNDescentBuilder(options).Build(
      sequential_distance, kCount, &baseline_stats);

  for (size_t threads : {size_t{4}, size_t{32}, size_t{64}}) {
    options.build_threads = threads;
    ScalarDistanceComputer distance(values);
    hypervec::NNDescentStats stats;
    const auto graph = hypervec::NNDescentBuilder(options).Build(
        distance, kCount, &stats,
        [&values] { return std::make_unique<ScalarDistanceComputer>(values); });
    EXPECT_EQ(stats.iterations, baseline_stats.iterations);
    EXPECT_EQ(stats.neighbor_updates, baseline_stats.neighbor_updates);
    EXPECT_EQ(stats.refinement_distance_computations,
              baseline_stats.refinement_distance_computations);
    EXPECT_EQ(stats.converged, baseline_stats.converged);
    for (size_t node = 0; node < kCount; ++node) {
      EXPECT_EQ(CopyNeighbors(graph, static_cast<hypervec::GraphId>(node)),
                CopyNeighbors(baseline, static_cast<hypervec::GraphId>(node)));
    }
  }
}

TEST(NNDescentBuilder, ParallelFailuresDoNotPublishStats) {
  std::vector<float> values(32);
  for (size_t node = 0; node < values.size(); ++node) {
    values[node] = static_cast<float>(node);
  }
  hypervec::NNDescentOptions options{6, 4, 0.0, 42};
  options.build_threads = 4;
  hypervec::NNDescentBuilder builder(options);
  ScalarDistanceComputer distance(values);
  hypervec::NNDescentStats stats;
  stats.iterations = 9;
  EXPECT_THROW(builder.Build(distance, values.size(), &stats),
               hypervec::HypervecException);
  EXPECT_EQ(stats.iterations, 9U);
  EXPECT_THROW(
      builder.Build(distance, values.size(), &stats,
                    [] { return std::make_unique<NaNDistanceComputer>(); }),
      hypervec::HypervecException);
  EXPECT_EQ(stats.iterations, 9U);
}

TEST(NNDescentBuilder, HandlesEmptyAndSingletonGraphs) {
  ScalarDistanceComputer distance({1.0F});
  const hypervec::NNDescentBuilder builder(
      hypervec::NNDescentOptions{3, 2, 0.0, 1});
  hypervec::NNDescentStats stats;

  const auto empty = builder.Build(distance, 0, &stats);
  EXPECT_EQ(empty.NodeCount(), 0U);
  EXPECT_TRUE(stats.converged);

  stats.Reset();
  const auto singleton = builder.Build(distance, 1, &stats);
  EXPECT_EQ(singleton.NodeCount(), 1U);
  EXPECT_TRUE(singleton.Neighbors(0).empty());
  EXPECT_TRUE(stats.converged);
  EXPECT_EQ(stats.initial_distance_computations, 0U);
}

TEST(NNDescentBuilder, ValidatesOptionsAndDoesNotPublishFailedStats) {
  EXPECT_THROW(
      (hypervec::NNDescentBuilder(hypervec::NNDescentOptions{0, 1, 0.0, 1})),
      hypervec::HypervecException);
  EXPECT_THROW(
      (hypervec::NNDescentBuilder(hypervec::NNDescentOptions{1, 0, 0.0, 1})),
      hypervec::HypervecException);
  EXPECT_THROW(
      (hypervec::NNDescentBuilder(hypervec::NNDescentOptions{1, 1, -0.1, 1})),
      hypervec::HypervecException);
  EXPECT_THROW(
      (hypervec::NNDescentBuilder(hypervec::NNDescentOptions{1, 1, 1.1, 1})),
      hypervec::HypervecException);
  EXPECT_THROW((hypervec::NNDescentBuilder(hypervec::NNDescentOptions{
                   1, 1, std::numeric_limits<double>::infinity(), 1})),
               hypervec::HypervecException);
  hypervec::NNDescentOptions invalid_sample_rate;
  invalid_sample_rate.sample_rate = 0.0;
  EXPECT_THROW((hypervec::NNDescentBuilder{invalid_sample_rate}),
               hypervec::HypervecException);
  invalid_sample_rate.sample_rate = 1.1;
  EXPECT_THROW((hypervec::NNDescentBuilder{invalid_sample_rate}),
               hypervec::HypervecException);
  invalid_sample_rate.sample_rate = std::numeric_limits<double>::infinity();
  EXPECT_THROW((hypervec::NNDescentBuilder{invalid_sample_rate}),
               hypervec::HypervecException);
  hypervec::NNDescentOptions invalid_threads;
  invalid_threads.build_threads = 0;
  EXPECT_THROW((hypervec::NNDescentBuilder{invalid_threads}),
               hypervec::HypervecException);

  NaNDistanceComputer distance;
  const hypervec::NNDescentBuilder builder(
      hypervec::NNDescentOptions{1, 2, 0.0, 1});
  hypervec::NNDescentStats stats;
  stats.iterations = 9;
  EXPECT_THROW(builder.Build(distance, 2, &stats), hypervec::HypervecException);
  EXPECT_EQ(stats.iterations, 9U);
}

TEST(NNDescentStats, ResetAndCombineAccumulateBuilds) {
  hypervec::NNDescentStats aggregate;
  hypervec::NNDescentStats update;
  update.iterations = 2;
  update.initial_distance_computations = 3;
  update.refinement_distance_computations = 5;
  update.neighbor_updates = 6;
  update.sampled_old_neighbors = 7;
  update.sampled_new_neighbors = 8;
  update.sampled_neighbors_trimmed = 9;
  update.peak_sampled_neighbors = 10;
  update.converged = true;

  aggregate.Combine(update);
  EXPECT_EQ(aggregate.iterations, 2U);
  EXPECT_EQ(aggregate.initial_distance_computations, 3U);
  EXPECT_EQ(aggregate.refinement_distance_computations, 5U);
  EXPECT_EQ(aggregate.neighbor_updates, 6U);
  EXPECT_EQ(aggregate.sampled_old_neighbors, 7U);
  EXPECT_EQ(aggregate.sampled_new_neighbors, 8U);
  EXPECT_EQ(aggregate.sampled_neighbors_trimmed, 9U);
  EXPECT_EQ(aggregate.peak_sampled_neighbors, 10U);
  EXPECT_TRUE(aggregate.converged);

  aggregate.Reset();
  EXPECT_EQ(aggregate.iterations, 0U);
  EXPECT_FALSE(aggregate.converged);
}
