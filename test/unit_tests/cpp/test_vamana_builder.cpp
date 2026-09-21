/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_validation.h>
#include <index/graph/vamana_builder.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace {

class ScalarDistanceComputer final : public hypervec::DistanceComputer {
 public:
  explicit ScalarDistanceComputer(std::vector<float> values)
      : values_(std::move(values)) {}

  void SetQuery(const float* query) override { query_ = *query; }

  float operator()(hypervec::idx_t index) override {
    return SquaredDistance(query_, values_.at(static_cast<size_t>(index)));
  }

  float symmetric_dis(hypervec::idx_t lhs, hypervec::idx_t rhs) override {
    return SquaredDistance(values_.at(static_cast<size_t>(lhs)),
                           values_.at(static_cast<size_t>(rhs)));
  }

 private:
  static float SquaredDistance(float lhs, float rhs) {
    const float difference = lhs - rhs;
    return difference * difference;
  }

  std::vector<float> values_;
  float query_ = 0.0F;
};

class TableDistanceComputer final : public hypervec::DistanceComputer {
 public:
  explicit TableDistanceComputer(std::vector<std::vector<float>> distances)
      : distances_(std::move(distances)) {}

  void SetQuery(const float*) override {}
  float operator()(hypervec::idx_t index) override {
    return distances_.at(0).at(static_cast<size_t>(index));
  }
  float symmetric_dis(hypervec::idx_t lhs, hypervec::idx_t rhs) override {
    return distances_.at(static_cast<size_t>(lhs)).at(static_cast<size_t>(rhs));
  }

 private:
  std::vector<std::vector<float>> distances_;
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

}  // namespace

TEST(VamanaRobustPruner, AlphaControlsDiversity) {
  TableDistanceComputer distance({
      {0.0F, 10.0F, 11.0F},
      {10.0F, 0.0F, 8.0F},
      {11.0F, 8.0F, 0.0F},
  });
  const std::vector<hypervec::NeighborCandidate> candidates = {{1, 10.0F},
                                                               {2, 11.0F}};

  hypervec::GraphPruneStats strict_stats;
  const auto strict = hypervec::VamanaRobustPruner(1.0F).Prune(
      candidates, 2, distance, &strict_stats);
  const auto relaxed =
      hypervec::VamanaRobustPruner(1.5F).Prune(candidates, 2, distance);

  ASSERT_EQ(strict.size(), 1U);
  EXPECT_EQ(strict.front().id, 1);
  EXPECT_EQ(strict_stats.rejected, 1U);
  ASSERT_EQ(relaxed.size(), 2U);
  EXPECT_EQ(relaxed[0].id, 1);
  EXPECT_EQ(relaxed[1].id, 2);
}

TEST(VamanaRobustPruner, RejectsInvalidInputsWithoutPublishingStats) {
  EXPECT_THROW(hypervec::VamanaRobustPruner(0.9F), hypervec::HypervecException);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(hypervec::VamanaRobustPruner(nan)),
               hypervec::HypervecException);

  ScalarDistanceComputer distance({0.0F, 1.0F});
  const hypervec::VamanaRobustPruner pruner(1.2F);
  const std::vector<hypervec::NeighborCandidate> one = {{1, 1.0F}};
  const std::vector<hypervec::NeighborCandidate> duplicates = {{1, 1.0F},
                                                               {1, 2.0F}};
  const std::vector<hypervec::NeighborCandidate> negative = {{-1, 1.0F}};
  EXPECT_THROW(pruner.Prune(one, 0, distance), hypervec::HypervecException);
  EXPECT_THROW(pruner.Prune(duplicates, 2, distance),
               hypervec::HypervecException);
  EXPECT_THROW(pruner.Prune(negative, 1, distance),
               hypervec::HypervecException);

  NaNDistanceComputer nan_distance;
  hypervec::GraphPruneStats stats;
  stats.accepted = 7;
  const std::vector<hypervec::NeighborCandidate> pair = {{0, 0.0F}, {1, 1.0F}};
  EXPECT_THROW(pruner.Prune(pair, 2, nan_distance, &stats),
               hypervec::HypervecException);
  EXPECT_EQ(stats.accepted, 7U);
}

TEST(VamanaBuilder, BuildsDeterministicBoundedGraph) {
  constexpr size_t kCount = 96;
  std::vector<float> values(kCount);
  for (size_t node = 0; node < kCount; ++node) {
    values[node] = static_cast<float>((node * 37) % kCount);
  }
  const hypervec::VamanaBuildOptions options{8, 20, 40, 1.2F, 2, 42};
  const hypervec::VamanaBuilder builder(options);
  ScalarDistanceComputer distance(values);
  hypervec::VamanaBuildStats stats;
  const auto graph = builder.Build(distance, values.size(), 48, &stats);

  ScalarDistanceComputer repeated_distance(values);
  const auto repeated = builder.Build(repeated_distance, values.size(), 48);
  const auto report = hypervec::ValidateGraph(graph, 48);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, values.size());
  EXPECT_LE(graph.MaxDegree(), options.max_degree);
  EXPECT_EQ(stats.passes_completed, options.build_passes);
  EXPECT_EQ(stats.nodes_processed, values.size() * options.build_passes);
  EXPECT_GT(stats.search.distance_computations, 0U);
  EXPECT_GT(stats.pruning.candidates_examined, 0U);
  EXPECT_GT(stats.reciprocal_edges_added, 0U);
  for (size_t node = 0; node < values.size(); ++node) {
    EXPECT_EQ(CopyNeighbors(graph, static_cast<hypervec::GraphId>(node)),
              CopyNeighbors(repeated, static_cast<hypervec::GraphId>(node)));
  }
}

TEST(VamanaBuilder, RepairsDuplicateVectorReachabilityWithinDegree) {
  constexpr size_t kCount = 20;
  std::vector<float> values(kCount, 0.0F);
  ScalarDistanceComputer distance(values);
  const hypervec::VamanaBuilder builder(
      hypervec::VamanaBuildOptions{4, 8, 12, 1.2F, 2, 42});
  hypervec::VamanaBuildStats stats;

  const auto graph = builder.Build(distance, values.size(), 0, &stats);

  const auto report = hypervec::ValidateGraph(graph, 0);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, values.size());
  EXPECT_EQ(graph.MaxDegree(), 4U);
  EXPECT_GT(stats.connectivity_edges_added, 0U);
  EXPECT_GT(stats.connectivity_edges_replaced, 0U);
}

TEST(VamanaBuilder, HandlesEmptyAndSingletonGraphs) {
  const hypervec::VamanaBuilder builder;
  ScalarDistanceComputer distance({1.0F});
  const auto empty = builder.Build(distance, 0, hypervec::kInvalidGraphId);
  EXPECT_EQ(empty.NodeCount(), 0U);
  EXPECT_THROW(builder.Build(distance, 0, 0), hypervec::HypervecException);

  hypervec::VamanaBuildStats stats;
  const auto singleton = builder.Build(distance, 1, 0, &stats);
  EXPECT_EQ(singleton.NodeCount(), 1U);
  EXPECT_TRUE(singleton.Neighbors(0).empty());
  EXPECT_EQ(stats.nodes_processed, 0U);
}

TEST(VamanaBuilder, RejectsInvalidOptionsAndDistancesTransactionally) {
  EXPECT_THROW((hypervec::VamanaBuilder(
                   hypervec::VamanaBuildOptions{0, 1, 1, 1.0F, 1, 0})),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::VamanaBuilder(
                   hypervec::VamanaBuildOptions{4, 3, 4, 1.0F, 1, 0})),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::VamanaBuilder(
                   hypervec::VamanaBuildOptions{4, 4, 3, 1.0F, 1, 0})),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::VamanaBuilder(
                   hypervec::VamanaBuildOptions{4, 4, 4, 0.9F, 1, 0})),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::VamanaBuilder(
                   hypervec::VamanaBuildOptions{4, 4, 4, 1.0F, 0, 0})),
               hypervec::HypervecException);
  auto invalid_threads = hypervec::VamanaBuildOptions{};
  invalid_threads.build_threads = 0;
  EXPECT_THROW(static_cast<void>(hypervec::VamanaBuilder(invalid_threads)),
               hypervec::HypervecException);

  const hypervec::VamanaBuilder builder(
      hypervec::VamanaBuildOptions{1, 1, 2, 1.2F, 1, 42});
  NaNDistanceComputer nan_distance;
  hypervec::VamanaBuildStats stats;
  stats.nodes_processed = 7;
  EXPECT_THROW(builder.Build(nan_distance, 2, 0, &stats),
               hypervec::HypervecException);
  EXPECT_EQ(stats.nodes_processed, 7U);
  ScalarDistanceComputer valid_distance({0.0F, 1.0F});
  EXPECT_THROW(builder.Build(valid_distance, 2, hypervec::kInvalidGraphId),
               hypervec::HypervecException);
  EXPECT_THROW(builder.Build(valid_distance, 2, 2),
               hypervec::HypervecException);
}

TEST(VamanaBuildStats, ResetAndCombineAccumulateBuilds) {
  hypervec::VamanaBuildStats aggregate;
  hypervec::VamanaBuildStats update;
  update.passes_completed = 1;
  update.nodes_processed = 2;
  update.candidate_distance_computations = 3;
  update.reciprocal_edges_added = 4;
  update.reciprocal_edges_repruned = 5;
  update.reciprocal_edges_rejected = 6;
  update.connectivity_distance_computations = 7;
  update.connectivity_edges_added = 8;
  update.connectivity_edges_replaced = 9;
  update.search.queries = 10;
  update.pruning.accepted = 11;

  aggregate.Combine(update);
  EXPECT_EQ(aggregate.passes_completed, 1U);
  EXPECT_EQ(aggregate.nodes_processed, 2U);
  EXPECT_EQ(aggregate.candidate_distance_computations, 3U);
  EXPECT_EQ(aggregate.reciprocal_edges_added, 4U);
  EXPECT_EQ(aggregate.reciprocal_edges_repruned, 5U);
  EXPECT_EQ(aggregate.reciprocal_edges_rejected, 6U);
  EXPECT_EQ(aggregate.connectivity_distance_computations, 7U);
  EXPECT_EQ(aggregate.connectivity_edges_added, 8U);
  EXPECT_EQ(aggregate.connectivity_edges_replaced, 9U);
  EXPECT_EQ(aggregate.search.queries, 10U);
  EXPECT_EQ(aggregate.pruning.accepted, 11U);

  aggregate.Reset();
  EXPECT_EQ(aggregate.nodes_processed, 0U);
  EXPECT_EQ(aggregate.search.queries, 0U);
}
