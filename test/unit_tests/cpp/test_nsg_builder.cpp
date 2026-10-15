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
#include <index/graph/nsg_builder.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
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

class MalformedGraph final : public hypervec::GraphStorage {
 public:
  size_t NodeCount() const noexcept override { return 2; }
  size_t MaxDegree() const noexcept override { return 1; }
  hypervec::GraphNeighborList Neighbors(hypervec::GraphId node) const override {
    return hypervec::GraphNeighborList(
        hypervec::GraphNeighborView(adjacency_.at(static_cast<size_t>(node))));
  }

 private:
  std::array<std::vector<hypervec::GraphId>, 2> adjacency_ = {{{2}, {0}}};
};

std::vector<hypervec::GraphId> CopyNeighbors(
    const hypervec::GraphStorage& graph, hypervec::GraphId node) {
  const auto neighbors = graph.Neighbors(node);
  return {neighbors.begin(), neighbors.end()};
}

hypervec::MutableBoundedGraph MakeDisconnectedCandidateGraph() {
  hypervec::MutableBoundedGraph graph(8, 2);
  const std::array<std::vector<hypervec::GraphId>, 8> adjacency = {
      std::vector<hypervec::GraphId>{1, 2},
      {0, 2},
      {1, 3},
      {1, 2},
      {5, 6},
      {4, 6},
      {5, 7},
      {5, 6},
  };
  for (size_t node = 0; node < adjacency.size(); ++node) {
    graph.SetNeighbors(static_cast<hypervec::GraphId>(node), adjacency[node]);
  }
  return graph;
}

hypervec::MutableBoundedGraph MakeCompleteGraph(size_t node_count) {
  hypervec::MutableBoundedGraph graph(node_count, node_count - 1);
  for (size_t node = 0; node < node_count; ++node) {
    std::vector<hypervec::GraphId> neighbors;
    neighbors.reserve(node_count - 1);
    for (size_t candidate = 0; candidate < node_count; ++candidate) {
      if (candidate != node) {
        neighbors.push_back(static_cast<hypervec::GraphId>(candidate));
      }
    }
    graph.SetNeighbors(static_cast<hypervec::GraphId>(node), neighbors);
  }
  return graph;
}

}  // namespace

TEST(NSGBuilder, RepairsDirectedConnectivityDeterministically) {
  const std::vector<float> values = {0.0F,   1.0F,   2.0F,   3.0F,
                                     100.0F, 101.0F, 102.0F, 103.0F};
  const hypervec::MutableBoundedGraph candidates =
      MakeDisconnectedCandidateGraph();
  const hypervec::NSGBuilder builder(hypervec::NSGBuildOptions{1, 4, 6, true});
  ScalarDistanceComputer distance(values);
  hypervec::NSGBuildStats first_stats;
  const hypervec::MutableBoundedGraph first =
      builder.Build(candidates, distance, 0, &first_stats);

  ScalarDistanceComputer repeated_distance(values);
  hypervec::NSGBuildStats repeated_stats;
  const hypervec::MutableBoundedGraph repeated =
      builder.Build(candidates, repeated_distance, 0, &repeated_stats);

  const auto report = hypervec::ValidateGraph(first, 0);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, values.size());
  EXPECT_EQ(report.weakly_connected_components, 1U);
  EXPECT_EQ(first.MaxDegree(), 2U);
  EXPECT_EQ(first_stats.pruned_nodes, values.size());
  EXPECT_GT(first_stats.connectivity_edges_added, 0U);
  EXPECT_EQ(first_stats.connectivity_edges_added,
            repeated_stats.connectivity_edges_added);
  for (size_t node = 0; node < values.size(); ++node) {
    EXPECT_EQ(CopyNeighbors(first, static_cast<hypervec::GraphId>(node)),
              CopyNeighbors(repeated, static_cast<hypervec::GraphId>(node)));
  }
}

TEST(NSGBuilder, PrunesDenseCandidatesAndBoundsReciprocalEdges) {
  const std::vector<float> values = {0.0F, 1.0F, 2.0F, 10.0F, 11.0F, 12.0F};
  const hypervec::MutableBoundedGraph candidates =
      MakeCompleteGraph(values.size());
  ScalarDistanceComputer distance(values);
  const hypervec::NSGBuilder builder(hypervec::NSGBuildOptions{2, 4, 6, true});
  hypervec::NSGBuildStats stats;

  const hypervec::MutableBoundedGraph graph =
      builder.Build(candidates, distance, 2, &stats);

  const auto report = hypervec::ValidateGraph(graph, 2);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, values.size());
  EXPECT_LE(report.edge_count, values.size() * graph.MaxDegree());
  EXPECT_GT(stats.pruning.rejected, 0U);
  EXPECT_GT(stats.pruning.candidates_examined, 0U);
}

TEST(NSGBuilder, ComposesWithNNDescentCandidateGraph) {
  constexpr size_t kCount = 128;
  std::vector<float> values(kCount);
  for (size_t node = 0; node < kCount; ++node) {
    values[node] = static_cast<float>((node * 37) % kCount);
  }
  ScalarDistanceComputer distance(values);
  const hypervec::NNDescentBuilder candidate_builder(
      hypervec::NNDescentOptions{10, 15, 0.001, 42});
  const hypervec::MutableBoundedGraph candidates =
      candidate_builder.Build(distance, values.size());
  const hypervec::NSGBuilder builder(
      hypervec::NSGBuildOptions{8, 20, 30, true});

  const hypervec::MutableBoundedGraph graph =
      builder.Build(candidates, distance, 64);

  const auto report = hypervec::ValidateGraph(graph, 64);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, values.size());
  EXPECT_LE(graph.MaxDegree(), 9U);
  EXPECT_LT(report.edge_count, candidates.NodeCount() * candidates.MaxDegree());
}

TEST(NSGBuilder, HandlesEmptyAndSingletonCandidateGraphs) {
  const hypervec::NSGBuilder builder;
  ScalarDistanceComputer distance({1.0F});
  const hypervec::MutableBoundedGraph empty_candidates(1);

  const auto empty =
      builder.Build(empty_candidates, distance, hypervec::kInvalidGraphId);
  EXPECT_EQ(empty.NodeCount(), 0U);
  EXPECT_THROW(builder.Build(empty_candidates, distance, 0),
               hypervec::HypervecException);

  hypervec::MutableBoundedGraph singleton_candidates(1, 1);
  hypervec::NSGBuildStats stats;
  const auto singleton =
      builder.Build(singleton_candidates, distance, 0, &stats);
  EXPECT_EQ(singleton.NodeCount(), 1U);
  EXPECT_TRUE(singleton.Neighbors(0).empty());
  EXPECT_EQ(stats.pruned_nodes, 0U);
}

TEST(NSGBuilder, RejectsInvalidInputsWithoutPublishingStats) {
  EXPECT_THROW((hypervec::NSGBuilder(hypervec::NSGBuildOptions{0, 1, 1, true})),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::NSGBuilder(hypervec::NSGBuildOptions{1, 0, 1, true})),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::NSGBuilder(hypervec::NSGBuildOptions{2, 1, 1, true})),
               hypervec::HypervecException);

  ScalarDistanceComputer distance({0.0F, 1.0F});
  const MalformedGraph malformed;
  const hypervec::NSGBuilder builder(hypervec::NSGBuildOptions{1, 2, 2, true});
  EXPECT_THROW(builder.Build(malformed, distance, 0),
               hypervec::HypervecException);

  const hypervec::MutableBoundedGraph candidates = MakeCompleteGraph(2);
  EXPECT_THROW(builder.Build(candidates, distance, hypervec::kInvalidGraphId),
               hypervec::HypervecException);
  EXPECT_THROW(builder.Build(candidates, distance, 2),
               hypervec::HypervecException);
  NaNDistanceComputer nan_distance;
  hypervec::NSGBuildStats stats;
  stats.pruned_nodes = 7;
  EXPECT_THROW(builder.Build(candidates, nan_distance, 0, &stats),
               hypervec::HypervecException);
  EXPECT_EQ(stats.pruned_nodes, 7U);
}

TEST(NSGBuildStats, ResetAndCombineAccumulateBuilds) {
  hypervec::NSGBuildStats aggregate;
  hypervec::NSGBuildStats update;
  update.pruned_nodes = 2;
  update.candidate_distance_computations = 3;
  update.reciprocal_edges_added = 4;
  update.reciprocal_edges_rejected = 5;
  update.connectivity_distance_computations = 6;
  update.connectivity_edges_added = 7;
  update.search.queries = 8;
  update.pruning.accepted = 9;

  aggregate.Combine(update);
  EXPECT_EQ(aggregate.pruned_nodes, 2U);
  EXPECT_EQ(aggregate.candidate_distance_computations, 3U);
  EXPECT_EQ(aggregate.reciprocal_edges_added, 4U);
  EXPECT_EQ(aggregate.reciprocal_edges_rejected, 5U);
  EXPECT_EQ(aggregate.connectivity_distance_computations, 6U);
  EXPECT_EQ(aggregate.connectivity_edges_added, 7U);
  EXPECT_EQ(aggregate.search.queries, 8U);
  EXPECT_EQ(aggregate.pruning.accepted, 9U);

  aggregate.Reset();
  EXPECT_EQ(aggregate.pruned_nodes, 0U);
  EXPECT_EQ(aggregate.search.queries, 0U);
}
