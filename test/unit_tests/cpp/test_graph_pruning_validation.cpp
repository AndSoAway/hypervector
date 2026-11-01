/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_storage.h>
#include <index/graph/graph_validation.h>
#include <index/graph/neighbor_pruner.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>

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

class MalformedGraph final : public hypervec::GraphStorage {
 public:
  size_t NodeCount() const noexcept override { return adjacency_.size(); }
  size_t MaxDegree() const noexcept override { return 1; }
  hypervec::GraphNeighborList Neighbors(hypervec::GraphId node) const override {
    return hypervec::GraphNeighborList(
        hypervec::GraphNeighborView(adjacency_.at(static_cast<size_t>(node))));
  }

 private:
  std::vector<std::vector<hypervec::GraphId>> adjacency_ = {
      {0, 1, 1, 9}, {2}, {}};
};

}  // namespace

TEST(NeighborPruner, HnswHeuristicPreservesDiversityAndCanFill) {
  ScalarDistanceComputer distance({0.0F, 1.0F, 1.1F, 10.0F});
  const std::array<hypervec::NeighborCandidate, 3> candidates = {
      hypervec::NeighborCandidate{3, 100.0F},
      hypervec::NeighborCandidate{2, 1.21F},
      hypervec::NeighborCandidate{1, 1.0F},
  };

  hypervec::GraphPruneStats stats;
  const hypervec::HnswHeuristicPruner sparse_pruner;
  const auto sparse = sparse_pruner.Prune(candidates, 2, distance, &stats);
  ASSERT_EQ(sparse.size(), 1U);
  EXPECT_EQ(sparse[0].id, 1);
  EXPECT_EQ(stats.candidates_examined, 3U);
  EXPECT_EQ(stats.distance_computations, 2U);
  EXPECT_EQ(stats.accepted, 1U);
  EXPECT_EQ(stats.rejected, 2U);

  hypervec::GraphPruneStats full_stats;
  const hypervec::HnswHeuristicPruner full_pruner(true);
  const auto full = full_pruner.Prune(candidates, 2, distance, &full_stats);
  ASSERT_EQ(full.size(), 2U);
  EXPECT_EQ(full[0].id, 1);
  EXPECT_EQ(full[1].id, 2);
  EXPECT_EQ(full_stats.accepted, 1U);
  EXPECT_EQ(full_stats.rejected, 2U);
  EXPECT_EQ(full_stats.filled, 1U);
}

TEST(NeighborPruner, RejectsInvalidCapacityAndCandidateIds) {
  ScalarDistanceComputer distance({0.0F, 1.0F, 2.0F});
  const hypervec::HnswHeuristicPruner pruner;
  const std::array<hypervec::NeighborCandidate, 1> valid = {
      hypervec::NeighborCandidate{1, 1.0F}};
  const std::array<hypervec::NeighborCandidate, 1> negative = {
      hypervec::NeighborCandidate{-1, 1.0F}};
  const std::array<hypervec::NeighborCandidate, 2> duplicate = {
      hypervec::NeighborCandidate{1, 1.0F},
      hypervec::NeighborCandidate{1, 2.0F}};
  const std::array<hypervec::NeighborCandidate, 1> nan_distance = {
      hypervec::NeighborCandidate{1,
                                  (std::numeric_limits<float>::quiet_NaN)()}};

  EXPECT_THROW(pruner.Prune(valid, 0, distance), hypervec::HypervecException);
  EXPECT_THROW(pruner.Prune(negative, 1, distance),
               hypervec::HypervecException);
  EXPECT_THROW(pruner.Prune(duplicate, 1, distance),
               hypervec::HypervecException);
  EXPECT_THROW(pruner.Prune(nan_distance, 1, distance),
               hypervec::HypervecException);
}

TEST(GraphValidation, ReportsMalformedEdgesAndDirectedReachability) {
  const MalformedGraph graph;
  const hypervec::GraphValidationReport report =
      hypervec::ValidateGraph(graph, 0);

  EXPECT_FALSE(report.IsStructurallyValid());
  EXPECT_EQ(report.node_count, 3U);
  EXPECT_EQ(report.edge_count, 5U);
  EXPECT_EQ(report.self_loops, 1U);
  EXPECT_EQ(report.duplicate_edges, 1U);
  EXPECT_EQ(report.out_of_range_edges, 1U);
  EXPECT_EQ(report.degree_violations, 1U);
  EXPECT_EQ(report.weakly_connected_components, 1U);
  EXPECT_EQ(report.reachable_nodes, 3U);
  EXPECT_DOUBLE_EQ(report.ReachableRatio(), 1.0);
}

TEST(GraphValidation, DistinguishesWeakComponentsFromReachability) {
  hypervec::MutableBoundedGraph graph(4, 1);
  const std::array<hypervec::GraphId, 1> zero = {1};
  const std::array<hypervec::GraphId, 1> two = {3};
  graph.SetNeighbors(0, zero);
  graph.SetNeighbors(2, two);

  const auto report = hypervec::ValidateGraph(graph, 0);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.weakly_connected_components, 2U);
  EXPECT_EQ(report.reachable_nodes, 2U);
  EXPECT_DOUBLE_EQ(report.ReachableRatio(), 0.5);

  const auto invalid_entry = hypervec::ValidateGraph(graph, 4);
  EXPECT_EQ(invalid_entry.invalid_entry_points, 1U);
  EXPECT_FALSE(invalid_entry.IsStructurallyValid());
}
