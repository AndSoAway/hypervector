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
#include <index/graph/nsw_builder.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
#include <stdexcept>
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

bool Contains(hypervec::GraphNeighborView neighbors, hypervec::GraphId node) {
  return std::find(neighbors.begin(), neighbors.end(), node) != neighbors.end();
}

}  // namespace

TEST(NSWIncrementalBuilder, BuildsValidConnectedGraphAndAccumulatesStats) {
  const std::vector<float> values = {0.0F, 10.0F, 5.0F, 6.0F, 20.0F};
  ScalarDistanceComputer distance(values);
  hypervec::MutableBoundedGraph graph(2);
  const hypervec::NSWIncrementalBuilder builder(
      hypervec::NSWBuildOptions{4, true, true});
  hypervec::NSWBuildStats stats;

  hypervec::GraphId entry_point = hypervec::kInvalidGraphId;
  for (size_t node = 0; node < values.size(); ++node) {
    distance.SetQuery(&values[node]);
    const hypervec::NSWInsertionResult result =
        builder.AddNode(distance, graph, entry_point, &stats);
    EXPECT_EQ(result.node_id, static_cast<hypervec::GraphId>(node));
    entry_point = result.entry_point;
  }

  EXPECT_EQ(entry_point, 0);
  EXPECT_EQ(graph.NodeCount(), values.size());
  EXPECT_EQ(stats.inserted_nodes, values.size());
  EXPECT_EQ(stats.search.queries, values.size() - 1);
  EXPECT_GT(stats.reciprocal_updates, 0U);

  const hypervec::GraphValidationReport report =
      hypervec::ValidateGraph(graph, entry_point);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.weakly_connected_components, 1U);
  EXPECT_EQ(report.reachable_nodes, values.size());
}

TEST(NSWIncrementalBuilder, PrunesFullReciprocalNeighborLists) {
  const std::vector<float> values = {0.0F, 10.0F, -10.0F, 1.0F};
  ScalarDistanceComputer distance(values);
  distance.SetQuery(&values[3]);
  hypervec::MutableBoundedGraph graph(3, 2);
  const std::array<hypervec::GraphId, 2> zero = {1, 2};
  const std::array<hypervec::GraphId, 1> one = {0};
  const std::array<hypervec::GraphId, 1> two = {0};
  graph.SetNeighbors(0, zero);
  graph.SetNeighbors(1, one);
  graph.SetNeighbors(2, two);

  const hypervec::NSWIncrementalBuilder builder(
      hypervec::NSWBuildOptions{3, true, true});
  hypervec::NSWBuildStats stats;
  const auto result = builder.AddNode(distance, graph, 0, &stats);

  EXPECT_EQ(result.node_id, 3);
  EXPECT_TRUE(Contains(graph.Neighbors(0), 3));
  EXPECT_FALSE(Contains(graph.Neighbors(0), 1));
  EXPECT_TRUE(Contains(graph.Neighbors(1), 3));
  EXPECT_EQ(stats.pruned_neighbor_lists, 1U);
  EXPECT_EQ(stats.reciprocal_distance_computations, 3U);
  EXPECT_EQ(stats.reciprocal_updates, 2U);
}

TEST(NSWIncrementalBuilder, ValidatesConfigurationAndEntryPointBeforeMutation) {
  ScalarDistanceComputer distance({0.0F, 1.0F});
  const float query = 1.0F;
  distance.SetQuery(&query);

  EXPECT_THROW(hypervec::NSWIncrementalBuilder(hypervec::NSWBuildOptions{0}),
               hypervec::HypervecException);

  hypervec::MutableBoundedGraph empty_graph(2);
  const hypervec::NSWIncrementalBuilder builder;
  EXPECT_THROW(builder.AddNode(distance, empty_graph, 0),
               hypervec::HypervecException);
  EXPECT_EQ(empty_graph.NodeCount(), 0U);

  const auto first =
      builder.AddNode(distance, empty_graph, hypervec::kInvalidGraphId);
  EXPECT_EQ(first.entry_point, 0);
  EXPECT_THROW(builder.AddNode(distance, empty_graph, 1),
               hypervec::HypervecException);
  EXPECT_EQ(empty_graph.NodeCount(), 1U);

  hypervec::MutableBoundedGraph wide_graph(4);
  wide_graph.Resize(1);
  const hypervec::NSWIncrementalBuilder narrow_search(
      hypervec::NSWBuildOptions{2, true, true});
  EXPECT_THROW(narrow_search.AddNode(distance, wide_graph, 0),
               hypervec::HypervecException);
  EXPECT_EQ(wide_graph.NodeCount(), 1U);
}

TEST(NSWIncrementalBuilder, DistanceFailureDoesNotResizeGraph) {
  ScalarDistanceComputer distance({0.0F, 10.0F, -10.0F});
  const float query = 1.0F;
  distance.SetQuery(&query);
  hypervec::MutableBoundedGraph graph(3, 2);
  const std::array<hypervec::GraphId, 2> zero = {1, 2};
  const std::array<hypervec::GraphId, 1> one = {0};
  const std::array<hypervec::GraphId, 1> two = {0};
  graph.SetNeighbors(0, zero);
  graph.SetNeighbors(1, one);
  graph.SetNeighbors(2, two);

  const hypervec::NSWIncrementalBuilder builder(
      hypervec::NSWBuildOptions{3, true, true});
  EXPECT_THROW(builder.AddNode(distance, graph, 0), std::out_of_range);

  EXPECT_EQ(graph.NodeCount(), 3U);
  EXPECT_EQ(std::vector<hypervec::GraphId>(graph.Neighbors(0).begin(),
                                           graph.Neighbors(0).end()),
            std::vector<hypervec::GraphId>(zero.begin(), zero.end()));
}

TEST(NSWBuildStats, ResetAndCombineCoverNestedCounters) {
  hypervec::NSWBuildStats aggregate;
  hypervec::NSWBuildStats update;
  update.inserted_nodes = 2;
  update.reciprocal_updates = 3;
  update.pruned_neighbor_lists = 1;
  update.reciprocal_distance_computations = 4;
  update.search.queries = 2;
  update.pruning.accepted = 5;

  aggregate.Combine(update);
  EXPECT_EQ(aggregate.inserted_nodes, 2U);
  EXPECT_EQ(aggregate.reciprocal_updates, 3U);
  EXPECT_EQ(aggregate.pruned_neighbor_lists, 1U);
  EXPECT_EQ(aggregate.reciprocal_distance_computations, 4U);
  EXPECT_EQ(aggregate.search.queries, 2U);
  EXPECT_EQ(aggregate.pruning.accepted, 5U);

  aggregate.Reset();
  EXPECT_EQ(aggregate.inserted_nodes, 0U);
  EXPECT_EQ(aggregate.search.queries, 0U);
  EXPECT_EQ(aggregate.pruning.accepted, 0U);
}
