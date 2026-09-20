/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_searcher.h>
#include <index/graph/graph_storage.h>
#include <index/graph/visited_table.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <array>
#include <utility>
#include <vector>

namespace {

class ScalarDistanceComputer final : public hypervec::DistanceComputer {
 public:
  explicit ScalarDistanceComputer(std::vector<float> values)
      : values_(std::move(values)) {}

  void SetQuery(const float* query) override {
    ASSERT_NE(query, nullptr);
    query_ = *query;
  }

  float operator()(hypervec::idx_t index) override {
    ++calls_;
    const float difference = values_.at(static_cast<size_t>(index)) - query_;
    return difference * difference;
  }

  float symmetric_dis(hypervec::idx_t lhs, hypervec::idx_t rhs) override {
    const float difference = values_.at(static_cast<size_t>(lhs)) -
                             values_.at(static_cast<size_t>(rhs));
    return difference * difference;
  }

  void distances_batch_4(hypervec::idx_t idx0, hypervec::idx_t idx1,
                         hypervec::idx_t idx2, hypervec::idx_t idx3,
                         float& dis0, float& dis1, float& dis2,
                         float& dis3) override {
    ++batch_calls_;
    DistanceComputer::distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1,
                                        dis2, dis3);
  }

  size_t Calls() const noexcept { return calls_; }
  size_t BatchCalls() const noexcept { return batch_calls_; }

 private:
  std::vector<float> values_;
  float query_ = 0.0F;
  size_t calls_ = 0;
  size_t batch_calls_ = 0;
};

void PopulateChain(hypervec::MutableGraphStorage* graph) {
  graph->Resize(5);
  const std::array<hypervec::GraphId, 1> zero = {1};
  const std::array<hypervec::GraphId, 1> one = {2};
  const std::array<hypervec::GraphId, 1> two = {3};
  const std::array<hypervec::GraphId, 1> three = {4};
  graph->SetNeighbors(0, zero);
  graph->SetNeighbors(1, one);
  graph->SetNeighbors(2, two);
  graph->SetNeighbors(3, three);
  graph->SetNeighbors(4, {});
}

std::vector<hypervec::GraphSearchResult> SearchChain(
    const hypervec::GraphStorage& graph, hypervec::GraphSearchStats* stats) {
  ScalarDistanceComputer distance({10.0F, 8.0F, 0.0F, 1.0F, 20.0F});
  const float query = 0.0F;
  distance.SetQuery(&query);
  hypervec::IDSelectorRange selector(2, 4);
  hypervec::VisitedTable visited(graph.NodeCount(), false);
  const hypervec::GraphSearcher searcher(graph);
  const std::array<hypervec::GraphId, 1> entry = {0};
  return searcher.Search(distance, entry,
                         hypervec::GraphSearchOptions{3, true, &selector},
                         &visited, stats);
}

}  // namespace

TEST(VisitedTable, VectorAndHashModesStartFreshEpochs) {
  for (bool use_hashset : {false, true}) {
    hypervec::VisitedTable visited(4, use_hashset);
    EXPECT_EQ(visited.Size(), 4U);
    EXPECT_TRUE(visited.set(2));
    EXPECT_FALSE(visited.set(2));
    EXPECT_TRUE(visited.get(2));
    visited.advance();
    EXPECT_FALSE(visited.get(2));
    EXPECT_TRUE(visited.set(2));
  }
}

TEST(GraphSearcher, FilteredNodesRemainNavigationIntermediatesAcrossLayouts) {
  hypervec::MutableBoundedGraph mutable_graph(1);
  PopulateChain(&mutable_graph);
  const hypervec::FixedDegreeGraph fixed_graph(mutable_graph);
  const hypervec::CsrGraph csr_graph(mutable_graph);

  for (const hypervec::GraphStorage* graph :
       {static_cast<const hypervec::GraphStorage*>(&mutable_graph),
        static_cast<const hypervec::GraphStorage*>(&fixed_graph),
        static_cast<const hypervec::GraphStorage*>(&csr_graph)}) {
    hypervec::GraphSearchStats stats;
    const auto results = SearchChain(*graph, &stats);
    ASSERT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0].id, 2);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0F);
    EXPECT_EQ(results[1].id, 3);
    EXPECT_FLOAT_EQ(results[1].distance, 1.0F);
    EXPECT_EQ(stats.queries, 1U);
    EXPECT_EQ(stats.exhausted_queries, 1U);
    EXPECT_EQ(stats.visited_nodes, 5U);
    EXPECT_EQ(stats.expanded_nodes, 5U);
    EXPECT_EQ(stats.traversed_edges, 4U);
  }
}

TEST(GraphSearcher, RelativeDistanceControlsEarlyTermination) {
  hypervec::MutableBoundedGraph graph(5, 2);
  const std::array<hypervec::GraphId, 2> zero = {1, 2};
  const std::array<hypervec::GraphId, 1> one = {3};
  const std::array<hypervec::GraphId, 1> two = {4};
  graph.SetNeighbors(0, zero);
  graph.SetNeighbors(1, one);
  graph.SetNeighbors(2, two);

  ScalarDistanceComputer distance({10.0F, 5.0F, 6.0F, 1.0F, 0.5F});
  const float query = 0.0F;
  distance.SetQuery(&query);
  const std::array<hypervec::GraphId, 1> entry = {0};
  const hypervec::GraphSearcher searcher(graph);

  hypervec::VisitedTable relative_visited(graph.NodeCount(), false);
  const auto relative = searcher.Search(
      distance, entry, hypervec::GraphSearchOptions{2, true, nullptr},
      &relative_visited);
  ASSERT_EQ(relative.size(), 2U);
  EXPECT_EQ(relative[0].id, 3);
  EXPECT_EQ(relative[1].id, 1);

  hypervec::VisitedTable exhaustive_visited(graph.NodeCount(), false);
  const auto exhaustive = searcher.Search(
      distance, entry, hypervec::GraphSearchOptions{2, false, nullptr},
      &exhaustive_visited);
  ASSERT_EQ(exhaustive.size(), 2U);
  EXPECT_EQ(exhaustive[0].id, 4);
  EXPECT_EQ(exhaustive[1].id, 3);
}

TEST(GraphSearcher, EqualDistancesUseNodeIdAsStableTieBreak) {
  hypervec::MutableBoundedGraph graph(3, 2);
  const std::array<hypervec::GraphId, 2> neighbors = {2, 1};
  graph.SetNeighbors(0, neighbors);

  ScalarDistanceComputer distance({10.0F, 1.0F, 1.0F});
  const float query = 0.0F;
  distance.SetQuery(&query);
  const std::array<hypervec::GraphId, 1> entry = {0};
  hypervec::VisitedTable visited(graph.NodeCount(), false);
  const hypervec::GraphSearcher searcher(graph);
  const auto results = searcher.Search(
      distance, entry, hypervec::GraphSearchOptions{1, false, nullptr},
      &visited);

  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(results[0].id, 1);
  EXPECT_FLOAT_EQ(results[0].distance, 1.0F);
}

TEST(GraphSearcher, PrecomputedSeedsAvoidDuplicateDistanceComputations) {
  hypervec::MutableBoundedGraph graph(1);
  PopulateChain(&graph);
  const hypervec::GraphSearcher searcher(graph);
  const std::array<hypervec::GraphSearchSeed, 2> seeds = {
      hypervec::GraphSearchSeed{0, 100.0F},
      hypervec::GraphSearchSeed{0, -1.0F}};
  ScalarDistanceComputer distance({10.0F, 8.0F, 0.0F, 1.0F, 20.0F});
  const float query = 0.0F;
  distance.SetQuery(&query);
  hypervec::VisitedTable visited(graph.NodeCount(), false);
  hypervec::GraphSearchStats stats;

  const auto results = searcher.Search(
      distance, seeds, hypervec::GraphSearchOptions{3, false, nullptr},
      &visited, &stats);

  ASSERT_EQ(results.size(), 3U);
  EXPECT_EQ(results[0].id, 2);
  EXPECT_EQ(results[1].id, 3);
  EXPECT_EQ(results[2].id, 1);
  EXPECT_EQ(distance.Calls(), 4U);
  EXPECT_EQ(stats.distance_computations, 4U);
  EXPECT_EQ(stats.visited_nodes, 5U);
}

TEST(GraphSearcher, BatchesFourNeighborDistancesWithoutChangingOrder) {
  hypervec::MutableBoundedGraph graph(6, 5);
  const std::array<hypervec::GraphId, 5> neighbors = {5, 4, 3, 2, 1};
  graph.SetNeighbors(0, neighbors);
  ScalarDistanceComputer distance({10.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const float query = 0.0F;
  distance.SetQuery(&query);
  const std::array<hypervec::GraphId, 1> entry = {0};
  hypervec::VisitedTable visited(graph.NodeCount(), false);
  hypervec::GraphSearchStats stats;
  const hypervec::GraphSearcher searcher(graph);

  const auto results = searcher.Search(
      distance, entry, hypervec::GraphSearchOptions{6, false, nullptr},
      &visited, &stats);

  ASSERT_EQ(results.size(), 6U);
  for (size_t index = 0; index < results.size(); ++index) {
    EXPECT_EQ(results[index].id, static_cast<hypervec::GraphId>(5 - index));
  }
  EXPECT_EQ(distance.BatchCalls(), 1U);
  EXPECT_EQ(distance.Calls(), 6U);
  EXPECT_EQ(stats.distance_computations, 6U);
  EXPECT_EQ(stats.visited_nodes, 6U);
}

TEST(GraphSearcher, NavigationBoundIsIndependentOfResultFiltering) {
  hypervec::MutableBoundedGraph graph(4, 2);
  const std::array<hypervec::GraphId, 2> start_neighbors = {1, 2};
  const std::array<hypervec::GraphId, 1> bridge = {3};
  graph.SetNeighbors(0, start_neighbors);
  graph.SetNeighbors(2, bridge);
  ScalarDistanceComputer distance({10.0F, 1.0F, 2.0F, 0.0F});
  const float query = 0.0F;
  distance.SetQuery(&query);
  const std::array<hypervec::GraphId, 1> entry = {0};
  hypervec::IDSelectorRange selector(3, 4);
  const hypervec::GraphSearcher searcher(graph);

  hypervec::VisitedTable result_visited(graph.NodeCount(), false);
  const auto result_bound =
      searcher.Search(distance, entry,
                      hypervec::GraphSearchOptions{
                          1, true, &selector,
                          hypervec::GraphSearchFrontierPolicy::kResultBound},
                      &result_visited);
  ASSERT_EQ(result_bound.size(), 1U);
  EXPECT_EQ(result_bound[0].id, 3);

  hypervec::VisitedTable navigation_visited(graph.NodeCount(), false);
  const auto navigation_bound = searcher.Search(
      distance, entry,
      hypervec::GraphSearchOptions{
          1, true, &selector,
          hypervec::GraphSearchFrontierPolicy::kNavigationBound},
      &navigation_visited);
  EXPECT_TRUE(navigation_bound.empty());
}

TEST(GraphSearcher, ValidatesOptionsEntryPointsAndVisitedSize) {
  hypervec::MutableBoundedGraph graph(2, 1);
  ScalarDistanceComputer distance({0.0F, 1.0F});
  const float query = 0.0F;
  distance.SetQuery(&query);
  const hypervec::GraphSearcher searcher(graph);
  const std::array<hypervec::GraphId, 1> valid_entry = {0};
  const std::array<hypervec::GraphId, 1> invalid_entry = {2};
  const std::array<hypervec::GraphSearchSeed, 1> invalid_seed = {
      hypervec::GraphSearchSeed{2, 0.0F}};
  hypervec::VisitedTable visited(graph.NodeCount(), false);
  hypervec::VisitedTable wrong_size(1, false);

  EXPECT_THROW(
      searcher.Search(distance, valid_entry,
                      hypervec::GraphSearchOptions{0, true, nullptr}, &visited),
      hypervec::HypervecException);
  EXPECT_THROW(searcher.Search(distance, invalid_entry, {}, &visited),
               hypervec::HypervecException);
  EXPECT_THROW(searcher.Search(distance, valid_entry, {}, &wrong_size),
               hypervec::HypervecException);
  EXPECT_THROW(searcher.Search(distance, valid_entry, {}, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(searcher.Search(distance, invalid_seed, {}, &visited),
               hypervec::HypervecException);
}
