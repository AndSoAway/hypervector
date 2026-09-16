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
#include <index/graph/graph_validation.h>
#include <index/graph/neighbor_pruner.h>
#include <index/graph/visited_table.h>
#include <index/hnsw/hnsw_graph_adapter.h>
#include <index/hnsw/index_hnsw.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <array>
#include <cstddef>
#include <cstdint>
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
  float query_ = 0.0f;
};

class RejectIdSelector final : public hypervec::IDSelector {
 public:
  explicit RejectIdSelector(hypervec::idx_t rejected) : rejected_(rejected) {}

  bool IsMember(hypervec::idx_t id) const override { return id != rejected_; }

 private:
  hypervec::idx_t rejected_;
};

class MalformedGraph final : public hypervec::GraphStorage {
 public:
  MalformedGraph() : adjacency_{{0, 1, 1, 9}, {2}, {}} {}

  size_t NodeCount() const noexcept override { return adjacency_.size(); }
  size_t MaxDegree() const noexcept override { return 1; }

  hypervec::GraphNeighborView Neighbors(hypervec::GraphId node) const override {
    return adjacency_.at(static_cast<size_t>(node));
  }

 private:
  std::vector<std::vector<hypervec::GraphId>> adjacency_;
};

void PopulateSearchGraph(hypervec::MutableGraphStorage* graph) {
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

hypervec::GraphSearchStats SearchGraph(const hypervec::GraphStorage& graph) {
  ScalarDistanceComputer distance({10.0f, 8.0f, 0.0f, 1.0f, 20.0f});
  const float query = 0.0f;
  distance.SetQuery(&query);
  RejectIdSelector selector(1);

  std::array<float, 2> distances;
  std::array<hypervec::idx_t, 2> labels;
  using Comparator = hypervec::CMax<float, hypervec::idx_t>;
  using Handler = hypervec::HeapBlockResultHandler<Comparator>;
  Handler block(1, distances.data(), labels.data(), labels.size());
  Handler::SingleResultHandler results(block);
  results.begin(0);

  hypervec::GraphCandidateQueue candidates(8);
  candidates.push(0, distance(0));
  hypervec::VisitedTable visited(graph.NodeCount(), false);
  hypervec::GraphSearchStats stats;
  const hypervec::GraphSearcher searcher(graph);
  const int result_count =
      searcher.Search(distance, results, candidates, visited,
                      hypervec::GraphSearchOptions{8, true, &selector}, &stats);
  results.end();

  EXPECT_GE(result_count, 2);
  EXPECT_EQ(labels[0], 2);
  EXPECT_FLOAT_EQ(distances[0], 0.0f);
  EXPECT_EQ(labels[1], 3);
  EXPECT_FLOAT_EQ(distances[1], 1.0f);
  EXPECT_NE(labels[0], 1);
  EXPECT_NE(labels[1], 1);
  return stats;
}

std::vector<float> MakeVectors(hypervec::idx_t count,
                               hypervec::idx_t dimension) {
  std::vector<float> vectors(static_cast<size_t>(count * dimension));
  for (hypervec::idx_t row = 0; row < count; ++row) {
    for (hypervec::idx_t column = 0; column < dimension; ++column) {
      vectors[static_cast<size_t>(row * dimension + column)] =
          static_cast<float>((row * 17 + column * 7) % 29) / 29.0f;
    }
  }
  return vectors;
}

}  // namespace

TEST(GraphStorage, MutableFixedAndCsrLayoutsAgree) {
  hypervec::MutableBoundedGraph mutable_graph(5, 2);
  const std::array<hypervec::GraphId, 2> neighbors0 = {1, 2};
  const std::array<hypervec::GraphId, 2> neighbors1 = {2, 3};
  const std::array<hypervec::GraphId, 1> neighbors2 = {3};
  const std::array<hypervec::GraphId, 1> neighbors3 = {4};
  const std::array<hypervec::GraphId, 1> neighbors4 = {0};
  mutable_graph.SetNeighbors(0, neighbors0);
  mutable_graph.SetNeighbors(1, neighbors1);
  mutable_graph.SetNeighbors(2, neighbors2);
  mutable_graph.SetNeighbors(3, neighbors3);
  mutable_graph.SetNeighbors(4, neighbors4);

  EXPECT_FALSE(mutable_graph.AddNeighbor(0, 1));
  EXPECT_TRUE(mutable_graph.RemoveNeighbor(0, 2));
  EXPECT_TRUE(mutable_graph.AddNeighbor(0, 2));
  EXPECT_THROW(mutable_graph.AddNeighbor(0, 3), hypervec::HypervecException);
  EXPECT_THROW(
      mutable_graph.SetNeighbors(0, std::array<hypervec::GraphId, 1>{0}),
      hypervec::HypervecException);
  EXPECT_THROW(mutable_graph.Resize(2), hypervec::HypervecException);
  EXPECT_EQ(mutable_graph.NodeCount(), 5U);

  hypervec::FixedDegreeGraph fixed_graph(5, 3);
  for (hypervec::GraphId node = 0; node < 5; ++node) {
    fixed_graph.SetNeighbors(node, mutable_graph.Neighbors(node));
  }
  EXPECT_THROW(fixed_graph.Resize(2), hypervec::HypervecException);
  EXPECT_EQ(fixed_graph.NodeCount(), 5U);
  const hypervec::CsrGraph csr_graph(fixed_graph);

  for (hypervec::GraphId node = 0; node < 5; ++node) {
    const auto expected = mutable_graph.Neighbors(node);
    const auto fixed = fixed_graph.Neighbors(node);
    const auto csr = csr_graph.Neighbors(node);
    EXPECT_EQ(std::vector<hypervec::GraphId>(fixed.begin(), fixed.end()),
              std::vector<hypervec::GraphId>(expected.begin(), expected.end()));
    EXPECT_EQ(std::vector<hypervec::GraphId>(csr.begin(), csr.end()),
              std::vector<hypervec::GraphId>(expected.begin(), expected.end()));
  }

  const hypervec::GraphValidationReport report =
      hypervec::ValidateGraph(csr_graph, 0);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.weakly_connected_components, 1U);
  EXPECT_EQ(report.reachable_nodes, 5U);
  EXPECT_DOUBLE_EQ(report.ReachableRatio(), 1.0);
}

TEST(GraphStorage, RejectsMalformedLayoutsAndReportsForeignImplementations) {
  EXPECT_THROW(hypervec::CsrGraph({0, 2, 1}, {1}), hypervec::HypervecException);
  EXPECT_THROW(hypervec::CsrGraph({0, 1}, {4}), hypervec::HypervecException);

  const MalformedGraph malformed;
  const hypervec::GraphValidationReport report =
      hypervec::ValidateGraph(malformed, 0);
  EXPECT_FALSE(report.IsStructurallyValid());
  EXPECT_EQ(report.self_loops, 1U);
  EXPECT_EQ(report.duplicate_edges, 1U);
  EXPECT_EQ(report.out_of_range_edges, 1U);
  EXPECT_EQ(report.degree_violations, 1U);

  const hypervec::CsrGraph empty_edges({0, 0, 0}, {});
  EXPECT_TRUE(empty_edges.Neighbors(0).empty());
  EXPECT_TRUE(empty_edges.Neighbors(1).empty());
  empty_edges.Prefetch(0);
  empty_edges.Prefetch(1);
}

TEST(GraphStorage, ReportsWeakComponentsAndDirectedReachability) {
  hypervec::MutableBoundedGraph graph(4, 1);
  graph.SetNeighbors(0, {});
  graph.SetNeighbors(1, std::array<hypervec::GraphId, 1>{0});
  graph.SetNeighbors(2, std::array<hypervec::GraphId, 1>{3});
  graph.SetNeighbors(3, {});

  const hypervec::GraphValidationReport report =
      hypervec::ValidateGraph(graph, 0);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.weakly_connected_components, 2U);
  EXPECT_EQ(report.reachable_nodes, 1U);
  EXPECT_DOUBLE_EQ(report.ReachableRatio(), 0.25);
}

TEST(GraphSearcher, CandidateQueueRejectsInvalidCapacity) {
  EXPECT_THROW(hypervec::GraphCandidateQueue(0), hypervec::HypervecException);
  EXPECT_THROW(hypervec::GraphCandidateQueue(-1), hypervec::HypervecException);
}

TEST(GraphSearcher, FilteredNodesRemainNavigationIntermediates) {
  hypervec::MutableBoundedGraph mutable_graph(1);
  PopulateSearchGraph(&mutable_graph);
  const hypervec::GraphSearchStats mutable_stats = SearchGraph(mutable_graph);

  const hypervec::CsrGraph static_graph(mutable_graph);
  const hypervec::GraphSearchStats static_stats = SearchGraph(static_graph);

  EXPECT_EQ(mutable_stats.queries, 1U);
  EXPECT_EQ(mutable_stats.exhausted_queries, 1U);
  EXPECT_EQ(mutable_stats.distance_computations, 4U);
  EXPECT_EQ(mutable_stats.visited_nodes, 5U);
  EXPECT_EQ(mutable_stats.expanded_nodes, 5U);
  EXPECT_EQ(mutable_stats.traversed_edges, 4U);
  EXPECT_EQ(static_stats.distance_computations,
            mutable_stats.distance_computations);
  EXPECT_EQ(static_stats.visited_nodes, mutable_stats.visited_nodes);
  EXPECT_EQ(static_stats.expanded_nodes, mutable_stats.expanded_nodes);
}

TEST(GraphSearcher, PreservesLegacyStoppingAndInitialResultCount) {
  const hypervec::MutableBoundedGraph graph(3, 1);
  ScalarDistanceComputer distance({0.0f, 1.0f, 2.0f});
  const float query = 0.0f;
  distance.SetQuery(&query);

  const auto run_search = [&](bool check_relative_distance) {
    std::array<float, 3> distances;
    std::array<hypervec::idx_t, 3> labels;
    using Comparator = hypervec::CMax<float, hypervec::idx_t>;
    using Handler = hypervec::HeapBlockResultHandler<Comparator>;
    Handler block(1, distances.data(), labels.data(), labels.size());
    Handler::SingleResultHandler results(block);
    results.begin(0);

    hypervec::GraphCandidateQueue candidates(3);
    candidates.push(0, distance(0));
    candidates.push(1, distance(1));
    candidates.push(2, distance(2));
    hypervec::VisitedTable visited(graph.NodeCount(), false);
    hypervec::GraphSearchStats stats;
    const hypervec::GraphSearcher searcher(graph);
    const int result_count = searcher.Search(
        distance, results, candidates, visited,
        hypervec::GraphSearchOptions{1, check_relative_distance, nullptr},
        &stats, 7);
    results.end();
    return std::pair(stats, result_count);
  };

  const auto [relative_stats, relative_count] = run_search(true);
  const auto [step_stats, step_count] = run_search(false);
  EXPECT_EQ(relative_count, 7);
  EXPECT_EQ(step_count, 7);
  EXPECT_EQ(relative_stats.expanded_nodes, 1U);
  EXPECT_EQ(step_stats.expanded_nodes, 2U);
}

TEST(NeighborPruner, HnswHeuristicPreservesDiversityAndOptionalFill) {
  ScalarDistanceComputer distance({0.0f, 1.0f, 1.1f, 10.0f});
  const std::array<hypervec::NeighborCandidate, 3> candidates = {
      hypervec::NeighborCandidate{1, 1.0f},
      hypervec::NeighborCandidate{2, 1.21f},
      hypervec::NeighborCandidate{3, 100.0f},
  };

  hypervec::GraphPruneStats stats;
  const hypervec::HnswHeuristicPruner sparse_pruner;
  const auto sparse = sparse_pruner.Prune(candidates, 2, distance, &stats);
  ASSERT_EQ(sparse.size(), 1U);
  EXPECT_EQ(sparse[0].id, 1);
  EXPECT_EQ(stats.candidates_examined, 3U);
  EXPECT_EQ(stats.distance_computations, 2U);

  const hypervec::HnswHeuristicPruner full_pruner(true);
  const auto full = full_pruner.Prune(candidates, 2, distance);
  ASSERT_EQ(full.size(), 2U);
  EXPECT_EQ(full[0].id, 1);
  EXPECT_EQ(full[1].id, 2);
}

TEST(HnswGraphLayerView, ExposesLevelZeroWithoutCopyingNeighbors) {
  constexpr hypervec::idx_t kDimension = 4;
  constexpr hypervec::idx_t kCount = 32;
  const std::vector<float> vectors = MakeVectors(kCount, kDimension);
  hypervec::IndexHNSWFlat index(kDimension, 4);
  index.Add(kCount, vectors.data());

  const hypervec::HnswGraphLayerView graph(index.hnsw);
  ASSERT_EQ(graph.NodeCount(), static_cast<size_t>(kCount));
  EXPECT_EQ(graph.MaxDegree(), static_cast<size_t>(index.hnsw.NbNeighbors(0)));
  for (hypervec::GraphId node = 0; node < kCount; ++node) {
    size_t begin = 0;
    size_t end = 0;
    index.hnsw.NeighborRange(node, 0, &begin, &end);
    const auto neighbors = graph.Neighbors(node);
    EXPECT_EQ(neighbors.data(), index.hnsw.neighbors.data() + begin);
    ASSERT_LE(neighbors.size(), end - begin);
    for (size_t offset = 0; offset < neighbors.size(); ++offset) {
      EXPECT_EQ(neighbors[offset], index.hnsw.neighbors[begin + offset]);
    }
  }

  const auto report = hypervec::ValidateGraph(graph, index.hnsw.entry_point);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, static_cast<size_t>(kCount));
}
