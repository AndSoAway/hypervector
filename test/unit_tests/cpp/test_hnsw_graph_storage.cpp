/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_searcher.h>
#include <index/hnsw/hnsw_graph_storage.h>
#include <utils/common/result_handler.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <utility>
#include <vector>

namespace {

class LineDistance final : public hypervec::DistanceComputer {
 public:
  explicit LineDistance(std::vector<float> values)
      : values_(std::move(values)) {}

  void SetQuery(const float* query) override { query_ = *query; }

  float operator()(hypervec::idx_t index) override {
    return std::abs(query_ - values_.at(static_cast<size_t>(index)));
  }

  float symmetric_dis(hypervec::idx_t lhs, hypervec::idx_t rhs) override {
    return std::abs(values_.at(static_cast<size_t>(lhs)) -
                    values_.at(static_cast<size_t>(rhs)));
  }

 private:
  std::vector<float> values_;
  float query_ = 0.0F;
};

hypervec::HNSW MakeLayeredGraph() {
  hypervec::HNSW hnsw(2);
  hnsw.levels = {2, 1, 2, 1};
  hnsw.PrepareLevelTab(hnsw.levels.size(), true);

  const auto set_neighbors = [&](hypervec::GraphId node, int layer,
                                 std::initializer_list<hypervec::GraphId> ids) {
    size_t begin = 0;
    size_t end = 0;
    hnsw.NeighborRange(node, layer, &begin, &end);
    ASSERT_LE(ids.size(), end - begin);
    std::copy(ids.begin(), ids.end(), hnsw.neighbors.begin() + begin);
  };
  set_neighbors(0, 0, {1});
  set_neighbors(1, 0, {0, 2});
  set_neighbors(2, 0, {1, 3});
  set_neighbors(3, 0, {2});
  set_neighbors(0, 1, {2});
  set_neighbors(2, 1, {0});
  return hnsw;
}

}  // namespace

TEST(HNSWGraphStorage, ExposesSelectedLayersWithoutCopying) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  hypervec::HNSWGraphStorage level0(hnsw);
  hypervec::HNSWGraphStorage level1(hnsw, 1);

  EXPECT_EQ(level0.NodeCount(), 4U);
  EXPECT_EQ(level0.MaxDegree(), 4U);
  EXPECT_EQ(level1.MaxDegree(), 2U);
  const hypervec::GraphNeighborList node1_level0 = level0.Neighbors(1);
  EXPECT_EQ(
      std::vector<hypervec::GraphId>(node1_level0.begin(), node1_level0.end()),
      (std::vector<hypervec::GraphId>{0, 2}));
  EXPECT_TRUE(level1.Neighbors(1).empty());
  const hypervec::GraphNeighborList node0_level1 = level1.Neighbors(0);
  EXPECT_EQ(
      std::vector<hypervec::GraphId>(node0_level1.begin(), node0_level1.end()),
      (std::vector<hypervec::GraphId>{2}));

  size_t begin = 0;
  size_t end = 0;
  hnsw.NeighborRange(0, 1, &begin, &end);
  hnsw.neighbors[begin] = hypervec::kInvalidGraphId;
  EXPECT_TRUE(level1.Neighbors(0).empty());
}

TEST(HNSWGraphStorage, ComposesWithCommonGraphSearcher) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  hypervec::HNSWGraphStorage graph(hnsw);
  hypervec::GraphSearcher searcher(graph);
  LineDistance distance({0.0F, 2.0F, 5.0F, 9.0F});
  const float query = 4.0F;
  distance.SetQuery(&query);
  hypervec::VisitedTable visited(graph.NodeCount());
  hypervec::GraphSearchOptions options;
  options.ef_search = graph.NodeCount();
  options.check_relative_distance = false;
  const std::array<hypervec::GraphId, 1> entry_points = {0};

  const std::vector<hypervec::GraphSearchResult> results =
      searcher.Search(distance, entry_points, options, &visited);

  ASSERT_EQ(results.size(), 4U);
  EXPECT_EQ(results[0].id, 2);
  EXPECT_FLOAT_EQ(results[0].distance, 1.0F);
  EXPECT_EQ(results[1].id, 1);
  EXPECT_FLOAT_EQ(results[1].distance, 2.0F);
}

TEST(HNSWGraphStorage, NavigationBoundMatchesNativeUnboundedSearch) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  LineDistance distance({0.0F, 2.0F, 5.0F, 9.0F});
  const float query = 4.0F;
  distance.SetQuery(&query);
  constexpr size_t ef_search = 4;
  constexpr hypervec::GraphId entry = 0;
  const float entry_distance = distance(entry);

  hypervec::VisitedTable native_visited(hnsw.levels.size());
  hypervec::HNSWStats native_stats;
  auto native = hypervec::SearchFromCandidateUnbounded(
      hnsw, {entry_distance, entry}, distance, ef_search, &native_visited,
      native_stats);
  std::vector<hypervec::GraphSearchResult> native_results;
  while (!native.empty()) {
    native_results.push_back({native.top().second, native.top().first});
    native.pop();
  }
  std::sort(native_results.begin(), native_results.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.distance != rhs.distance ? lhs.distance < rhs.distance
                                                  : lhs.id < rhs.id;
            });

  const hypervec::HNSWGraphStorage graph(
      hnsw, 0, hypervec::HNSWGraphValidation::kOnAccess);
  const hypervec::GraphSearcher searcher(graph);
  const std::array<hypervec::GraphSearchSeed, 1> seeds = {
      hypervec::GraphSearchSeed{entry, entry_distance}};
  hypervec::VisitedTable common_visited(graph.NodeCount());
  hypervec::GraphSearchStats common_stats;
  const auto common_results = searcher.Search(
      distance, seeds,
      hypervec::GraphSearchOptions{
          ef_search, true, nullptr,
          hypervec::GraphSearchFrontierPolicy::kNavigationBound},
      &common_visited, &common_stats);

  ASSERT_EQ(common_results.size(), native_results.size());
  for (size_t result = 0; result < common_results.size(); ++result) {
    EXPECT_EQ(common_results[result].id, native_results[result].id);
    EXPECT_FLOAT_EQ(common_results[result].distance,
                    native_results[result].distance);
  }
  EXPECT_EQ(common_stats.distance_computations, native_stats.ndis);
  EXPECT_EQ(common_stats.expanded_nodes, native_stats.nhops);
  EXPECT_EQ(common_stats.exhausted_queries, native_stats.n2);
}

TEST(HNSWGraphStorage, CandidateBoundMatchesNativeBoundedSearch) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  LineDistance distance({0.0F, 2.0F, 5.0F, 9.0F});
  const float query = 4.0F;
  distance.SetQuery(&query);
  constexpr int ef_search = 2;
  constexpr int k = 2;
  constexpr hypervec::GraphId entry = 0;
  const float entry_distance = distance(entry);

  std::array<float, k> native_distances{};
  std::array<hypervec::idx_t, k> native_labels{};
  using ResultHandler = hypervec::HeapBlockResultHandler<hypervec::HNSW::C>;
  ResultHandler block(1, native_distances.data(), native_labels.data(), k);
  ResultHandler::SingleResultHandler result(block);
  hypervec::HNSW::MinimaxHeap native_candidates(ef_search);
  native_candidates.push(entry, entry_distance);
  hypervec::VisitedTable native_visited(hnsw.levels.size());
  hypervec::HNSWStats native_stats;
  hypervec::SearchParametersHNSW native_options;
  native_options.ef_search = ef_search;
  result.begin(0);
  hypervec::SearchFromCandidates(hnsw, distance, result, native_candidates,
                                 native_visited, native_stats, 0, 0,
                                 &native_options);
  result.end();

  const hypervec::HNSWGraphStorage graph(
      hnsw, 0, hypervec::HNSWGraphValidation::kOnAccess);
  const hypervec::GraphSearcher searcher(graph);
  const std::array<hypervec::GraphSearchSeed, 1> seeds = {
      hypervec::GraphSearchSeed{entry, entry_distance}};
  hypervec::VisitedTable common_visited(graph.NodeCount());
  hypervec::GraphSearchStats common_stats;
  const auto common_results = searcher.Search(
      distance, seeds,
      hypervec::GraphSearchOptions{
          ef_search, true, nullptr,
          hypervec::GraphSearchFrontierPolicy::kNavigationBound, 0, ef_search},
      &common_visited, &common_stats);

  ASSERT_EQ(common_results.size(), static_cast<size_t>(k));
  for (size_t index = 0; index < common_results.size(); ++index) {
    EXPECT_EQ(common_results[index].id, native_labels[index]);
    EXPECT_FLOAT_EQ(common_results[index].distance, native_distances[index]);
  }
  EXPECT_EQ(common_stats.distance_computations, native_stats.ndis);
  EXPECT_EQ(common_stats.expanded_nodes, native_stats.nhops);
  EXPECT_EQ(common_stats.exhausted_queries, native_stats.n2);
  EXPECT_LE(common_stats.peak_candidates, static_cast<size_t>(ef_search));
}

TEST(HNSWGraphStorage, DisabledRelativeCheckUsesRuntimeExpansionBudget) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  hnsw.entry_point = 0;
  hnsw.max_level = 0;
  LineDistance distance({0.0F, 2.0F, 5.0F, 9.0F});
  const float query = 4.0F;
  distance.SetQuery(&query);

  constexpr int k = 3;
  std::array<float, k> distances{};
  std::array<hypervec::idx_t, k> labels{};
  using ResultHandler = hypervec::HeapBlockResultHandler<hypervec::HNSW::C>;
  ResultHandler block(1, distances.data(), labels.data(), k);
  ResultHandler::SingleResultHandler result(block);
  hypervec::VisitedTable visited(hnsw.levels.size());
  hypervec::SearchParametersHNSW options;
  options.ef_search = 1;
  options.check_relative_distance = false;
  options.bounded_queue = true;

  result.begin(0);
  const hypervec::HNSWStats stats =
      hnsw.Search(distance, nullptr, result, visited, &options);
  result.end();

  EXPECT_EQ(stats.nhops, 2U);
}

TEST(HNSWGraphStorage, RuntimeEfControlsRelativeTerminationWhenKIsLarger) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  hnsw.entry_point = 0;
  hnsw.max_level = 0;
  LineDistance distance({0.0F, 2.0F, 5.0F, 9.0F});
  const float query = 0.0F;
  distance.SetQuery(&query);

  constexpr int k = 3;
  std::array<float, k> distances{};
  std::array<hypervec::idx_t, k> labels{};
  using ResultHandler = hypervec::HeapBlockResultHandler<hypervec::HNSW::C>;
  ResultHandler block(1, distances.data(), labels.data(), k);
  ResultHandler::SingleResultHandler result(block);
  hypervec::VisitedTable visited(hnsw.levels.size());
  hypervec::SearchParametersHNSW options;
  options.ef_search = 1;
  options.check_relative_distance = true;
  options.bounded_queue = true;

  result.begin(0);
  const hypervec::HNSWStats stats =
      hnsw.Search(distance, nullptr, result, visited, &options);
  result.end();

  EXPECT_EQ(stats.nhops, 1U);
  EXPECT_EQ(labels[0], 0);
  EXPECT_EQ(labels[1], 1);
}

TEST(HNSWGraphStorage, SharedLevel0SearchMatchesNativeTraversal) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  LineDistance distance({0.0F, 2.0F, 5.0F, 9.0F});
  const float query = 4.0F;
  distance.SetQuery(&query);
  constexpr int k = 2;
  constexpr int nprobe = 2;
  const std::array<hypervec::HNSW::storage_idx_t, nprobe> entries = {0, 3};
  const std::array<float, nprobe> entry_distances = {4.0F, 5.0F};
  hypervec::SearchParametersHNSW options;
  options.ef_search = 2;
  options.check_relative_distance = true;

  using ResultHandler = hypervec::HeapBlockResultHandler<hypervec::HNSW::C>;
  for (int search_type : {1, 2}) {
    std::array<float, k> common_distances{};
    std::array<hypervec::idx_t, k> common_labels{};
    ResultHandler common_block(1, common_distances.data(), common_labels.data(),
                               k);
    ResultHandler::SingleResultHandler common_result(common_block);
    hypervec::VisitedTable common_visited(hnsw.levels.size());
    hypervec::HNSWStats common_stats;
    common_result.begin(0);
    hnsw.SearchLevel0(distance, common_result, nprobe, entries.data(),
                      entry_distances.data(), search_type, common_stats,
                      common_visited, &options);
    common_result.end();

    std::array<float, k> native_distances{};
    std::array<hypervec::idx_t, k> native_labels{};
    ResultHandler native_block(1, native_distances.data(), native_labels.data(),
                               k);
    ResultHandler::SingleResultHandler native_result(native_block);
    hypervec::VisitedTable native_visited(hnsw.levels.size());
    hypervec::HNSWStats native_stats;
    native_result.begin(0);
    if (search_type == 1) {
      int result_count = 0;
      for (int probe = 0; probe < nprobe; ++probe) {
        if (native_visited.get(entries[probe])) {
          continue;
        }
        hypervec::HNSW::MinimaxHeap candidates(options.ef_search);
        candidates.push(entries[probe], entry_distances[probe]);
        result_count = hypervec::SearchFromCandidates(
            hnsw, distance, native_result, candidates, native_visited,
            native_stats, 0, result_count, &options);
        result_count = std::min(result_count, options.ef_search);
      }
    } else {
      hypervec::HNSW::MinimaxHeap candidates(nprobe);
      for (int probe = 0; probe < nprobe; ++probe) {
        candidates.push(entries[probe], entry_distances[probe]);
      }
      hypervec::SearchFromCandidates(hnsw, distance, native_result, candidates,
                                     native_visited, native_stats, 0, 0,
                                     &options);
    }
    native_result.end();

    EXPECT_EQ(common_labels, native_labels);
    EXPECT_EQ(common_distances, native_distances);
    EXPECT_EQ(common_stats.n1, native_stats.n1);
    EXPECT_EQ(common_stats.n2, native_stats.n2);
    EXPECT_EQ(common_stats.ndis, native_stats.ndis);
    EXPECT_EQ(common_stats.nhops, native_stats.nhops);
  }
}

TEST(HNSWGraphStorage, RejectsInvalidLayersAndMutatedLayouts) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  EXPECT_THROW((hypervec::HNSWGraphStorage(hnsw, -1)),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::HNSWGraphStorage(hnsw, 100)),
               hypervec::HypervecException);

  hypervec::HNSWGraphStorage graph(hnsw);
  hnsw.offsets.pop_back();
  EXPECT_THROW(graph.Neighbors(0), hypervec::HypervecException);
}

TEST(HNSWGraphStorage, OnAccessValidationAvoidsTheFullConstructionScan) {
  hypervec::HNSW hnsw = MakeLayeredGraph();
  hnsw.levels[3] = 0;

  EXPECT_THROW((hypervec::HNSWGraphStorage(hnsw)), hypervec::HypervecException);
  const hypervec::HNSWGraphStorage graph(
      hnsw, 0, hypervec::HNSWGraphValidation::kOnAccess);
  EXPECT_NO_THROW(graph.Neighbors(0));
  EXPECT_THROW(graph.Neighbors(3), hypervec::HypervecException);
}
