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
