/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_storage.h>
#include <utils/log/exception.h>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

std::vector<hypervec::GraphId> CopyNeighbors(
    const hypervec::GraphStorage& graph, hypervec::GraphId node) {
  const auto neighbors = graph.Neighbors(node);
  return {neighbors.begin(), neighbors.end()};
}

}  // namespace

TEST(GraphStorage, LayoutsPreserveTheSameAdjacency) {
  hypervec::MutableBoundedGraph mutable_graph(5, 3);
  const std::array<hypervec::GraphId, 2> zero = {1, 2};
  const std::array<hypervec::GraphId, 2> one = {2, 3};
  const std::array<hypervec::GraphId, 1> two = {4};
  const std::array<hypervec::GraphId, 1> three = {0};
  mutable_graph.SetNeighbors(0, zero);
  mutable_graph.SetNeighbors(1, one);
  mutable_graph.SetNeighbors(2, two);
  mutable_graph.SetNeighbors(3, three);
  mutable_graph.SetNeighbors(4, {});

  EXPECT_FALSE(mutable_graph.AddNeighbor(0, 1));
  EXPECT_TRUE(mutable_graph.AddNeighbor(0, 3));
  EXPECT_TRUE(mutable_graph.RemoveNeighbor(0, 2));
  EXPECT_FALSE(mutable_graph.RemoveNeighbor(0, 4));

  hypervec::FixedDegreeGraph fixed_graph(5, 3);
  for (hypervec::GraphId node = 0; node < 5; ++node) {
    fixed_graph.SetNeighbors(node, mutable_graph.Neighbors(node));
  }
  const hypervec::CsrGraph csr_graph(fixed_graph);

  EXPECT_EQ(fixed_graph.Data().size(), 15U);
  EXPECT_EQ(csr_graph.Offsets().size(), 6U);
  for (hypervec::GraphId node = 0; node < 5; ++node) {
    const auto expected = CopyNeighbors(mutable_graph, node);
    EXPECT_EQ(CopyNeighbors(fixed_graph, node), expected);
    EXPECT_EQ(CopyNeighbors(csr_graph, node), expected);
  }
}

TEST(GraphStorage, RejectsInvalidEdgesWithoutChangingTheNode) {
  hypervec::MutableBoundedGraph mutable_graph(4, 2);
  const std::array<hypervec::GraphId, 1> initial = {1};
  mutable_graph.SetNeighbors(0, initial);

  const std::array<hypervec::GraphId, 1> self_loop = {0};
  const std::array<hypervec::GraphId, 2> duplicates = {1, 1};
  const std::array<hypervec::GraphId, 1> outside = {4};
  const std::array<hypervec::GraphId, 3> too_many = {1, 2, 3};
  EXPECT_THROW(mutable_graph.SetNeighbors(0, self_loop),
               hypervec::HypervecException);
  EXPECT_THROW(mutable_graph.SetNeighbors(0, duplicates),
               hypervec::HypervecException);
  EXPECT_THROW(mutable_graph.SetNeighbors(0, outside),
               hypervec::HypervecException);
  EXPECT_THROW(mutable_graph.SetNeighbors(0, too_many),
               hypervec::HypervecException);
  EXPECT_EQ(CopyNeighbors(mutable_graph, 0),
            std::vector<hypervec::GraphId>({1}));

  EXPECT_THROW(mutable_graph.AddNeighbor(0, 0), hypervec::HypervecException);
  EXPECT_THROW(mutable_graph.AddNeighbor(-1, 1), hypervec::HypervecException);
  EXPECT_THROW(mutable_graph.Neighbors(4), hypervec::HypervecException);
}

TEST(GraphStorage, ResizeRejectsDanglingEdgesBeforeMutation) {
  hypervec::MutableBoundedGraph mutable_graph(4, 2);
  const std::array<hypervec::GraphId, 1> edge_to_removed_node = {3};
  mutable_graph.SetNeighbors(0, edge_to_removed_node);
  EXPECT_THROW(mutable_graph.Resize(3), hypervec::HypervecException);
  EXPECT_EQ(mutable_graph.NodeCount(), 4U);

  hypervec::FixedDegreeGraph fixed_graph(mutable_graph);
  EXPECT_THROW(fixed_graph.Resize(3), hypervec::HypervecException);
  EXPECT_EQ(fixed_graph.NodeCount(), 4U);

  mutable_graph.SetNeighbors(0, {});
  mutable_graph.Resize(3);
  EXPECT_EQ(mutable_graph.NodeCount(), 3U);
}

TEST(GraphStorage, CsrValidatesOffsetsAndEdges) {
  EXPECT_THROW(hypervec::CsrGraph({}, {}), hypervec::HypervecException);
  EXPECT_THROW(hypervec::CsrGraph({1}, {}), hypervec::HypervecException);
  EXPECT_THROW(hypervec::CsrGraph({0, 2, 1}, {1}), hypervec::HypervecException);
  EXPECT_THROW(hypervec::CsrGraph({0, 1}, {1}), hypervec::HypervecException);
  EXPECT_THROW(hypervec::CsrGraph({0, 1, 2}, {1, 1}),
               hypervec::HypervecException);

  const hypervec::CsrGraph empty({0, 0, 0}, {});
  EXPECT_EQ(empty.NodeCount(), 2U);
  EXPECT_EQ(empty.MaxDegree(), 0U);
  EXPECT_TRUE(empty.Neighbors(0).empty());
  empty.Prefetch(-1);
  empty.Prefetch(0);
  empty.Prefetch(2);
}

TEST(GraphStorage, ConstructorsRejectInvalidDegree) {
  EXPECT_THROW(hypervec::MutableBoundedGraph(0), hypervec::HypervecException);
  EXPECT_THROW(hypervec::FixedDegreeGraph(0), hypervec::HypervecException);
  const size_t wider_than_degree =
      static_cast<size_t>((std::numeric_limits<uint32_t>::max)()) + 1;
  EXPECT_THROW((void)hypervec::FixedDegreeGraph(wider_than_degree),
               hypervec::HypervecException);
}
