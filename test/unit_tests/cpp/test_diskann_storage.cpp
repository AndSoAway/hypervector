/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/diskann/diskann_storage.h>
#include <index/graph/graph_storage.h>
#include <index/graph/paged_graph_storage.h>
#include <persistence/io.h>
#include <persistence/page_cache.h>
#include <persistence/random_access_io.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

class ShortWriter final : public hypervec::IOWriter {
 public:
  explicit ShortWriter(size_t limit) : limit_(limit) {}

  size_t operator()(const void* pointer, size_t size, size_t count) override {
    if (size != 1 || count == 0) {
      return 0;
    }
    const size_t written = std::min(limit_, count);
    const auto* bytes = static_cast<const uint8_t*>(pointer);
    data.insert(data.end(), bytes, bytes + written);
    return written;
  }

  std::vector<uint8_t> data;

 private:
  size_t limit_;
};

class StalledWriter final : public hypervec::IOWriter {
 public:
  size_t operator()(const void*, size_t, size_t) override { return 0; }
};

class MalformedGraph final : public hypervec::GraphStorage {
 public:
  size_t NodeCount() const noexcept override { return 2; }
  size_t MaxDegree() const noexcept override { return 2; }
  hypervec::GraphNeighborList Neighbors(hypervec::GraphId node) const override {
    const std::vector<hypervec::GraphId>& neighbors =
        adjacency_[static_cast<size_t>(node)];
    return hypervec::GraphNeighborList(hypervec::GraphNeighborView(neighbors));
  }

 private:
  std::array<std::vector<hypervec::GraphId>, 2> adjacency_ = {
      std::vector<hypervec::GraphId>{1, 1}, {0}};
};

hypervec::MutableBoundedGraph MakeGraph() {
  hypervec::MutableBoundedGraph graph(5, 2);
  const std::array<std::vector<hypervec::GraphId>, 5> adjacency = {
      std::vector<hypervec::GraphId>{1, 2}, {2}, {3, 4}, {4}, {0}};
  for (size_t node = 0; node < adjacency.size(); ++node) {
    graph.SetNeighbors(static_cast<hypervec::GraphId>(node), adjacency[node]);
  }
  return graph;
}

std::vector<float> MakeVectors(size_t count, size_t dimension) {
  std::vector<float> vectors(count * dimension);
  for (size_t index = 0; index < vectors.size(); ++index) {
    vectors[index] = static_cast<float>(index) + 0.25F;
  }
  return vectors;
}

std::vector<hypervec::GraphId> CopyNeighbors(
    const hypervec::GraphStorage& graph, hypervec::GraphId node) {
  const hypervec::GraphNeighborList neighbors = graph.Neighbors(node);
  return {neighbors.begin(), neighbors.end()};
}

}  // namespace

TEST(DiskAnnStorage, ComputesPageAlignedNodeLocations) {
  const hypervec::DiskAnnNodeLayout layout(5, 3, 2, 64);

  EXPECT_EQ(layout.VectorBytes(), 12U);
  EXPECT_EQ(layout.DegreeOffset(), 12U);
  EXPECT_EQ(layout.RecordSize(), 24U);
  EXPECT_EQ(layout.RecordsPerPage(), 2U);
  EXPECT_EQ(layout.PageCount(), 3U);
  EXPECT_EQ(layout.StorageSize(), 192U);
  EXPECT_EQ(layout.Locate(0).page_id, 0U);
  EXPECT_EQ(layout.Locate(1).offset, 24U);
  EXPECT_EQ(layout.Locate(2).page_id, 1U);
  EXPECT_EQ(layout.Locate(4).page_id, 2U);
  EXPECT_THROW(layout.Locate(-1), hypervec::HypervecException);
  EXPECT_THROW(layout.Locate(5), hypervec::HypervecException);
}

TEST(DiskAnnStorage, WritesAndReadsVectorsAndGraphThroughOnePageCache) {
  const hypervec::MutableBoundedGraph source = MakeGraph();
  const std::vector<float> vectors = MakeVectors(source.NodeCount(), 3);
  const hypervec::DiskAnnNodeLayout layout(source.NodeCount(), 3, 2, 64);
  ShortWriter writer(7);

  hypervec::WriteDiskAnnNodes(layout, vectors.data(), source, &writer);

  ASSERT_EQ(writer.data.size(), layout.StorageSize());
  auto reader = std::make_shared<hypervec::VectorRandomAccessReader>(
      std::move(writer.data));
  auto cache = std::make_shared<hypervec::PageCache>(reader, 64, 1);
  const hypervec::PagedGraphStorage graph(
      cache, source.NodeCount(), source.MaxDegree(), layout.RecordSize(),
      layout.DegreeOffset());
  const hypervec::PagedVectorStorage paged_vectors(cache, layout);

  const hypervec::DiskAnnVectorView pinned = paged_vectors.Vector(0);
  EXPECT_EQ(CopyNeighbors(graph, 0), CopyNeighbors(source, 0));
  EXPECT_EQ(CopyNeighbors(graph, 2), CopyNeighbors(source, 2));
  EXPECT_EQ(CopyNeighbors(graph, 4), CopyNeighbors(source, 4));
  for (size_t node = 0; node < source.NodeCount(); ++node) {
    const hypervec::DiskAnnVectorView vector =
        paged_vectors.Vector(static_cast<hypervec::GraphId>(node));
    EXPECT_TRUE(std::equal(vector.begin(), vector.end(),
                           vectors.begin() + node * layout.Dimension()));
  }
  EXPECT_TRUE(std::equal(pinned.begin(), pinned.end(), vectors.begin()));
  EXPECT_GT(cache->Stats().cache_hits, 0U);
  EXPECT_GT(cache->Stats().evictions, 0U);
}

TEST(DiskAnnStorage, ZeroFillsPageAndNeighborPadding) {
  hypervec::MutableBoundedGraph graph(1, 2);
  graph.SetNeighbors(0, {});
  const std::array<float, 1> vector = {2.5F};
  const hypervec::DiskAnnNodeLayout layout(1, 1, 2, 32);
  hypervec::VectorIOWriter writer;

  hypervec::WriteDiskAnnNodes(layout, vector.data(), graph, &writer);

  ASSERT_EQ(writer.data.size(), 32U);
  uint32_t degree = 1;
  std::memcpy(&degree, writer.data.data() + layout.DegreeOffset(),
              sizeof(degree));
  EXPECT_EQ(degree, 0U);
  EXPECT_TRUE(std::all_of(
      writer.data.begin() + layout.DegreeOffset() + sizeof(uint32_t),
      writer.data.end(), [](uint8_t value) { return value == 0; }));
}

TEST(DiskAnnStorage, RejectsInvalidLayoutsAndBackingData) {
  EXPECT_THROW(hypervec::DiskAnnNodeLayout(1, 0, 2, 64),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::DiskAnnNodeLayout(1, 3, 0, 64),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::DiskAnnNodeLayout(1, 3, 2, 0),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::DiskAnnNodeLayout(1, 16, 2, 64),
               hypervec::HypervecException);

  const hypervec::DiskAnnNodeLayout layout(1, 3, 2, 64);
  auto wrong_page_reader = std::make_shared<hypervec::VectorRandomAccessReader>(
      std::vector<uint8_t>(64));
  auto wrong_page_cache =
      std::make_shared<hypervec::PageCache>(wrong_page_reader, 32, 1);
  EXPECT_THROW(hypervec::PagedVectorStorage(wrong_page_cache, layout),
               hypervec::HypervecException);

  auto valid_reader = std::make_shared<hypervec::VectorRandomAccessReader>(
      std::vector<uint8_t>(64));
  auto valid_cache = std::make_shared<hypervec::PageCache>(valid_reader, 64, 1);
  EXPECT_THROW(hypervec::PagedGraphStorage(
                   valid_cache, 1, 2, layout.RecordSize(), layout.RecordSize()),
               hypervec::HypervecException);
  EXPECT_THROW(
      hypervec::PagedGraphStorage(valid_cache, 1, 2, 25, layout.DegreeOffset()),
      hypervec::HypervecException);

  auto short_reader = std::make_shared<hypervec::VectorRandomAccessReader>(
      std::vector<uint8_t>(63));
  auto short_cache = std::make_shared<hypervec::PageCache>(short_reader, 64, 1);
  EXPECT_THROW(hypervec::PagedVectorStorage(short_cache, layout),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::PagedVectorStorage(nullptr, layout),
               hypervec::HypervecException);

  const hypervec::PagedVectorStorage vectors(valid_cache, layout);
  EXPECT_THROW(vectors.Vector(-1), hypervec::HypervecException);
  EXPECT_THROW(vectors.Vector(1), hypervec::HypervecException);
  vectors.Prefetch(-1);
  vectors.Prefetch(0);
  vectors.Prefetch(1);
}

TEST(DiskAnnStorage, ValidatesInputsBeforeWriting) {
  const hypervec::MutableBoundedGraph graph = MakeGraph();
  const std::vector<float> vectors = MakeVectors(graph.NodeCount(), 3);
  const hypervec::DiskAnnNodeLayout layout(graph.NodeCount(), 3, 2, 64);
  hypervec::VectorIOWriter writer;

  EXPECT_THROW(
      hypervec::WriteDiskAnnNodes(layout, vectors.data(), graph, nullptr),
      hypervec::HypervecException);
  EXPECT_THROW(hypervec::WriteDiskAnnNodes(layout, nullptr, graph, &writer),
               hypervec::HypervecException);
  hypervec::MutableBoundedGraph short_graph(1, 2);
  EXPECT_THROW(
      hypervec::WriteDiskAnnNodes(layout, vectors.data(), short_graph, &writer),
      hypervec::HypervecException);
  EXPECT_TRUE(writer.data.empty());

  const hypervec::DiskAnnNodeLayout malformed_layout(2, 3, 2, 64);
  const MalformedGraph malformed;
  EXPECT_THROW(hypervec::WriteDiskAnnNodes(malformed_layout, vectors.data(),
                                           malformed, &writer),
               hypervec::HypervecException);
  EXPECT_TRUE(writer.data.empty());

  StalledWriter stalled;
  EXPECT_THROW(
      hypervec::WriteDiskAnnNodes(layout, vectors.data(), graph, &stalled),
      hypervec::HypervecException);
}
