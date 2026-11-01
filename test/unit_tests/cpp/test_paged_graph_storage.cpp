/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_searcher.h>
#include <index/graph/paged_graph_storage.h>
#include <index/graph/visited_table.h>
#include <persistence/page_cache.h>
#include <persistence/random_access_io.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace {

using Adjacency = std::vector<std::vector<hypervec::GraphId>>;

std::vector<uint8_t> Encode(const Adjacency& adjacency, size_t max_degree,
                            size_t page_size) {
  const size_t record_size =
      sizeof(uint32_t) + max_degree * sizeof(hypervec::GraphId);
  const size_t records_per_page = page_size / record_size;
  const size_t page_count =
      adjacency.size() / records_per_page +
      (adjacency.size() % records_per_page != 0 ? 1U : 0U);
  std::vector<uint8_t> data(page_count * page_size, 0);
  for (size_t node = 0; node < adjacency.size(); ++node) {
    const size_t page = node / records_per_page;
    const size_t slot = node % records_per_page;
    uint8_t* record = data.data() + page * page_size + slot * record_size;
    const uint32_t degree = static_cast<uint32_t>(adjacency[node].size());
    std::memcpy(record, &degree, sizeof(degree));
    std::memcpy(record + sizeof(degree), adjacency[node].data(),
                adjacency[node].size() * sizeof(hypervec::GraphId));
  }
  return data;
}

std::shared_ptr<hypervec::PageCache> MakeCache(std::vector<uint8_t> data,
                                               size_t page_size,
                                               size_t capacity) {
  auto reader =
      std::make_shared<hypervec::VectorRandomAccessReader>(std::move(data));
  return std::make_shared<hypervec::PageCache>(reader, page_size, capacity);
}

std::vector<hypervec::GraphId> CopyNeighbors(
    const hypervec::GraphStorage& graph, hypervec::GraphId node) {
  const auto neighbors = graph.Neighbors(node);
  return {neighbors.begin(), neighbors.end()};
}

class IdDistanceComputer final : public hypervec::DistanceComputer {
 public:
  void SetQuery(const float* query) override {
    target_ = static_cast<hypervec::GraphId>(*query);
  }

  float operator()(hypervec::idx_t index) override {
    const auto difference = static_cast<float>(index - target_);
    return difference * difference;
  }

  float symmetric_dis(hypervec::idx_t lhs, hypervec::idx_t rhs) override {
    const auto difference = static_cast<float>(lhs - rhs);
    return difference * difference;
  }

 private:
  hypervec::GraphId target_ = 0;
};

void StoreUint32(std::vector<uint8_t>* data, size_t offset, uint32_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void StoreGraphId(std::vector<uint8_t>* data, size_t offset,
                  hypervec::GraphId value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

}  // namespace

TEST(PagedGraphStorage, MapsFixedRecordsAcrossPagesAndPinsEvictedData) {
  const Adjacency adjacency = {{1, 2}, {2}, {3, 4}, {4}, {0}};
  constexpr size_t kMaxDegree = 2;
  constexpr size_t kPageSize = 32;
  const auto cache =
      MakeCache(Encode(adjacency, kMaxDegree, kPageSize), kPageSize, 1);
  const hypervec::PagedGraphStorage graph(cache, adjacency.size(), kMaxDegree);

  const hypervec::GraphNeighborList pinned = graph.Neighbors(0);
  EXPECT_EQ(CopyNeighbors(graph, 1), adjacency[1]);
  EXPECT_EQ(CopyNeighbors(graph, 2), adjacency[2]);
  EXPECT_EQ(CopyNeighbors(graph, 4), adjacency[4]);

  EXPECT_EQ(std::vector<hypervec::GraphId>(pinned.begin(), pinned.end()),
            adjacency[0]);
  EXPECT_EQ(graph.RecordSize(), 12U);
  EXPECT_EQ(graph.RecordsPerPage(), 2U);
  EXPECT_EQ(graph.StorageSize(), 96U);
  EXPECT_EQ(cache->CachedPages(), 1U);
  const auto stats = cache->Stats();
  EXPECT_EQ(stats.cache_hits, 1U);
  EXPECT_EQ(stats.cache_misses, 3U);
  EXPECT_EQ(stats.evictions, 2U);
}

TEST(PagedGraphStorage, ComposesWithTheSharedGraphSearcher) {
  const Adjacency adjacency = {{1}, {0, 2}, {1, 3}, {2}};
  const auto cache = MakeCache(Encode(adjacency, 2, 32), 32, 2);
  const hypervec::PagedGraphStorage graph(cache, adjacency.size(), 2);
  hypervec::GraphSearcher searcher(graph);
  hypervec::VisitedTable visited(graph.NodeCount(), false);
  IdDistanceComputer distance;
  const float target = 3.0F;
  distance.SetQuery(&target);
  const std::array<hypervec::GraphId, 1> entry = {0};

  const auto results = searcher.Search(
      distance, entry, hypervec::GraphSearchOptions{4, false, nullptr},
      &visited);

  ASSERT_EQ(results.size(), 4U);
  EXPECT_EQ(results[0].id, 3);
  EXPECT_EQ(results[1].id, 2);
  EXPECT_EQ(results[2].id, 1);
  EXPECT_EQ(results[3].id, 0);
}

TEST(PagedGraphStorage, RejectsConfigurationAndStorageSizeMismatch) {
  const auto empty_cache = MakeCache({}, 32, 1);
  EXPECT_THROW(hypervec::PagedGraphStorage(nullptr, 0, 2),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::PagedGraphStorage(empty_cache, 0, 0),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::PagedGraphStorage(empty_cache, 1, 8),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::PagedGraphStorage(empty_cache, 1, 2),
               hypervec::HypervecException);

  const auto trailing_cache = MakeCache(std::vector<uint8_t>(64), 32, 1);
  EXPECT_THROW(hypervec::PagedGraphStorage(trailing_cache, 1, 2),
               hypervec::HypervecException);

  const hypervec::PagedGraphStorage empty(empty_cache, 0, 2);
  EXPECT_EQ(empty.NodeCount(), 0U);
  EXPECT_EQ(empty.StorageSize(), 0U);
}

TEST(PagedGraphStorage, RejectsInvalidNodesAndCorruptedRecords) {
  constexpr size_t kPageSize = 32;
  constexpr size_t kMaxDegree = 2;
  const Adjacency valid = {{1}, {0}};

  auto excessive_degree = Encode(valid, kMaxDegree, kPageSize);
  StoreUint32(&excessive_degree, 0, 3);
  hypervec::PagedGraphStorage excessive(
      MakeCache(std::move(excessive_degree), kPageSize, 1), valid.size(),
      kMaxDegree);
  EXPECT_THROW(excessive.Neighbors(0), hypervec::HypervecException);

  auto outside_id = Encode(valid, kMaxDegree, kPageSize);
  StoreGraphId(&outside_id, sizeof(uint32_t), 2);
  hypervec::PagedGraphStorage outside(
      MakeCache(std::move(outside_id), kPageSize, 1), valid.size(), kMaxDegree);
  EXPECT_THROW(outside.Neighbors(0), hypervec::HypervecException);

  auto self_loop = Encode(valid, kMaxDegree, kPageSize);
  StoreGraphId(&self_loop, sizeof(uint32_t), 0);
  hypervec::PagedGraphStorage self(
      MakeCache(std::move(self_loop), kPageSize, 1), valid.size(), kMaxDegree);
  EXPECT_THROW(self.Neighbors(0), hypervec::HypervecException);

  const Adjacency duplicate_source = {{1, 1}, {0}};
  hypervec::PagedGraphStorage duplicate(
      MakeCache(Encode(duplicate_source, kMaxDegree, kPageSize), kPageSize, 1),
      duplicate_source.size(), kMaxDegree);
  EXPECT_THROW(duplicate.Neighbors(0), hypervec::HypervecException);

  const hypervec::PagedGraphStorage graph(
      MakeCache(Encode(valid, kMaxDegree, kPageSize), kPageSize, 1),
      valid.size(), kMaxDegree);
  EXPECT_THROW(graph.Neighbors(-1), hypervec::HypervecException);
  EXPECT_THROW(graph.Neighbors(2), hypervec::HypervecException);
  graph.Prefetch(-1);
  graph.Prefetch(0);
  graph.Prefetch(2);
}
