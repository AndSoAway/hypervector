/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/diskann/diskann_searcher.h>
#include <index/graph/paged_graph_storage.h>
#include <persistence/io.h>
#include <persistence/page_cache.h>
#include <persistence/random_access_io.h>
#include <quantization/quantizer.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

class DiskSearchFixture {
 public:
  DiskSearchFixture(std::vector<float> vectors,
                    std::vector<std::vector<hypervec::GraphId>> adjacency,
                    size_t dimension, size_t max_degree, size_t page_size = 64,
                    size_t cache_pages = 1)
      : node_count_(adjacency.size()),
        layout_(node_count_, dimension, max_degree, page_size) {
    hypervec::MutableBoundedGraph source(node_count_, max_degree);
    for (size_t node = 0; node < adjacency.size(); ++node) {
      source.SetNeighbors(static_cast<hypervec::GraphId>(node),
                          adjacency[node]);
    }
    hypervec::VectorIOWriter writer;
    hypervec::WriteDiskAnnNodes(layout_, vectors.data(), source, &writer);
    reader_ = std::make_shared<hypervec::VectorRandomAccessReader>(
        std::move(writer.data));
    cache_ =
        std::make_shared<hypervec::PageCache>(reader_, page_size, cache_pages);
    graph_ = std::make_unique<hypervec::PagedGraphStorage>(
        cache_, node_count_, max_degree, layout_.RecordSize(),
        layout_.DegreeOffset());
    vectors_ = std::make_unique<hypervec::PagedVectorStorage>(cache_, layout_);
  }

  const hypervec::PagedGraphStorage& Graph() const { return *graph_; }
  const hypervec::PagedVectorStorage& Vectors() const { return *vectors_; }
  const std::shared_ptr<hypervec::PageCache>& Cache() const { return cache_; }

 private:
  size_t node_count_;
  hypervec::DiskAnnNodeLayout layout_;
  std::shared_ptr<hypervec::VectorRandomAccessReader> reader_;
  std::shared_ptr<hypervec::PageCache> cache_;
  std::unique_ptr<hypervec::PagedGraphStorage> graph_;
  std::unique_ptr<hypervec::PagedVectorStorage> vectors_;
};

hypervec::InMemoryCodeStore MakeScalarCodes(const std::vector<float>& values) {
  hypervec::InMemoryCodeStore codes(sizeof(float));
  codes.Append(static_cast<hypervec::idx_t>(values.size()),
               reinterpret_cast<const uint8_t*>(values.data()));
  return codes;
}

std::vector<std::vector<hypervec::GraphId>> MakeCompleteGraph(size_t count) {
  std::vector<std::vector<hypervec::GraphId>> adjacency(count);
  for (size_t node = 0; node < count; ++node) {
    for (size_t neighbor = 0; neighbor < count; ++neighbor) {
      if (node != neighbor) {
        adjacency[node].push_back(static_cast<hypervec::GraphId>(neighbor));
      }
    }
  }
  return adjacency;
}

}  // namespace

TEST(DiskAnnSearcher, ReranksApproximateCandidatesWithPagedRawVectors) {
  const std::vector<float> raw = {10.0F, 0.0F, 2.0F, 4.0F};
  DiskSearchFixture fixture(raw, MakeCompleteGraph(raw.size()), 1, 3);
  const hypervec::InMemoryCodeStore codes =
      MakeScalarCodes({0.0F, 3.0F, 2.0F, 1.0F});
  const hypervec::FlatQuantizer quantizer(1, hypervec::kMetricL2);
  const hypervec::DiskAnnSearcher searcher(fixture.Graph(), fixture.Vectors(),
                                           quantizer, codes.View(), 0);
  const float query = 0.0F;
  hypervec::DiskAnnSearchStats stats;

  const auto results = searcher.Search(
      &query, 3, hypervec::DiskAnnSearchOptions{4, false, nullptr}, &stats);

  ASSERT_EQ(results.size(), 3U);
  EXPECT_EQ(results[0].id, 1);
  EXPECT_FLOAT_EQ(results[0].distance, 0.0F);
  EXPECT_EQ(results[1].id, 2);
  EXPECT_FLOAT_EQ(results[1].distance, 4.0F);
  EXPECT_EQ(results[2].id, 3);
  EXPECT_FLOAT_EQ(results[2].distance, 16.0F);
  EXPECT_EQ(stats.graph.queries, 1U);
  EXPECT_EQ(stats.graph.visited_nodes, 4U);
  EXPECT_EQ(stats.exact_distance_computations, 4U);
  EXPECT_GT(fixture.Cache()->Stats().pages_loaded, 0U);
}

TEST(DiskAnnSearcher, FilteredNodesRemainAvailableForGraphNavigation) {
  const std::vector<std::vector<hypervec::GraphId>> chain = {{1}, {2}, {}};
  DiskSearchFixture fixture({10.0F, 5.0F, 0.0F}, chain, 1, 1);
  const hypervec::InMemoryCodeStore codes =
      MakeScalarCodes({10.0F, 5.0F, 0.0F});
  const hypervec::FlatQuantizer quantizer(1, hypervec::kMetricL2);
  const hypervec::DiskAnnSearcher searcher(fixture.Graph(), fixture.Vectors(),
                                           quantizer, codes.View(), 0);
  hypervec::IDSelectorRange selector(2, 3);
  const float query = 0.0F;

  const auto results = searcher.Search(
      &query, 1, hypervec::DiskAnnSearchOptions{1, true, &selector});

  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(results[0].id, 2);
  EXPECT_FLOAT_EQ(results[0].distance, 0.0F);
}

TEST(DiskAnnSearcher, ExactTiesUseNodeIdAsStableTieBreak) {
  DiskSearchFixture fixture({5.0F, -1.0F, 1.0F}, MakeCompleteGraph(3), 1, 2);
  const hypervec::InMemoryCodeStore codes = MakeScalarCodes({0.0F, 2.0F, 1.0F});
  const hypervec::FlatQuantizer quantizer(1, hypervec::kMetricL2);
  const hypervec::DiskAnnSearcher searcher(fixture.Graph(), fixture.Vectors(),
                                           quantizer, codes.View(), 0);
  const float query = 0.0F;

  const auto results = searcher.Search(
      &query, 2, hypervec::DiskAnnSearchOptions{3, false, nullptr});

  ASSERT_EQ(results.size(), 2U);
  EXPECT_EQ(results[0].id, 1);
  EXPECT_EQ(results[1].id, 2);
  EXPECT_FLOAT_EQ(results[0].distance, 1.0F);
  EXPECT_FLOAT_EQ(results[1].distance, 1.0F);
}

TEST(DiskAnnSearcher, SearchIsSafeForConcurrentCallers) {
  DiskSearchFixture fixture({3.0F, 2.0F, 1.0F, 0.0F}, MakeCompleteGraph(4), 1,
                            3);
  const hypervec::InMemoryCodeStore codes =
      MakeScalarCodes({3.0F, 2.0F, 1.0F, 0.0F});
  const hypervec::FlatQuantizer quantizer(1, hypervec::kMetricL2);
  const hypervec::DiskAnnSearcher searcher(fixture.Graph(), fixture.Vectors(),
                                           quantizer, codes.View(), 0);

  std::vector<std::future<hypervec::GraphId>> calls;
  for (size_t index = 0; index < 8; ++index) {
    calls.push_back(std::async(std::launch::async, [&searcher, index] {
      const float query = static_cast<float>(index % 4);
      return searcher
          .Search(&query, 1, hypervec::DiskAnnSearchOptions{4, false, nullptr})
          .front()
          .id;
    }));
  }
  for (size_t index = 0; index < calls.size(); ++index) {
    EXPECT_EQ(calls[index].get(),
              static_cast<hypervec::GraphId>(3 - index % 4));
  }
}

TEST(DiskAnnSearcher, ValidatesAlignedDependenciesAndOptions) {
  DiskSearchFixture fixture({0.0F, 1.0F}, MakeCompleteGraph(2), 1, 1);
  const hypervec::InMemoryCodeStore codes = MakeScalarCodes({0.0F, 1.0F});
  const hypervec::FlatQuantizer l2(1, hypervec::kMetricL2);
  const hypervec::FlatQuantizer inner_product(1, hypervec::kMetricInnerProduct);
  const std::array<uint8_t, 16> wrong_codes = {};

  EXPECT_THROW(hypervec::DiskAnnSearcher(fixture.Graph(), fixture.Vectors(),
                                         inner_product, codes.View(), 0),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::DiskAnnSearcher(
                   fixture.Graph(), fixture.Vectors(), l2,
                   hypervec::EncodedVectorView(wrong_codes.data(), 2, 8), 0),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::DiskAnnSearcher(
                   fixture.Graph(), fixture.Vectors(), l2,
                   hypervec::EncodedVectorView(wrong_codes.data(), 1, 4), 0),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::DiskAnnSearcher(fixture.Graph(), fixture.Vectors(), l2,
                                         codes.View(), 2),
               hypervec::HypervecException);

  const hypervec::DiskAnnSearcher searcher(fixture.Graph(), fixture.Vectors(),
                                           l2, codes.View(), 0);
  const float query = 0.0F;
  EXPECT_THROW(searcher.Search(nullptr, 1), hypervec::HypervecException);
  EXPECT_THROW(searcher.Search(&query, 0), hypervec::HypervecException);
  EXPECT_THROW(searcher.Search(
                   &query, 1, hypervec::DiskAnnSearchOptions{0, true, nullptr}),
               hypervec::HypervecException);
}

TEST(DiskAnnSearcher, RejectsNaNDistancesWithoutPublishingStats) {
  DiskSearchFixture fixture({0.0F, 1.0F}, MakeCompleteGraph(2), 1, 1);
  const hypervec::InMemoryCodeStore bad_codes =
      MakeScalarCodes({0.0F, (std::numeric_limits<float>::quiet_NaN)()});
  const hypervec::FlatQuantizer quantizer(1, hypervec::kMetricL2);
  const hypervec::DiskAnnSearcher searcher(fixture.Graph(), fixture.Vectors(),
                                           quantizer, bad_codes.View(), 0);
  const float query = 0.0F;
  hypervec::DiskAnnSearchStats stats;

  EXPECT_THROW(
      searcher.Search(
          &query, 2, hypervec::DiskAnnSearchOptions{2, false, nullptr}, &stats),
      hypervec::HypervecException);
  EXPECT_EQ(stats.graph.queries, 0U);
  EXPECT_EQ(stats.exact_distance_computations, 0U);

  hypervec::DiskAnnSearchStats first;
  first.graph.queries = 2;
  first.exact_distance_computations = 3;
  hypervec::DiskAnnSearchStats second;
  second.graph.queries = 4;
  second.exact_distance_computations = 5;
  first.Combine(second);
  EXPECT_EQ(first.graph.queries, 6U);
  EXPECT_EQ(first.exact_distance_computations, 8U);
  first.Reset();
  EXPECT_EQ(first.graph.queries, 0U);
  EXPECT_EQ(first.exact_distance_computations, 0U);
}
