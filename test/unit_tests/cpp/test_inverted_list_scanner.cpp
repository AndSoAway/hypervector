/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/ivf/inverted_list_scanner.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

class ByteScanner final : public hypervec::InvertedListScanner {
 public:
  ByteScanner(hypervec::MetricType metric, size_t code_size)
      : InvertedListScanner(metric, code_size) {}

  void SetQuery(const float* query) override {
    query_offset_ = query == nullptr ? 0.0F : query[0];
  }

  void SetList(hypervec::idx_t list_no, float coarse_distance) override {
    list_offset_ = static_cast<float>(list_no) + coarse_distance;
  }

  float DistanceToCode(const uint8_t* code) const override {
    return static_cast<float>(code[0]) + query_offset_ + list_offset_;
  }

 private:
  float query_offset_ = 0.0F;
  float list_offset_ = 0.0F;
};

}  // namespace

TEST(InvertedListScanner, DistanceKnnUsesStrideAndSelector) {
  ByteScanner scanner(hypervec::kMetricL2, 2);
  const float query[] = {1.0F};
  scanner.SetQuery(query);
  scanner.SetList(1, 1.0F);

  const uint8_t codes[] = {5, 99, 1, 99, 3, 99};
  const hypervec::idx_t ids[] = {10, 11, 12};
  hypervec::IDSelectorRange selector(11, 13);
  float distances[2];
  hypervec::idx_t labels[2];
  hypervec::heap_heapify<hypervec::CMax<float, hypervec::idx_t>>(2, distances,
                                                                 labels);

  EXPECT_EQ(scanner.ScanKnn(codes, ids, 3, &selector, 2, distances, labels),
            2U);
  hypervec::heap_reorder<hypervec::CMax<float, hypervec::idx_t>>(2, distances,
                                                                 labels);
  EXPECT_FLOAT_EQ(distances[0], 4.0F);
  EXPECT_EQ(labels[0], 11);
  EXPECT_FLOAT_EQ(distances[1], 6.0F);
  EXPECT_EQ(labels[1], 12);
}

TEST(InvertedListScanner, SimilarityKnnKeepsLargestValues) {
  ByteScanner scanner(hypervec::kMetricInnerProduct, 1);
  scanner.SetQuery(nullptr);
  scanner.SetList(0, 0.0F);

  const uint8_t codes[] = {2, 9, 5};
  const hypervec::idx_t ids[] = {20, 21, 22};
  float distances[2];
  hypervec::idx_t labels[2];
  hypervec::heap_heapify<hypervec::CMin<float, hypervec::idx_t>>(2, distances,
                                                                 labels);

  EXPECT_EQ(scanner.ScanKnn(codes, ids, 3, nullptr, 2, distances, labels), 3U);
  hypervec::heap_reorder<hypervec::CMin<float, hypervec::idx_t>>(2, distances,
                                                                 labels);
  EXPECT_FLOAT_EQ(distances[0], 9.0F);
  EXPECT_EQ(labels[0], 21);
  EXPECT_FLOAT_EQ(distances[1], 5.0F);
  EXPECT_EQ(labels[1], 22);
}

TEST(InvertedListScanner, RangeComparisonIsMetricAwareAndInclusive) {
  const uint8_t codes[] = {2, 5, 8};
  const hypervec::idx_t ids[] = {30, 31, 32};

  ByteScanner distance_scanner(hypervec::kMetricL2, 1);
  distance_scanner.SetQuery(nullptr);
  distance_scanner.SetList(0, 0.0F);
  std::vector<std::pair<float, hypervec::idx_t>> distance_results;
  EXPECT_EQ(distance_scanner.ScanRange(codes, ids, 3, nullptr, 5.0F,
                                       &distance_results),
            3U);
  EXPECT_EQ(distance_results, (std::vector<std::pair<float, hypervec::idx_t>>{
                                  {2.0F, 30}, {5.0F, 31}}));

  ByteScanner similarity_scanner(hypervec::kMetricInnerProduct, 1);
  similarity_scanner.SetQuery(nullptr);
  similarity_scanner.SetList(0, 0.0F);
  std::vector<std::pair<float, hypervec::idx_t>> similarity_results;
  EXPECT_EQ(similarity_scanner.ScanRange(codes, ids, 3, nullptr, 5.0F,
                                         &similarity_results),
            3U);
  EXPECT_EQ(similarity_results, (std::vector<std::pair<float, hypervec::idx_t>>{
                                    {5.0F, 31}, {8.0F, 32}}));
}

TEST(InvertedListScanner, RejectsInvalidArgumentsAndOverflow) {
  EXPECT_THROW(ByteScanner(hypervec::kMetricL2, 0),
               hypervec::HypervecException);

  ByteScanner scanner(hypervec::kMetricL2, 2);
  const uint8_t code = 1;
  const hypervec::idx_t id = 1;
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  std::vector<std::pair<float, hypervec::idx_t>> results;

  EXPECT_THROW(scanner.ScanKnn(nullptr, &id, 1, nullptr, 1, &distance, &label),
               hypervec::HypervecException);
  EXPECT_THROW(
      scanner.ScanKnn(&code, nullptr, 1, nullptr, 1, &distance, &label),
      hypervec::HypervecException);
  EXPECT_THROW(scanner.ScanKnn(&code, &id, 1, nullptr, 0, &distance, &label),
               hypervec::HypervecException);
  EXPECT_THROW(scanner.ScanKnn(&code, &id, 1, nullptr, 1, nullptr, &label),
               hypervec::HypervecException);
  EXPECT_THROW(scanner.ScanRange(&code, &id, 1, nullptr, 1.0F, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(scanner.ScanRange(&code, &id,
                                 (std::numeric_limits<size_t>::max)() / 2 + 1,
                                 nullptr, 1.0F, &results),
               hypervec::HypervecException);

  EXPECT_EQ(scanner.ScanRange(nullptr, nullptr, 0, nullptr, 1.0F, &results),
            0U);
}
