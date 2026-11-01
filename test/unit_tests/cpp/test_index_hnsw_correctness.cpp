/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <utils/structures/random.h>

#include <cstdint>
#include <vector>

namespace {

std::vector<float> RandomVectors(hypervec::idx_t n, hypervec::idx_t d,
                                 int64_t seed) {
  hypervec::RandomGenerator rng(seed);
  std::vector<float> result(static_cast<size_t>(n) * d);
  for (float& value : result) {
    value = rng.rand_float();
  }
  return result;
}

void ExpectConsistentGraph(const hypervec::IndexHNSW& index) {
  ASSERT_EQ(index.hnsw.levels.size(), static_cast<size_t>(index.n_total));
  ASSERT_EQ(index.hnsw.offsets.size(), static_cast<size_t>(index.n_total) + 1);
  for (const auto neighbor : index.hnsw.neighbors) {
    EXPECT_TRUE(neighbor == -1 || (neighbor >= 0 && neighbor < index.n_total));
  }
}

void ExpectSearchLabelsValid(const hypervec::Index& index, const float* query) {
  constexpr hypervec::idx_t k = 5;
  std::vector<float> distances(k);
  std::vector<hypervec::idx_t> labels(k);
  index.Search(1, query, k, distances.data(), labels.data());
  for (const auto label : labels) {
    EXPECT_GE(label, 0);
    EXPECT_LT(label, index.n_total);
  }
}

}  // namespace

TEST(IndexHNSWCorrectness, RepeatedAddFlatKeepsGraphAligned) {
  constexpr hypervec::idx_t d = 8;
  constexpr hypervec::idx_t batch = 32;
  const auto data = RandomVectors(batch * 2, d, 1001);

  hypervec::IndexHNSWFlat index(d, 8);
  index.Add(batch, data.data());
  index.Add(batch, data.data() + batch * d);

  EXPECT_EQ(index.n_total, batch * 2);
  ExpectConsistentGraph(index);
  ExpectSearchLabelsValid(index, data.data());
}

TEST(IndexHNSWCorrectness, RepeatedAddPQKeepsGraphAligned) {
  constexpr hypervec::idx_t d = 8;
  constexpr hypervec::idx_t batch = 32;
  const auto data = RandomVectors(batch * 2, d, 1002);

  hypervec::IndexHNSWPQ index(d, 2, 4, 8);
  index.Train(batch * 2, data.data());
  index.Add(batch, data.data());
  index.Add(batch, data.data() + batch * d);

  EXPECT_EQ(index.n_total, batch * 2);
  ExpectConsistentGraph(index);
  ExpectSearchLabelsValid(index, data.data());
}

TEST(IndexHNSWCorrectness, RepeatedAddLVQKeepsGraphAligned) {
  constexpr hypervec::idx_t d = 8;
  constexpr hypervec::idx_t batch = 32;
  const auto data = RandomVectors(batch * 2, d, 1003);

  hypervec::IndexHNSWLVQ index(d, 4, 3, 8);
  index.Train(batch * 2, data.data());
  index.Add(batch, data.data());
  index.Add(batch, data.data() + batch * d);

  EXPECT_EQ(index.n_total, batch * 2);
  ExpectConsistentGraph(index);
  ExpectSearchLabelsValid(index, data.data());
}
