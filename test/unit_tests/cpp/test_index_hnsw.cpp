/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <utils/common/result_handler.h>
#include <utils/log/exception.h>
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

struct DummyResultHandler : hypervec::ResultHandler {
  bool AddResult(float, hypervec::idx_t) override { return false; }
};

struct WrongCountFlat : hypervec::IndexFlatL2 {
  explicit WrongCountFlat(hypervec::idx_t d) : IndexFlatL2(d) {}

  void Add(hypervec::idx_t n, const float* x) override {
    IndexFlatL2::Add(n, x);
    ++n_total;
  }
};

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

TEST(IndexHNSWCorrectness, RejectsStorageCountMismatch) {
  constexpr hypervec::idx_t d = 4;
  const auto data = RandomVectors(4, d, 1004);
  WrongCountFlat storage(d);
  hypervec::IndexHNSW index(&storage, 4);

  EXPECT_THROW(index.Add(4, data.data()), hypervec::HypervecException);
  EXPECT_TRUE(index.hnsw.levels.empty());
  EXPECT_EQ(index.hnsw.offsets.size(), 1);
}

TEST(IndexHNSWCorrectness, UnsupportedEntryPointsFailExplicitly) {
  constexpr hypervec::idx_t d = 4;
  const auto data = RandomVectors(8, d, 1005);
  hypervec::IndexHNSWFlat index(d, 4);
  index.Add(8, data.data());

  DummyResultHandler handler;
  EXPECT_THROW(index.Search1(data.data(), handler),
               hypervec::HypervecException);
  std::vector<hypervec::idx_t> permutation(8);
  EXPECT_THROW(index.PermuteEntries(permutation.data()),
               hypervec::HypervecException);
}
