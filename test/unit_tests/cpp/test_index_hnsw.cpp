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
#include <utils/selector/id_selector.h>
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

TEST(IndexHNSWCorrectness, Search1UsesGraphAndUnsupportedPermuteFails) {
  constexpr hypervec::idx_t d = 4;
  const auto data = RandomVectors(8, d, 1005);
  hypervec::IndexHNSWFlat index(d, 4);
  index.Add(8, data.data());

  constexpr hypervec::idx_t k = 3;
  std::vector<float> expected_distances(k);
  std::vector<hypervec::idx_t> expected_labels(k);
  index.Search(1, data.data(), k, expected_distances.data(),
               expected_labels.data());

  std::vector<float> actual_distances(k);
  std::vector<hypervec::idx_t> actual_labels(k);
  using Handler = hypervec::HeapBlockResultHandler<hypervec::HNSW::C>;
  Handler block(1, actual_distances.data(), actual_labels.data(), k);
  Handler::SingleResultHandler handler(block);
  handler.begin(0);
  index.Search1(data.data(), handler);
  handler.end();
  EXPECT_EQ(actual_labels, expected_labels);
  EXPECT_EQ(actual_distances, expected_distances);

  hypervec::IDSelectorRange selector(1, index.n_total);
  hypervec::SearchParametersHNSW params;
  params.sel = &selector;
  index.Search(1, data.data(), k, expected_distances.data(),
               expected_labels.data(), &params);
  handler.begin(0);
  index.Search1(data.data(), handler, &params);
  handler.end();
  EXPECT_EQ(actual_labels, expected_labels);
  EXPECT_EQ(actual_distances, expected_distances);
  for (hypervec::idx_t label : actual_labels) {
    EXPECT_GE(label, 1);
  }

  std::vector<hypervec::idx_t> permutation(8);
  EXPECT_THROW(index.PermuteEntries(permutation.data()),
               hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, RejectsNonPositiveEfSearch) {
  constexpr hypervec::idx_t d = 4;
  const auto data = RandomVectors(8, d, 1006);
  hypervec::IndexHNSWFlat index(d, 4);
  index.Add(8, data.data());

  constexpr hypervec::idx_t k = 3;
  std::vector<float> distances(k);
  std::vector<hypervec::idx_t> labels(k);
  index.hnsw.ef_search = 0;
  EXPECT_THROW(index.Search(1, data.data(), k, distances.data(), labels.data()),
               hypervec::HypervecException);

  index.hnsw.ef_search = 16;
  hypervec::SearchParametersHNSW params;
  params.ef_search = -1;
  EXPECT_THROW(
      index.Search(1, data.data(), k, distances.data(), labels.data(), &params),
      hypervec::HypervecException);
}
