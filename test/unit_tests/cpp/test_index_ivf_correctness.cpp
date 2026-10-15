/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/ivf/index_ivf_flat.h>
#include <invlists/inverted_lists.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
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

class FailingArrayInvertedLists : public hypervec::ArrayInvertedLists {
 public:
  FailingArrayInvertedLists(size_t nlist, size_t code_size, int fail_on_call)
      : ArrayInvertedLists(nlist, code_size), fail_on_call_(fail_on_call) {}

  size_t add_entries(size_t list_no, size_t n_entry, const hypervec::idx_t* ids,
                     const uint8_t* codes) override {
    const size_t previous =
        ArrayInvertedLists::add_entries(list_no, n_entry, ids, codes);
    ++calls_;
    if (calls_ == fail_on_call_) {
      throw std::runtime_error("injected inverted-list failure");
    }
    return previous;
  }

 private:
  int fail_on_call_;
  int calls_ = 0;
};

void InstallFailingLists(hypervec::IndexIVF* index) {
  const size_t code_size = index->invlists->code_size;
  delete index->invlists;
  index->invlists = new FailingArrayInvertedLists(
      static_cast<size_t>(index->nlist), code_size, /*fail_on_call=*/2);
  index->own_invlists = true;
}

void ExpectCompressedAddRollback(hypervec::IndexIVF* index) {
  ASSERT_EQ(index->d, 4);
  ASSERT_EQ(index->nlist, 2);
  index->centroids = {
      0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 100.0f, 100.0f, 100.0f,
  };
  InstallFailingLists(index);

  const std::vector<float> vectors = {
      0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 100.0f, 100.0f, 100.0f,
  };
  EXPECT_THROW(index->Add(2, vectors.data()), std::runtime_error);
  EXPECT_EQ(index->n_total, 0);
  EXPECT_EQ(index->invlists->list_size(0), 0);
  EXPECT_EQ(index->invlists->list_size(1), 0);
}

}  // namespace

TEST(IndexIVFCorrectness, InnerProductTrainingUsesSphericalCentroids) {
  constexpr hypervec::idx_t d = 2;
  constexpr hypervec::idx_t nlist = 2;
  const std::vector<float> training = {
      10.0f, 0.0f, 9.0f, 0.0f, 1.0f, 1.0f, 0.9f, 0.9f,
  };

  hypervec::IndexIVFFlat index(d, nlist, hypervec::kMetricInnerProduct);
  index.Train(4, training.data());

  for (hypervec::idx_t list_no = 0; list_no < nlist; ++list_no) {
    const float* centroid = index.centroids.data() + list_no * d;
    const float norm =
        std::sqrt(centroid[0] * centroid[0] + centroid[1] * centroid[1]);
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
  }
}

TEST(IndexIVFCorrectness, FailedRetrainingPreservesUsableState) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexIVFFlat index(d, 2);
  const std::vector<float> training = {0.0f, 0.0f, 10.0f, 10.0f};
  index.Train(2, training.data());
  const std::vector<float> original_centroids = index.centroids;

  EXPECT_THROW(index.Train(1, training.data()), hypervec::HypervecException);
  EXPECT_TRUE(index.is_trained);
  EXPECT_EQ(index.centroids, original_centroids);
}

TEST(IndexIVFCorrectness, RetrainingNonEmptyIndexIsRejected) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexIVFFlat index(d, 1);
  const std::vector<float> vector = {0.0f, 0.0f};
  index.Train(1, vector.data());
  index.Add(1, vector.data());

  EXPECT_THROW(index.Train(1, vector.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(index.invlists->compute_ntotal(), 1);
}

TEST(IndexIVFCorrectness, AddRollsBackEveryListOnFailure) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexIVFFlat index(d, 2);
  index.centroids = {0.0f, 0.0f, 100.0f, 100.0f};
  index.is_trained = true;

  delete index.invlists;
  index.invlists =
      new FailingArrayInvertedLists(2, sizeof(float) * d, /*fail_on_call=*/2);
  index.own_invlists = true;

  const std::vector<float> vectors = {0.0f, 0.0f, 100.0f, 100.0f};
  EXPECT_THROW(index.Add(2, vectors.data()), std::runtime_error);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.invlists->list_size(0), 0);
  EXPECT_EQ(index.invlists->list_size(1), 0);
}

TEST(IndexIVFCorrectness, TrainedCompressedRangeSearchFailsExplicitly) {
  constexpr hypervec::idx_t d = 4;
  constexpr hypervec::idx_t count = 16;
  const auto training = RandomVectors(count, d, 2001);

  hypervec::IndexIVFPQ pq(d, 2, 2, 2);
  pq.Train(count, training.data());
  pq.Add(count, training.data());
  hypervec::RangeSearchResult pq_result(1);
  EXPECT_THROW(pq.RangeSearch(1, training.data(), 1.0f, &pq_result),
               hypervec::HypervecException);

  hypervec::IndexIVFLVQ lvq(d, 2, 2, 2);
  lvq.Train(count, training.data());
  lvq.Add(count, training.data());
  hypervec::RangeSearchResult lvq_result(1);
  EXPECT_THROW(lvq.RangeSearch(1, training.data(), 1.0f, &lvq_result),
               hypervec::HypervecException);
}

TEST(IndexIVFCorrectness, NonPositiveNprobeIsRejected) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexIVFFlat index(d, 1);
  const std::vector<float> vector = {0.0f, 0.0f};
  index.Train(1, vector.data());
  index.Add(1, vector.data());

  hypervec::IVFSearchParameters params;
  params.nprobe = 0;
  float distance;
  hypervec::idx_t label;
  EXPECT_THROW(index.Search(1, vector.data(), 1, &distance, &label, &params),
               hypervec::HypervecException);
}

TEST(IndexIVFCorrectness, FailedQuantizerTrainingDoesNotPartiallyCommit) {
  constexpr hypervec::idx_t d = 4;
  const auto insufficient = RandomVectors(4, d, 2002);

  hypervec::IndexIVFPQ pq(d, 2, 2, 3);
  const auto pq_centroids = pq.centroids;
  EXPECT_THROW(pq.Train(4, insufficient.data()), hypervec::HypervecException);
  EXPECT_FALSE(pq.is_trained);
  EXPECT_FALSE(pq.pq.is_trained);
  EXPECT_EQ(pq.centroids, pq_centroids);

  hypervec::IndexIVFLVQ lvq(d, 2, 8, 2);
  const auto lvq_centroids = lvq.centroids;
  EXPECT_THROW(lvq.Train(4, insufficient.data()), hypervec::HypervecException);
  EXPECT_FALSE(lvq.is_trained);
  EXPECT_FALSE(lvq.lvq.is_trained);
  EXPECT_EQ(lvq.centroids, lvq_centroids);
}

TEST(IndexIVFCorrectness, FailedQuantizerRetrainingPreservesUsableState) {
  constexpr hypervec::idx_t d = 4;
  const auto training = RandomVectors(16, d, 2003);
  const auto insufficient = RandomVectors(2, d, 2004);

  hypervec::IndexIVFPQ pq(d, 2, 2, 2);
  pq.use_precomputed_table = 1;
  pq.Train(16, training.data());
  const auto pq_coarse = pq.centroids;
  const auto pq_codebooks = pq.pq.centroids;
  const auto pq_table = pq.precomputed_table;
  EXPECT_THROW(pq.Train(2, insufficient.data()), hypervec::HypervecException);
  EXPECT_TRUE(pq.is_trained);
  EXPECT_TRUE(pq.pq.is_trained);
  EXPECT_EQ(pq.centroids, pq_coarse);
  EXPECT_EQ(pq.pq.centroids, pq_codebooks);
  EXPECT_EQ(pq.precomputed_table, pq_table);

  hypervec::IndexIVFLVQ lvq(d, 2, 4, 2);
  lvq.Train(16, training.data());
  const auto lvq_coarse = lvq.centroids;
  const auto lvq_codebooks = lvq.lvq.decoded_codebooks;
  EXPECT_THROW(lvq.Train(2, insufficient.data()), hypervec::HypervecException);
  EXPECT_TRUE(lvq.is_trained);
  EXPECT_TRUE(lvq.lvq.is_trained);
  EXPECT_EQ(lvq.centroids, lvq_coarse);
  EXPECT_EQ(lvq.lvq.decoded_codebooks, lvq_codebooks);
}

TEST(IndexIVFCorrectness, InvalidPrecomputedModeDoesNotCommitTraining) {
  constexpr hypervec::idx_t d = 4;
  const auto training = RandomVectors(16, d, 2005);
  hypervec::IndexIVFPQ pq(d, 2, 2, 2);
  pq.by_residual = false;
  pq.use_precomputed_table = 1;

  EXPECT_THROW(pq.Train(16, training.data()), hypervec::HypervecException);
  EXPECT_FALSE(pq.is_trained);
  EXPECT_FALSE(pq.pq.is_trained);
  EXPECT_TRUE(pq.precomputed_table.empty());
}

TEST(IndexIVFCorrectness, QuantizedAddsRollBackEveryTouchedList) {
  constexpr hypervec::idx_t d = 4;
  const auto training = RandomVectors(16, d, 2006);

  hypervec::IndexIVFPQ pq(d, 2, 2, 2);
  pq.Train(16, training.data());
  ExpectCompressedAddRollback(&pq);

  hypervec::IndexIVFLVQ lvq(d, 2, 2, 2);
  lvq.Train(16, training.data());
  ExpectCompressedAddRollback(&lvq);
}
