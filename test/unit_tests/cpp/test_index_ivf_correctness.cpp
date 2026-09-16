/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/ivf/inverted_list_scanner.h>
#include <invlists/inverted_lists.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

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

class ScalarL2Scanner : public hypervec::InvertedListScanner {
 public:
  ScalarL2Scanner()
    : InvertedListScanner(hypervec::kMetricL2, sizeof(float)) {}

  void SetQuery(const float* query) override { query_ = *query; }
  void SetList(hypervec::idx_t /*list_no*/,
               float /*coarse_distance*/) override {}

  float DistanceToCode(const uint8_t* code) const override {
    float value;
    std::memcpy(&value, code, sizeof(value));
    const float difference = query_ - value;
    return difference * difference;
  }

 private:
  float query_ = 0.0f;
};

std::vector<float> MakeVectors(hypervec::idx_t count,
                               hypervec::idx_t dimension) {
  std::vector<float> vectors(static_cast<size_t>(count * dimension));
  for (hypervec::idx_t i = 0; i < count; ++i) {
    for (hypervec::idx_t j = 0; j < dimension; ++j) {
      vectors[static_cast<size_t>(i * dimension + j)] =
        0.2f * static_cast<float>(i) +
        0.05f * static_cast<float>((5 * i + 3 * j) % 11);
    }
  }
  return vectors;
}

template <typename IndexType>
void ExpectCompressedRangeMatchesKnn(IndexType* index,
                                     const std::vector<float>& query,
                                     hypervec::idx_t database_size) {
  hypervec::IDSelectorRange selector(0, database_size / 2);
  hypervec::IVFSearchParameters parameters;
  parameters.nprobe = index->nlist;
  parameters.sel = &selector;

  std::vector<float> distances(static_cast<size_t>(database_size));
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(database_size));
  index->Search(1, query.data(), database_size, distances.data(), labels.data(),
                &parameters);

  const float radius = distances[static_cast<size_t>(database_size / 4)];
  std::vector<std::pair<hypervec::idx_t, float>> expected;
  for (hypervec::idx_t i = 0; i < database_size; ++i) {
    if (labels[static_cast<size_t>(i)] >= 0 &&
        distances[static_cast<size_t>(i)] <= radius) {
      expected.emplace_back(labels[static_cast<size_t>(i)],
                            distances[static_cast<size_t>(i)]);
    }
  }

  hypervec::RangeSearchResult result(1);
  index->RangeSearch(1, query.data(), radius, &result, &parameters);
  std::vector<std::pair<hypervec::idx_t, float>> actual;
  for (size_t i = result.lims[0]; i < result.lims[1]; ++i) {
    actual.emplace_back(result.labels[i], result.distances[i]);
  }

  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].first, expected[i].first);
    EXPECT_FLOAT_EQ(actual[i].second, expected[i].second);
  }
}

}  // namespace

TEST(InvertedListScanner, CommonLoopsApplyFilterHeapAndRangeSemantics) {
  ScalarL2Scanner scanner;
  const float query = 0.0f;
  const std::vector<float> values = {3.0f, 1.0f, 2.0f, 4.0f};
  const hypervec::idx_t ids[] = {1, 2, 3, 4};
  hypervec::IDSelectorRange selector(2, 4);
  scanner.SetQuery(&query);
  scanner.SetList(0, 0.0f);

  float heap_distances[2];
  hypervec::idx_t heap_ids[2];
  hypervec::heap_heapify<hypervec::CMax<float, hypervec::idx_t>>(
    2, heap_distances, heap_ids);
  EXPECT_EQ(scanner.ScanKnn(reinterpret_cast<const uint8_t*>(values.data()),
                            ids, values.size(), &selector, 2, heap_distances,
                            heap_ids),
            2U);
  hypervec::heap_reorder<hypervec::CMax<float, hypervec::idx_t>>(
    2, heap_distances, heap_ids);
  EXPECT_EQ(heap_ids[0], 2);
  EXPECT_FLOAT_EQ(heap_distances[0], 1.0f);
  EXPECT_EQ(heap_ids[1], 3);
  EXPECT_FLOAT_EQ(heap_distances[1], 4.0f);

  std::vector<std::pair<float, hypervec::idx_t>> range_results;
  EXPECT_EQ(scanner.ScanRange(
              reinterpret_cast<const uint8_t*>(values.data()), ids,
              values.size(), &selector, 2.0f, &range_results),
            2U);
  ASSERT_EQ(range_results.size(), 1U);
  EXPECT_EQ(range_results[0].second, 2);
  EXPECT_FLOAT_EQ(range_results[0].first, 1.0f);
}

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

  std::vector<float> distances(4);
  std::vector<hypervec::idx_t> labels(4);
  index.FindNearestCentroids(4, training.data(), 1, distances.data(),
                             labels.data());
  for (hypervec::idx_t i = 0; i < 4; ++i) {
    const float* vector = training.data() + i * d;
    hypervec::idx_t expected = 0;
    float best = -INFINITY;
    for (hypervec::idx_t list_no = 0; list_no < nlist; ++list_no) {
      const float* centroid = index.centroids.data() + list_no * d;
      const float dot = vector[0] * centroid[0] + vector[1] * centroid[1];
      if (dot > best) {
        best = dot;
        expected = list_no;
      }
    }
    EXPECT_EQ(labels[static_cast<size_t>(i)], expected);
  }
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

TEST(IndexIVFCorrectness, UntrainedCompressedRangeSearchFailsExplicitly) {
  hypervec::IndexIVFPQ pq(8, 4, 2, 4);
  hypervec::RangeSearchResult pq_result(1);
  std::vector<float> query(8);
  EXPECT_THROW(pq.RangeSearch(1, query.data(), 1.0f, &pq_result),
               hypervec::HypervecException);

  hypervec::IndexIVFLVQ lvq(8, 4, 2, 4);
  hypervec::RangeSearchResult lvq_result(1);
  EXPECT_THROW(lvq.RangeSearch(1, query.data(), 1.0f, &lvq_result),
               hypervec::HypervecException);
}

TEST(IndexIVFCorrectness, PQScannerSupportsFilteredRangeSearch) {
  constexpr hypervec::idx_t kDimension = 4;
  constexpr hypervec::idx_t kDatabaseSize = 64;
  std::vector<float> vectors = MakeVectors(kDatabaseSize, kDimension);

  hypervec::IndexIVFPQ index(kDimension, 4, 2, 2);
  index.Train(kDatabaseSize, vectors.data());
  index.Add(kDatabaseSize, vectors.data());
  const std::vector<float> query = {1.1f, 0.7f, 1.4f, 0.2f};
  ExpectCompressedRangeMatchesKnn(&index, query, kDatabaseSize);
  EXPECT_TRUE(hypervec::HasCapability(
    index.Capabilities(), hypervec::IndexCapability::RangeSearch));
}

TEST(IndexIVFCorrectness, LVQScannerSupportsFilteredRangeSearch) {
  constexpr hypervec::idx_t kDimension = 4;
  constexpr hypervec::idx_t kDatabaseSize = 64;
  std::vector<float> vectors = MakeVectors(kDatabaseSize, kDimension);

  hypervec::IndexIVFLVQ index(kDimension, 4, 2, 2);
  index.Train(kDatabaseSize, vectors.data());
  index.Add(kDatabaseSize, vectors.data());
  const std::vector<float> query = {1.1f, 0.7f, 1.4f, 0.2f};
  ExpectCompressedRangeMatchesKnn(&index, query, kDatabaseSize);
  EXPECT_TRUE(hypervec::HasCapability(
    index.Capabilities(), hypervec::IndexCapability::RangeSearch));
}

TEST(IndexIVFCorrectness, FailedCompressedTrainingLeavesIndexUntrained) {
  const std::vector<float> pq_training(8 * 4, 0.5f);
  hypervec::IndexIVFPQ pq(4, 2, 2, 8);
  EXPECT_THROW(pq.Train(8, pq_training.data()), hypervec::HypervecException);
  EXPECT_FALSE(pq.is_trained);
  EXPECT_FALSE(pq.pq.is_trained);

  const std::vector<float> lvq_training(4 * 4, 0.5f);
  hypervec::IndexIVFLVQ lvq(4, 2, 8, 2);
  EXPECT_THROW(lvq.Train(4, lvq_training.data()), hypervec::HypervecException);
  EXPECT_FALSE(lvq.is_trained);
  EXPECT_FALSE(lvq.lvq.is_trained);
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

TEST(IndexIVFCorrectness, ZeroCountBehaviorIsExplicit) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexIVFFlat untrained(d, 1);
  EXPECT_THROW(untrained.Train(0, nullptr), hypervec::HypervecException);
  EXPECT_FALSE(untrained.is_trained);

  const std::vector<float> vector = {0.0f, 0.0f};
  untrained.Train(1, vector.data());
  EXPECT_NO_THROW(untrained.Add(0, nullptr));
  EXPECT_NO_THROW(untrained.SearchPreassigned(
    0, nullptr, 1, nullptr, nullptr, nullptr, nullptr, 1, nullptr));
  EXPECT_EQ(untrained.n_total, 0);
}
