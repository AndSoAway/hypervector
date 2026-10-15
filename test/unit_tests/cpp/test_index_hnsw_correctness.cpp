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
#include <utils/common/range_search_result.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distance_computer.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/random.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

class ThrowingSearchDistanceComputer final : public hypervec::DistanceComputer {
 public:
  explicit ThrowingSearchDistanceComputer(int* destruction_count)
      : destruction_count_(destruction_count) {}

  ~ThrowingSearchDistanceComputer() override { ++*destruction_count_; }

  void SetQuery(const float*) override {}

  float operator()(hypervec::idx_t) override {
    throw std::runtime_error("injected search failure");
  }

  float symmetric_dis(hypervec::idx_t, hypervec::idx_t) override {
    return 0.0F;
  }

 private:
  int* destruction_count_;
};

class CollectingResultHandler final : public hypervec::ResultHandler {
 public:
  bool AddResult(float distance, hypervec::idx_t id) override {
    results.emplace_back(distance, id);
    return false;
  }

  std::vector<std::pair<float, hypervec::idx_t>> results;
};

class ThrowingSearchStorage final : public hypervec::Index {
 public:
  explicit ThrowingSearchStorage(int* destruction_count)
      : Index(1, hypervec::kMetricL2), destruction_count_(destruction_count) {
    n_total = 1;
  }

  void Add(hypervec::idx_t, const float*) override {}

  void Search(hypervec::idx_t, const float*, hypervec::idx_t, float*,
              hypervec::idx_t*,
              const hypervec::SearchParameters*) const override {}

  void Reset() override { n_total = 0; }

  hypervec::DistanceComputer* GetDistanceComputer() const override {
    return new ThrowingSearchDistanceComputer(destruction_count_);
  }

 private:
  int* destruction_count_;
};

class ThrowingReconstructStorage final : public hypervec::IndexFlatL2 {
 public:
  explicit ThrowingReconstructStorage(hypervec::idx_t dimension)
      : IndexFlatL2(dimension) {}

  void Reconstruct(hypervec::idx_t key, float* output) const override {
    if (key == fail_key) {
      throw std::runtime_error("injected reconstruct failure");
    }
    hypervec::IndexFlat::Reconstruct(key, output);
  }

  hypervec::idx_t fail_key = -1;
};

class ThrowingResetStorage final : public hypervec::IndexFlatL2 {
 public:
  explicit ThrowingResetStorage(hypervec::idx_t dimension)
      : IndexFlatL2(dimension) {}

  void Reset() override {
    if (fail_reset) {
      throw std::runtime_error("injected reset failure");
    }
    IndexFlatL2::Reset();
  }

  bool fail_reset = false;
};

class FailingAddDistanceComputer final : public hypervec::DistanceComputer {
 public:
  FailingAddDistanceComputer(
      std::unique_ptr<hypervec::DistanceComputer> delegate, int fail_on_query)
      : delegate_(std::move(delegate)), fail_on_query_(fail_on_query) {}

  void SetQuery(const float* query) override {
    if (query_count_++ == fail_on_query_) {
      throw std::runtime_error("injected graph-build failure");
    }
    delegate_->SetQuery(query);
  }

  float operator()(hypervec::idx_t index) override {
    return (*delegate_)(index);
  }

  float symmetric_dis(hypervec::idx_t first, hypervec::idx_t second) override {
    return delegate_->symmetric_dis(first, second);
  }

 private:
  std::unique_ptr<hypervec::DistanceComputer> delegate_;
  int fail_on_query_;
  int query_count_ = 0;
};

class FailingAddStorage final : public hypervec::IndexFlatL2 {
 public:
  explicit FailingAddStorage(hypervec::idx_t dimension)
      : IndexFlatL2(dimension) {}

  bool fail_graph_build = false;

  hypervec::DistanceComputer* GetDistanceComputer() const override {
    std::unique_ptr<hypervec::DistanceComputer> delegate(
        IndexFlatL2::GetDistanceComputer());
    if (!fail_graph_build) {
      return delegate.release();
    }
    return new FailingAddDistanceComputer(std::move(delegate), 1);
  }
};

class FailingAfterAddPQ final : public hypervec::IndexPQ {
 public:
  explicit FailingAfterAddPQ(const hypervec::IndexPQ& source)
      : IndexPQ(source) {}

  bool fail_after_add = true;

  void Add(hypervec::idx_t n, const float* x) override {
    IndexPQ::Add(n, x);
    if (fail_after_add) {
      throw std::runtime_error("injected compressed-storage failure");
    }
  }
};

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

size_t CountLevel0Neighbors(const hypervec::IndexHNSW& index,
                            hypervec::idx_t node) {
  size_t begin = 0;
  size_t end = 0;
  index.hnsw.NeighborRange(node, 0, &begin, &end);
  size_t count = 0;
  while (begin + count < end && index.hnsw.neighbors[begin + count] >= 0) {
    ++count;
  }
  return count;
}

std::vector<hypervec::HNSW::storage_idx_t> Level0Neighbors(
    const hypervec::IndexHNSW& index, hypervec::idx_t node) {
  size_t begin = 0;
  size_t end = 0;
  index.hnsw.NeighborRange(node, 0, &begin, &end);
  std::vector<hypervec::HNSW::storage_idx_t> result;
  while (begin < end && index.hnsw.neighbors[begin] >= 0) {
    result.push_back(index.hnsw.neighbors[begin++]);
  }
  return result;
}

std::vector<size_t> Level0IncomingCounts(const hypervec::IndexHNSW& index) {
  std::vector<size_t> incoming(static_cast<size_t>(index.n_total), 0);
  for (hypervec::idx_t node = 0; node < index.n_total; ++node) {
    for (const auto neighbor : Level0Neighbors(index, node)) {
      ++incoming[static_cast<size_t>(neighbor)];
    }
  }
  return incoming;
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

std::vector<hypervec::idx_t> MakePermutation(hypervec::idx_t count) {
  std::vector<hypervec::idx_t> permutation(static_cast<size_t>(count));
  for (hypervec::idx_t new_id = 0; new_id < count; ++new_id) {
    permutation[static_cast<size_t>(new_id)] = (new_id * 5 + 3) % count;
  }
  return permutation;
}

std::vector<hypervec::idx_t> InvertPermutation(
    const std::vector<hypervec::idx_t>& permutation) {
  std::vector<hypervec::idx_t> inverse(permutation.size());
  for (size_t new_id = 0; new_id < permutation.size(); ++new_id) {
    inverse[static_cast<size_t>(permutation[new_id])] =
        static_cast<hypervec::idx_t>(new_id);
  }
  return inverse;
}

void ExpectVector(const hypervec::Index& index, hypervec::idx_t id,
                  const float* expected) {
  std::vector<float> reconstructed(static_cast<size_t>(index.d));
  index.Reconstruct(id, reconstructed.data());
  for (int component = 0; component < index.d; ++component) {
    EXPECT_FLOAT_EQ(reconstructed[static_cast<size_t>(component)],
                    expected[component]);
  }
}

void ExpectCompressedPermutation(hypervec::IndexHNSW* index,
                                 hypervec::Index* raw_storage,
                                 const std::vector<float>& data,
                                 hypervec::idx_t initial_count,
                                 hypervec::idx_t extra_count) {
  ASSERT_NE(index, nullptr);
  ASSERT_NE(raw_storage, nullptr);
  const auto permutation = MakePermutation(initial_count);
  std::vector<float> decoded_before(
      static_cast<size_t>(initial_count * index->d));
  for (hypervec::idx_t id = 0; id < initial_count; ++id) {
    index->Reconstruct(
        id, decoded_before.data() + static_cast<size_t>(id * index->d));
  }

  index->PermuteEntries(permutation.data());
  ExpectConsistentGraph(*index);
  for (hypervec::idx_t new_id = 0; new_id < initial_count; ++new_id) {
    const hypervec::idx_t old_id = permutation[static_cast<size_t>(new_id)];
    ExpectVector(
        *index, new_id,
        decoded_before.data() + static_cast<size_t>(old_id * index->d));
    ExpectVector(*raw_storage, new_id,
                 data.data() + static_cast<size_t>(old_id * index->d));
  }

  index->Add(extra_count,
             data.data() + static_cast<size_t>(initial_count * index->d));
  ExpectConsistentGraph(*index);
  EXPECT_EQ(raw_storage->n_total, initial_count + extra_count);
  for (hypervec::idx_t id = 0; id < extra_count; ++id) {
    ExpectVector(
        *raw_storage, initial_count + id,
        data.data() + static_cast<size_t>((initial_count + id) * index->d));
  }
}

}  // namespace

TEST(IndexHNSWCorrectness, ConstructionRejectsInvalidGraphParameters) {
  EXPECT_THROW((hypervec::HNSW(0)), hypervec::HypervecException);
  EXPECT_THROW((hypervec::HNSW(1)), hypervec::HypervecException);
  EXPECT_THROW((hypervec::HNSW(-1)), hypervec::HypervecException);
  EXPECT_THROW((hypervec::HNSW(std::numeric_limits<int>::max())),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::IndexHNSW(nullptr, 8)), hypervec::HypervecException);
  EXPECT_THROW((hypervec::IndexHNSWFlat(4, 1)), hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, LayoutReconfigurationIsValidatedAndTransactional) {
  hypervec::HNSW graph(4);
  const auto probabilities = graph.assign_probas;
  const auto capacities = graph.cum_nneighbor_per_level;

  EXPECT_THROW(graph.SetDefaultProbas(1, 1.0F), hypervec::HypervecException);
  EXPECT_THROW(
      graph.SetDefaultProbas(4, std::numeric_limits<float>::infinity()),
      hypervec::HypervecException);
  EXPECT_EQ(graph.assign_probas, probabilities);
  EXPECT_EQ(graph.cum_nneighbor_per_level, capacities);

  graph.SetDefaultProbas(8, static_cast<float>(1.0 / std::log(8.0)));
  EXPECT_EQ(graph.NbNeighbors(0), 16);
  EXPECT_EQ(graph.NbNeighbors(1), 8);

  const auto updated_capacities = graph.cum_nneighbor_per_level;
  EXPECT_THROW(graph.NbNeighbors(-1), hypervec::HypervecException);
  EXPECT_THROW(graph.SetNbNeighbors(0, 0), hypervec::HypervecException);
  EXPECT_THROW(graph.SetNbNeighbors(0, std::numeric_limits<int>::max()),
               hypervec::HypervecException);
  EXPECT_EQ(graph.cum_nneighbor_per_level, updated_capacities);

  graph.SetNbNeighbors(0, 12);
  EXPECT_EQ(graph.NbNeighbors(0), 12);
  EXPECT_EQ(graph.NbNeighbors(1), 8);
  graph.PrepareLevelTab(1);
  EXPECT_THROW(graph.SetNbNeighbors(0, 16), hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, VisitedTablePoliciesCoverBuildAndSearchPaths) {
  constexpr hypervec::idx_t dimension = 4;
  constexpr hypervec::idx_t count = 32;
  constexpr hypervec::idx_t k = 5;
  const auto data = RandomVectors(count, dimension, 2111);

  hypervec::IndexHNSWFlat vector_index(dimension, 4);
  vector_index.hnsw.use_visited_hashset = false;
  vector_index.use_visited_hashset = false;
  vector_index.Add(count, data.data());

  hypervec::IndexHNSWFlat hash_index(dimension, 4);
  hash_index.hnsw.use_visited_hashset = true;
  hash_index.use_visited_hashset = true;
  hash_index.Add(count, data.data());

  EXPECT_EQ(vector_index.hnsw.levels, hash_index.hnsw.levels);
  EXPECT_EQ(vector_index.hnsw.offsets, hash_index.hnsw.offsets);
  EXPECT_EQ(
      std::vector<hypervec::HNSW::storage_idx_t>(
          vector_index.hnsw.neighbors.begin(),
          vector_index.hnsw.neighbors.end()),
      std::vector<hypervec::HNSW::storage_idx_t>(
          hash_index.hnsw.neighbors.begin(), hash_index.hnsw.neighbors.end()));

  std::array<float, k> vector_distances{};
  std::array<float, k> hash_distances{};
  std::array<hypervec::idx_t, k> vector_labels{};
  std::array<hypervec::idx_t, k> hash_labels{};
  vector_index.Search(1, data.data(), k, vector_distances.data(),
                      vector_labels.data());
  hash_index.Search(1, data.data(), k, hash_distances.data(),
                    hash_labels.data());
  EXPECT_EQ(vector_labels, hash_labels);
  EXPECT_EQ(vector_distances, hash_distances);

  hypervec::RangeSearchResult vector_range(1);
  hypervec::RangeSearchResult hash_range(1);
  vector_index.RangeSearch(1, data.data(), 2.0F, &vector_range);
  hash_index.RangeSearch(1, data.data(), 2.0F, &hash_range);
  ASSERT_EQ(vector_range.lims[1], hash_range.lims[1]);
  EXPECT_EQ(
      std::vector<hypervec::idx_t>(vector_range.labels,
                                   vector_range.labels + vector_range.lims[1]),
      std::vector<hypervec::idx_t>(hash_range.labels,
                                   hash_range.labels + hash_range.lims[1]));
  EXPECT_EQ(std::vector<float>(vector_range.distances,
                               vector_range.distances + vector_range.lims[1]),
            std::vector<float>(hash_range.distances,
                               hash_range.distances + hash_range.lims[1]));

  CollectingResultHandler vector_handler;
  CollectingResultHandler hash_handler;
  vector_index.Search1(data.data(), vector_handler);
  hash_index.Search1(data.data(), hash_handler);
  EXPECT_EQ(vector_handler.results, hash_handler.results);

  const hypervec::HNSW::storage_idx_t entry = 0;
  const float entry_distance = 0.0F;
  vector_index.SearchLevel0(1, data.data(), k, &entry, &entry_distance,
                            vector_distances.data(), vector_labels.data());
  hash_index.SearchLevel0(1, data.data(), k, &entry, &entry_distance,
                          hash_distances.data(), hash_labels.data());
  EXPECT_EQ(vector_labels, hash_labels);
  EXPECT_EQ(vector_distances, hash_distances);
}

TEST(IndexHNSWCorrectness, LifecycleRejectsMissingStorageWithoutMutation) {
  float reconstructed = 0.0F;

  hypervec::IndexHNSW bare;
  const auto bare_offsets = bare.hnsw.offsets;
  EXPECT_THROW(bare.Reset(), hypervec::HypervecException);
  EXPECT_THROW(bare.Reconstruct(0, &reconstructed),
               hypervec::HypervecException);
  EXPECT_EQ(bare.hnsw.offsets, bare_offsets);

  hypervec::IndexHNSWFlat flat;
  EXPECT_THROW(flat.Reset(), hypervec::HypervecException);
  EXPECT_THROW(flat.Reconstruct(0, &reconstructed),
               hypervec::HypervecException);

  hypervec::IndexHNSWPQ pq;
  EXPECT_THROW(pq.Reset(), hypervec::HypervecException);
  EXPECT_THROW(pq.Reconstruct(0, &reconstructed), hypervec::HypervecException);

  hypervec::IndexHNSWLVQ lvq;
  EXPECT_THROW(lvq.Reset(), hypervec::HypervecException);
  EXPECT_THROW(lvq.Reconstruct(0, &reconstructed), hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, FailedStorageResetPreservesGraphState) {
  constexpr hypervec::idx_t dimension = 2;
  constexpr hypervec::idx_t count = 8;
  const auto data = RandomVectors(count, dimension, 2137);
  ThrowingResetStorage storage(dimension);
  hypervec::IndexHNSW index(&storage, 4);
  index.Add(count, data.data());

  const auto levels = index.hnsw.levels;
  const auto offsets = index.hnsw.offsets;
  const auto neighbors = std::vector<hypervec::HNSW::storage_idx_t>(
      index.hnsw.neighbors.begin(), index.hnsw.neighbors.end());
  storage.fail_reset = true;
  EXPECT_THROW(index.Reset(), std::runtime_error);

  EXPECT_EQ(index.n_total, count);
  EXPECT_EQ(storage.n_total, count);
  EXPECT_EQ(index.hnsw.levels, levels);
  EXPECT_EQ(index.hnsw.offsets, offsets);
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.begin(), index.hnsw.neighbors.end()),
            neighbors);
}

TEST(IndexHNSWCorrectness, SearchReleasesDistanceComputerAfterException) {
  int destruction_count = 0;
  ThrowingSearchStorage storage(&destruction_count);
  hypervec::IndexHNSW index(&storage, 4);
  index.n_total = 1;
  index.hnsw.levels = {1};
  index.hnsw.offsets = {0, 8};
  index.hnsw.neighbors.resize(8);
  std::fill(index.hnsw.neighbors.begin(), index.hnsw.neighbors.end(), -1);
  index.hnsw.entry_point = 0;
  index.hnsw.max_level = 0;

  const float query = 0.0F;
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  EXPECT_THROW(index.Search(1, &query, 1, &distance, &label),
               std::runtime_error);
  EXPECT_EQ(destruction_count, 1);
}

TEST(IndexHNSWCorrectness, SearchValidatesInputsAndAlignedState) {
  hypervec::IndexHNSWFlat empty(2, 4);
  const std::array<float, 2> query = {0.0F, 1.0F};
  float distance = 0.0F;
  hypervec::idx_t label = 0;

  EXPECT_THROW(empty.Search(-1, nullptr, 1, nullptr, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(empty.Search(1, query.data(), 0, &distance, &label),
               hypervec::HypervecException);
  EXPECT_THROW(empty.Search(1, nullptr, 1, &distance, &label),
               hypervec::HypervecException);
  EXPECT_THROW(empty.Search(1, query.data(), 1, nullptr, &label),
               hypervec::HypervecException);
  EXPECT_THROW(empty.Search(1, query.data(), 1, &distance, nullptr),
               hypervec::HypervecException);
  EXPECT_NO_THROW(empty.Search(0, nullptr, 1, nullptr, nullptr));

  empty.Search(1, query.data(), 1, &distance, &label);
  EXPECT_EQ(label, -1);
  EXPECT_EQ(distance, (std::numeric_limits<float>::max)());

  hypervec::IndexHNSWFlat populated(2, 4);
  populated.Add(1, query.data());
  populated.storage->n_total = 0;
  EXPECT_THROW(populated.Search(1, query.data(), 1, &distance, &label),
               hypervec::HypervecException);
  populated.storage->n_total = 1;

  populated.hnsw.entry_point = 1;
  EXPECT_THROW(populated.Search(1, query.data(), 1, &distance, &label),
               hypervec::HypervecException);
  populated.hnsw.entry_point = 0;

  hypervec::SearchParametersHNSW params;
  params.ef_search = 0;
  EXPECT_THROW(populated.Search(1, query.data(), 1, &distance, &label, &params),
               hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, SearchRejectsUnsupportedPanoramaMode) {
  const std::array<float, 2> vector = {0.0F, 1.0F};
  hypervec::IndexHNSWFlat index(2, 4);
  index.Add(1, vector.data());
  index.hnsw.is_panorama = true;

  float distance = 0.0F;
  hypervec::idx_t label = -1;
  EXPECT_THROW(index.Search(1, vector.data(), 1, &distance, &label),
               hypervec::HypervecException);

  hypervec::SearchParametersHNSW params;
  params.bounded_queue = false;
  EXPECT_THROW(index.Search(1, vector.data(), 1, &distance, &label, &params),
               hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, SearchReturnsExternalSimilarityScores) {
  constexpr hypervec::idx_t dimension = 2;
  constexpr hypervec::idx_t count = 3;
  constexpr hypervec::idx_t k = 4;
  const std::vector<float> database = {1.0F, 0.2F, 0.2F, 1.0F, 0.7F, 0.7F};
  const std::array<float, dimension> query = {1.0F, 0.1F};

  for (const hypervec::MetricType metric :
       {hypervec::kMetricInnerProduct, hypervec::kMetricJaccard}) {
    hypervec::IndexFlat expected(dimension, metric);
    expected.Add(count, database.data());
    hypervec::IndexHNSWFlat actual(dimension, 4, metric);
    actual.Add(count, database.data());

    std::array<float, k> expected_distances{};
    std::array<float, k> actual_distances{};
    std::array<hypervec::idx_t, k> expected_labels{};
    std::array<hypervec::idx_t, k> actual_labels{};
    expected.Search(1, query.data(), k, expected_distances.data(),
                    expected_labels.data());
    actual.Search(1, query.data(), k, actual_distances.data(),
                  actual_labels.data());

    EXPECT_EQ(actual_labels, expected_labels);
    EXPECT_GT(actual_distances[0], 0.0F);
    for (size_t result = 0; result < static_cast<size_t>(k); ++result) {
      EXPECT_FLOAT_EQ(actual_distances[result], expected_distances[result]);
    }

    std::unique_ptr<hypervec::DistanceComputer> distance(
        actual.GetDistanceComputer());
    distance->SetQuery(query.data());
    EXPECT_FLOAT_EQ((*distance)(actual_labels[0]), actual_distances[0]);
  }
}

TEST(IndexHNSWCorrectness, AuxiliarySearchUsesGraphMetricAndFiltering) {
  constexpr hypervec::idx_t dimension = 2;
  constexpr hypervec::idx_t count = 4;
  const std::array<float, dimension * count> database = {
      1.0F, 0.0F, 0.0F, 1.0F, 0.8F, 0.2F, 0.2F, 0.8F};
  const std::array<float, dimension> query = {1.0F, 0.1F};
  hypervec::IndexHNSWFlat index(dimension, 4, hypervec::kMetricInnerProduct);
  index.Add(count, database.data());

  hypervec::IDSelectorRange selector(0, 3);
  hypervec::SearchParametersHNSW params;
  params.ef_search = count;
  params.check_relative_distance = false;
  params.sel = &selector;

  CollectingResultHandler collected;
  index.Search1(query.data(), collected, &params);
  ASSERT_EQ(collected.results.size(), 3U);
  for (const auto& [score, id] : collected.results) {
    EXPECT_TRUE(selector.IsMember(id));
    const float expected =
        database[static_cast<size_t>(id) * dimension] +
        0.1F * database[static_cast<size_t>(id) * dimension + 1];
    EXPECT_FLOAT_EQ(score, expected);
  }

  hypervec::RangeSearchResult range(1);
  index.RangeSearch(1, query.data(), 0.5F, &range, &params);
  ASSERT_EQ(range.lims[1], 2U);
  for (size_t result = 0; result < range.lims[1]; ++result) {
    EXPECT_TRUE(range.labels[result] == 0 || range.labels[result] == 2);
    EXPECT_GT(range.distances[result], 0.5F);
  }
}

TEST(IndexHNSWCorrectness, AuxiliarySearchValidatesInputs) {
  const std::array<float, 2> query = {0.0F, 1.0F};
  hypervec::IndexHNSWFlat index(2, 4);

  EXPECT_NO_THROW(index.RangeSearch(0, nullptr, 1.0F, nullptr));
  CollectingResultHandler empty_collected;
  EXPECT_THROW(index.Search1(nullptr, empty_collected),
               hypervec::HypervecException);

  hypervec::RangeSearchResult result(1);
  hypervec::RangeSearchResult wrong_query_count(2);
  EXPECT_THROW(index.RangeSearch(-1, nullptr, 1.0F, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.RangeSearch(1, nullptr, 1.0F, &result),
               hypervec::HypervecException);
  EXPECT_THROW(index.RangeSearch(1, query.data(), 1.0F, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.RangeSearch(1, query.data(), 1.0F, &wrong_query_count),
               hypervec::HypervecException);

  index.Add(1, query.data());
  hypervec::SearchParametersHNSW params;
  params.ef_search = 0;
  EXPECT_THROW(index.RangeSearch(1, query.data(), 1.0F, &result, &params),
               hypervec::HypervecException);
  CollectingResultHandler collected;
  EXPECT_THROW(index.Search1(query.data(), collected, &params),
               hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, PublicLevel0SearchSupportsBatchesAndEntryModes) {
  constexpr hypervec::idx_t count = 8;
  constexpr hypervec::idx_t query_count = 2;
  constexpr hypervec::idx_t k = 3;
  constexpr int nprobe = 2;
  const std::array<float, count> database = {0.0F, 1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F, 7.0F};
  const std::array<float, query_count> queries = {1.2F, 6.2F};
  const std::array<hypervec::HNSW::storage_idx_t, query_count * nprobe>
      entries = {0, 7, 7, 0};
  const std::array<float, query_count * nprobe> entry_distances = {
      1.44F, 33.64F, 0.64F, 38.44F};
  const std::array<hypervec::idx_t, query_count * k> expected_labels = {
      1, 2, 0, 6, 7, 5};
  const std::array<float, query_count * k> expected_distances = {
      0.04F, 0.64F, 1.44F, 0.04F, 0.64F, 1.44F};

  hypervec::IndexHNSWFlat index(1, 8);
  index.Add(count, database.data());
  hypervec::SearchParametersHNSW params;
  params.ef_search = count;
  params.check_relative_distance = false;

  for (int search_type : {1, 2}) {
    std::array<float, query_count * k> distances{};
    std::array<hypervec::idx_t, query_count * k> labels{};
    index.SearchLevel0(query_count, queries.data(), k, entries.data(),
                       entry_distances.data(), distances.data(), labels.data(),
                       nprobe, search_type, &params);

    EXPECT_EQ(labels, expected_labels);
    for (size_t result = 0; result < distances.size(); ++result) {
      EXPECT_NEAR(distances[result], expected_distances[result], 1e-5F);
    }
  }
}

TEST(IndexHNSWCorrectness, PublicLevel0SearchPreservesSimilarityDirection) {
  constexpr hypervec::idx_t dimension = 2;
  constexpr hypervec::idx_t count = 4;
  constexpr hypervec::idx_t k = 3;
  constexpr int nprobe = 2;
  const std::array<float, dimension * count> database = {
      1.0F, 0.0F, 0.0F, 1.0F, 0.8F, 0.2F, 0.2F, 0.8F};
  const std::array<float, dimension> query = {1.0F, 0.1F};
  const std::array<hypervec::HNSW::storage_idx_t, nprobe> entries = {0, 1};
  const std::array<float, nprobe> entry_similarities = {1.0F, 0.1F};

  hypervec::IndexFlatIP exact(dimension);
  exact.Add(count, database.data());
  hypervec::IndexHNSWFlat index(dimension, 4, hypervec::kMetricInnerProduct);
  index.Add(count, database.data());
  hypervec::SearchParametersHNSW params;
  params.ef_search = count;
  params.check_relative_distance = false;

  std::array<float, k> expected_distances{};
  std::array<hypervec::idx_t, k> expected_labels{};
  std::array<float, k> actual_distances{};
  std::array<hypervec::idx_t, k> actual_labels{};
  exact.Search(1, query.data(), k, expected_distances.data(),
               expected_labels.data());
  index.SearchLevel0(1, query.data(), k, entries.data(),
                     entry_similarities.data(), actual_distances.data(),
                     actual_labels.data(), nprobe, 2, &params);

  EXPECT_EQ(actual_labels, expected_labels);
  for (size_t result = 0; result < actual_distances.size(); ++result) {
    EXPECT_FLOAT_EQ(actual_distances[result], expected_distances[result]);
  }
}

TEST(IndexHNSWCorrectness, PublicLevel0SearchValidatesInputsAndEmptyIndex) {
  constexpr hypervec::idx_t k = 2;
  const std::array<float, 2> query = {0.0F, 1.0F};
  const hypervec::HNSW::storage_idx_t entry = -1;
  const float entry_distance = 0.0F;
  std::array<float, k> distances{};
  std::array<hypervec::idx_t, k> labels{};
  hypervec::IndexHNSWFlat index(2, 4);

  EXPECT_NO_THROW(
      index.SearchLevel0(0, nullptr, k, nullptr, nullptr, nullptr, nullptr));
  index.SearchLevel0(1, query.data(), k, &entry, &entry_distance,
                     distances.data(), labels.data());
  EXPECT_EQ(labels, (std::array<hypervec::idx_t, k>{-1, -1}));
  EXPECT_EQ(distances,
            (std::array<float, k>{(std::numeric_limits<float>::max)(),
                                  (std::numeric_limits<float>::max)()}));

  EXPECT_THROW(
      index.SearchLevel0(-1, nullptr, k, nullptr, nullptr, nullptr, nullptr),
      hypervec::HypervecException);
  EXPECT_THROW(index.SearchLevel0(1, nullptr, k, &entry, &entry_distance,
                                  distances.data(), labels.data()),
               hypervec::HypervecException);
  EXPECT_THROW(index.SearchLevel0(1, query.data(), 0, &entry, &entry_distance,
                                  distances.data(), labels.data()),
               hypervec::HypervecException);
  EXPECT_THROW(index.SearchLevel0(1, query.data(), k, &entry, &entry_distance,
                                  distances.data(), labels.data(), 0),
               hypervec::HypervecException);
  EXPECT_THROW(index.SearchLevel0(1, query.data(), k, &entry, &entry_distance,
                                  distances.data(), labels.data(), 1, 3),
               hypervec::HypervecException);

  index.Add(1, query.data());
  const hypervec::HNSW::storage_idx_t invalid_entry = 1;
  EXPECT_THROW(
      index.SearchLevel0(1, query.data(), k, &invalid_entry, &entry_distance,
                         distances.data(), labels.data()),
      hypervec::HypervecException);
  hypervec::SearchParametersHNSW params;
  params.ef_search = 0;
  const hypervec::HNSW::storage_idx_t valid_entry = 0;
  EXPECT_THROW(
      index.SearchLevel0(1, query.data(), k, &valid_entry, &entry_distance,
                         distances.data(), labels.data(), 1, 1, &params),
      hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, UnboundedQueueUsesSharedFilteredTraversal) {
  constexpr hypervec::idx_t count = 32;
  constexpr hypervec::idx_t k = 5;
  std::array<float, count> database{};
  for (size_t index = 0; index < database.size(); ++index) {
    database[index] = static_cast<float>(index);
  }
  hypervec::IndexHNSWFlat index(1, 4);
  index.Add(count, database.data());

  hypervec::IDSelectorRange selector(10, 20);
  hypervec::SearchParametersHNSW params;
  params.ef_search = count;
  params.bounded_queue = false;
  params.sel = &selector;
  const float query = 12.25F;
  std::array<float, k> distances{};
  std::array<hypervec::idx_t, k> labels{};

  index.Search(1, &query, k, distances.data(), labels.data(), &params);

  EXPECT_EQ(labels, (std::array<hypervec::idx_t, k>{12, 13, 11, 14, 10}));
  for (hypervec::idx_t label : labels) {
    EXPECT_TRUE(selector.IsMember(label));
  }
}

TEST(IndexHNSWCorrectness, BoundedQueueUsesSharedFilteredTraversal) {
  constexpr hypervec::idx_t count = 32;
  constexpr hypervec::idx_t k = 5;
  std::array<float, count> database{};
  for (size_t index = 0; index < database.size(); ++index) {
    database[index] = static_cast<float>(index);
  }
  hypervec::IndexHNSWFlat index(1, 4);
  index.Add(count, database.data());

  hypervec::IDSelectorRange selector(10, 20);
  hypervec::SearchParametersHNSW params;
  params.ef_search = count;
  params.bounded_queue = true;
  params.sel = &selector;
  const float query = 12.25F;
  std::array<float, k> distances{};
  std::array<hypervec::idx_t, k> labels{};

  index.Search(1, &query, k, distances.data(), labels.data(), &params);

  EXPECT_EQ(labels, (std::array<hypervec::idx_t, k>{12, 13, 11, 14, 10}));
  for (hypervec::idx_t label : labels) {
    EXPECT_TRUE(selector.IsMember(label));
  }
}

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

TEST(IndexHNSWCorrectness, FailedGraphBuildRollsBackStorageAndGraph) {
  constexpr hypervec::idx_t dimension = 4;
  constexpr hypervec::idx_t initial_count = 24;
  constexpr hypervec::idx_t extra_count = 4;
  const auto data = RandomVectors(initial_count + extra_count, dimension, 1004);
  FailingAddStorage storage(dimension);
  hypervec::IndexHNSW index(&storage, 4);
  index.Add(initial_count, data.data());
  storage.SyncL2Norms();

  const auto codes_before = storage.codes.owned_data;
  const auto norms_before = storage.cached_l2norms;
  const auto levels_before = index.hnsw.levels;
  const auto offsets_before = index.hnsw.offsets;
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  const auto entry_before = index.hnsw.entry_point;
  const auto max_level_before = index.hnsw.max_level;
  const auto rng_before = index.hnsw.rng.mt;

  storage.fail_graph_build = true;
  EXPECT_THROW(index.Add(extra_count, data.data() + initial_count * dimension),
               std::runtime_error);
  EXPECT_EQ(index.n_total, initial_count);
  EXPECT_EQ(storage.n_total, initial_count);
  EXPECT_EQ(storage.codes.owned_data, codes_before);
  EXPECT_EQ(storage.cached_l2norms, norms_before);
  EXPECT_EQ(index.hnsw.levels, levels_before);
  EXPECT_EQ(index.hnsw.offsets, offsets_before);
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            neighbors_before);
  EXPECT_EQ(index.hnsw.entry_point, entry_before);
  EXPECT_EQ(index.hnsw.max_level, max_level_before);
  EXPECT_EQ(index.hnsw.rng.mt, rng_before);

  storage.fail_graph_build = false;
  index.Add(extra_count, data.data() + initial_count * dimension);
  EXPECT_EQ(index.n_total, initial_count + extra_count);
  ExpectConsistentGraph(index);
  ExpectSearchLabelsValid(index, data.data());
}

TEST(IndexHNSWCorrectness, AddRejectsInvalidInputAndMappedGraph) {
  constexpr hypervec::idx_t dimension = 2;
  const std::array<float, dimension> vector = {1.0F, 2.0F};
  hypervec::IndexHNSWFlat index(dimension, 4);

  EXPECT_THROW(index.Add(-1, nullptr), hypervec::HypervecException);
  EXPECT_THROW(index.Add(1, nullptr), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 0);
  index.Add(1, vector.data());

  std::vector<hypervec::HNSW::storage_idx_t> mapped_neighbors(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  index.hnsw.neighbors =
      hypervec::MaybeOwnedVector<hypervec::HNSW::storage_idx_t>::create_view(
          mapped_neighbors.data(), mapped_neighbors.size(), nullptr);
  EXPECT_THROW(index.Add(1, vector.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(index.storage->n_total, 1);
}

TEST(IndexHNSWCorrectness, CompressedStorageFailureRollsBackBothStores) {
  constexpr hypervec::idx_t dimension = 8;
  constexpr hypervec::idx_t count = 32;
  const auto data = RandomVectors(count, dimension, 1005);
  hypervec::IndexHNSWPQ index(dimension, 2, 3, 8);
  index.Train(count, data.data());

  auto* original = dynamic_cast<hypervec::IndexPQ*>(index.storage);
  ASSERT_NE(original, nullptr);
  auto* failing = new FailingAfterAddPQ(*original);
  delete original;
  index.storage = failing;
  const auto rng_before = index.hnsw.rng.mt;

  EXPECT_THROW(index.Add(count, data.data()), std::runtime_error);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.storage->n_total, 0);
  EXPECT_EQ(index.raw_storage->n_total, 0);
  EXPECT_EQ(failing->codes.size(), 0U);
  auto* raw = dynamic_cast<hypervec::IndexFlatL2*>(index.raw_storage);
  ASSERT_NE(raw, nullptr);
  EXPECT_EQ(raw->codes.size(), 0U);
  EXPECT_TRUE(index.hnsw.levels.empty());
  EXPECT_EQ(index.hnsw.offsets, std::vector<size_t>{0});
  EXPECT_EQ(index.hnsw.neighbors.size(), 0U);
  EXPECT_EQ(index.hnsw.rng.mt, rng_before);

  failing->fail_after_add = false;
  index.Add(count, data.data());
  EXPECT_EQ(index.n_total, count);
  EXPECT_EQ(index.storage->n_total, count);
  EXPECT_EQ(index.raw_storage->n_total, count);
  ExpectConsistentGraph(index);
}

TEST(IndexHNSWCorrectness, ShrinkLevel0NeighborsPrunesTransactionally) {
  constexpr hypervec::idx_t dimension = 4;
  constexpr hypervec::idx_t count = 48;
  constexpr int target_neighbors = 3;
  const auto data = RandomVectors(count, dimension, 2000);
  hypervec::IndexHNSWFlat index(dimension, 8);
  index.Add(count, data.data());

  size_t begin = 0;
  size_t end = 0;
  index.hnsw.NeighborRange(0, 0, &begin, &end);
  ASSERT_GE(end - begin, 6U);
  for (size_t offset = 0; offset < 6; ++offset) {
    index.hnsw.neighbors[begin + offset] =
        static_cast<hypervec::HNSW::storage_idx_t>(offset + 1);
  }
  for (size_t offset = begin + 6; offset < end; ++offset) {
    index.hnsw.neighbors[offset] = -1;
  }

  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  index.ShrinkLevel0Neighbors(target_neighbors);

  for (hypervec::idx_t node = 0; node < count; ++node) {
    EXPECT_LE(CountLevel0Neighbors(index, node),
              static_cast<size_t>(target_neighbors));
    size_t level0_begin = 0;
    size_t level0_end = 0;
    index.hnsw.NeighborRange(node, 0, &level0_begin, &level0_end);
    const size_t node_end = index.hnsw.offsets[static_cast<size_t>(node) + 1];
    for (size_t offset = level0_end; offset < node_end; ++offset) {
      EXPECT_EQ(index.hnsw.neighbors[offset], neighbors_before[offset]);
    }
  }
  EXPECT_LE(CountLevel0Neighbors(index, 0),
            static_cast<size_t>(target_neighbors));
  ExpectConsistentGraph(index);

  index.hnsw.NeighborRange(count - 1, 0, &begin, &end);
  ASSERT_LT(begin, end);
  index.hnsw.neighbors[begin] =
      static_cast<hypervec::HNSW::storage_idx_t>(count);
  const std::vector<hypervec::HNSW::storage_idx_t> corrupted_graph(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  EXPECT_THROW(index.ShrinkLevel0Neighbors(2), hypervec::HypervecException);
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            corrupted_graph);
}

TEST(IndexHNSWCorrectness, ShrinkLevel0NeighborsValidatesSizeAndEmptyState) {
  hypervec::IndexHNSWFlat empty(2, 4);
  EXPECT_NO_THROW(empty.ShrinkLevel0Neighbors(1));
  EXPECT_THROW(empty.ShrinkLevel0Neighbors(0), hypervec::HypervecException);
  EXPECT_THROW(empty.ShrinkLevel0Neighbors(empty.hnsw.NbNeighbors(0) + 1),
               hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, ReorderLinksSortsDistanceWithoutChangingEdges) {
  constexpr hypervec::idx_t count = 16;
  std::vector<float> data(static_cast<size_t>(count));
  for (hypervec::idx_t id = 0; id < count; ++id) {
    data[static_cast<size_t>(id)] = static_cast<float>(id);
  }
  hypervec::IndexHNSWFlat index(1, 4);
  index.Add(count, data.data());

  size_t begin = 0;
  size_t end = 0;
  index.hnsw.NeighborRange(0, 0, &begin, &end);
  ASSERT_GE(end - begin, 3U);
  const std::array<hypervec::HNSW::storage_idx_t, 3> unordered = {3, 1, 2};
  std::copy(unordered.begin(), unordered.end(),
            index.hnsw.neighbors.data() + begin);
  std::fill(index.hnsw.neighbors.data() + begin + unordered.size(),
            index.hnsw.neighbors.data() + end, -1);
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());

  index.ReorderLinks();
  EXPECT_EQ(Level0Neighbors(index, 0),
            (std::vector<hypervec::HNSW::storage_idx_t>{1, 2, 3}));
  for (hypervec::idx_t node = 0; node < count; ++node) {
    size_t level0_begin = 0;
    size_t level0_end = 0;
    index.hnsw.NeighborRange(node, 0, &level0_begin, &level0_end);
    std::vector<hypervec::HNSW::storage_idx_t> old_edges;
    for (size_t offset = level0_begin;
         offset < level0_end && neighbors_before[offset] >= 0; ++offset) {
      old_edges.push_back(neighbors_before[offset]);
    }
    auto new_edges = Level0Neighbors(index, node);
    std::sort(old_edges.begin(), old_edges.end());
    std::sort(new_edges.begin(), new_edges.end());
    EXPECT_EQ(new_edges, old_edges);

    const size_t node_end = index.hnsw.offsets[static_cast<size_t>(node) + 1];
    for (size_t offset = level0_end; offset < node_end; ++offset) {
      EXPECT_EQ(index.hnsw.neighbors[offset], neighbors_before[offset]);
    }
  }
}

TEST(IndexHNSWCorrectness, ReorderLinksUsesSimilarityDirection) {
  constexpr hypervec::idx_t dimension = 2;
  const std::array<float, 8> data = {1.0F, 0.0F, 0.9F, 0.0F,
                                     0.4F, 0.0F, 0.7F, 0.0F};
  hypervec::IndexHNSWFlat index(dimension, 4, hypervec::kMetricInnerProduct);
  index.Add(4, data.data());

  size_t begin = 0;
  size_t end = 0;
  index.hnsw.NeighborRange(0, 0, &begin, &end);
  const std::array<hypervec::HNSW::storage_idx_t, 3> unordered = {2, 1, 3};
  std::copy(unordered.begin(), unordered.end(),
            index.hnsw.neighbors.data() + begin);
  std::fill(index.hnsw.neighbors.data() + begin + unordered.size(),
            index.hnsw.neighbors.data() + end, -1);

  index.ReorderLinks();
  EXPECT_EQ(Level0Neighbors(index, 0),
            (std::vector<hypervec::HNSW::storage_idx_t>{1, 3, 2}));
}

TEST(IndexHNSWCorrectness, ReorderLinksRejectsCorruptionWithoutMutation) {
  constexpr hypervec::idx_t count = 24;
  const auto data = RandomVectors(count, 3, 2004);
  hypervec::IndexHNSWFlat index(3, 4);
  index.Add(count, data.data());
  size_t begin = 0;
  size_t end = 0;
  index.hnsw.NeighborRange(count - 1, 0, &begin, &end);
  ASSERT_LT(begin, end);
  index.hnsw.neighbors[begin] =
      static_cast<hypervec::HNSW::storage_idx_t>(count);
  const std::vector<hypervec::HNSW::storage_idx_t> corrupted_graph(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());

  EXPECT_THROW(index.ReorderLinks(), hypervec::HypervecException);
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            corrupted_graph);

  hypervec::IndexHNSWFlat empty(3, 4);
  EXPECT_NO_THROW(empty.ReorderLinks());
}

TEST(IndexHNSWCorrectness, InitLevel0FromKnngraphReplacesOnlyBaseLayer) {
  constexpr hypervec::idx_t count = 24;
  constexpr int k = 4;
  std::vector<float> data(static_cast<size_t>(count));
  for (hypervec::idx_t id = 0; id < count; ++id) {
    data[static_cast<size_t>(id)] = static_cast<float>(id);
  }
  hypervec::IndexHNSWFlat index(1, 4);
  index.Add(count, data.data());
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());

  std::vector<float> distances(static_cast<size_t>(count) * k, 0.0F);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(count) * k, -1);
  for (hypervec::idx_t node = 0; node < count; ++node) {
    const size_t row = static_cast<size_t>(node) * k;
    labels[row] = node;
    for (int column = 1; column < 3; ++column) {
      const hypervec::idx_t neighbor = (node + column) % count;
      labels[row + static_cast<size_t>(column)] = neighbor;
      const float delta =
          data[static_cast<size_t>(node)] - data[static_cast<size_t>(neighbor)];
      distances[row + static_cast<size_t>(column)] = delta * delta;
    }
  }

  index.InitLevel0FromKnngraph(k, distances.data(), labels.data());
  for (hypervec::idx_t node = 0; node < count; ++node) {
    const auto level0 = Level0Neighbors(index, node);
    ASSERT_FALSE(level0.empty());
    EXPECT_LE(level0.size(), 2U);
    for (const auto neighbor : level0) {
      EXPECT_NE(neighbor, node);
      EXPECT_TRUE(neighbor == (node + 1) % count ||
                  neighbor == (node + 2) % count);
    }

    size_t level0_begin = 0;
    size_t level0_end = 0;
    index.hnsw.NeighborRange(node, 0, &level0_begin, &level0_end);
    const size_t node_end = index.hnsw.offsets[static_cast<size_t>(node) + 1];
    for (size_t offset = level0_end; offset < node_end; ++offset) {
      EXPECT_EQ(index.hnsw.neighbors[offset], neighbors_before[offset]);
    }
  }
}

TEST(IndexHNSWCorrectness,
     InitLevel0FromKnngraphConvertsPublicSimilarityValues) {
  constexpr hypervec::idx_t dimension = 2;
  constexpr hypervec::idx_t count = 4;
  constexpr int k = 4;
  const std::array<float, 8> data = {1.0F, 0.0F, 0.9F, 0.0F,
                                     0.4F, 0.0F, 0.7F, 0.0F};
  const std::array<hypervec::idx_t, 16> labels = {0, 2, 1, 3, 1, 0, 2, 3,
                                                  2, 0, 1, 3, 3, 0, 1, 2};
  std::array<float, 16> similarities{};
  for (hypervec::idx_t node = 0; node < count; ++node) {
    for (int column = 0; column < k; ++column) {
      const hypervec::idx_t neighbor =
          labels[static_cast<size_t>(node) * k + column];
      similarities[static_cast<size_t>(node) * k + column] =
          data[static_cast<size_t>(node) * dimension] *
              data[static_cast<size_t>(neighbor) * dimension] +
          data[static_cast<size_t>(node) * dimension + 1] *
              data[static_cast<size_t>(neighbor) * dimension + 1];
    }
  }
  hypervec::IndexHNSWFlat index(dimension, 4, hypervec::kMetricInnerProduct);
  index.Add(count, data.data());

  index.InitLevel0FromKnngraph(k, similarities.data(), labels.data());
  const auto level0 = Level0Neighbors(index, 0);
  ASSERT_FALSE(level0.empty());
  EXPECT_EQ(level0.front(), 1);
}

TEST(IndexHNSWCorrectness,
     InitLevel0FromKnngraphRejectsMalformedInputTransactionally) {
  constexpr hypervec::idx_t count = 8;
  constexpr int k = 3;
  const auto data = RandomVectors(count, 3, 2005);
  hypervec::IndexHNSWFlat index(3, 4);
  index.Add(count, data.data());
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  std::vector<float> distances(static_cast<size_t>(count) * k, 1.0F);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(count) * k, -1);
  for (hypervec::idx_t node = 0; node < count; ++node) {
    const size_t row = static_cast<size_t>(node) * k;
    labels[row] = node;
    labels[row + 1] = (node + 1) % count;
  }

  labels.back() = count;
  EXPECT_THROW(index.InitLevel0FromKnngraph(k, distances.data(), labels.data()),
               hypervec::HypervecException);
  labels.back() = -1;
  labels[0] = 1;
  labels[1] = 1;
  EXPECT_THROW(index.InitLevel0FromKnngraph(k, distances.data(), labels.data()),
               hypervec::HypervecException);
  labels[0] = 0;
  labels[1] = 1;
  distances[1] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(index.InitLevel0FromKnngraph(k, distances.data(), labels.data()),
               hypervec::HypervecException);
  distances[1] = 1.0F;
  labels[1] = -1;
  labels[2] = 2;
  EXPECT_THROW(index.InitLevel0FromKnngraph(k, distances.data(), labels.data()),
               hypervec::HypervecException);
  EXPECT_THROW(index.InitLevel0FromKnngraph(0, distances.data(), labels.data()),
               hypervec::HypervecException);
  EXPECT_THROW(index.InitLevel0FromKnngraph(k, nullptr, labels.data()),
               hypervec::HypervecException);
  EXPECT_THROW(index.InitLevel0FromKnngraph(k, distances.data(), nullptr),
               hypervec::HypervecException);

  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            neighbors_before);
  hypervec::IndexHNSWFlat empty(3, 4);
  EXPECT_NO_THROW(empty.InitLevel0FromKnngraph(1, nullptr, nullptr));
}

TEST(IndexHNSWCorrectness,
     InitLevel0FromEntryPointsAddsReciprocalBaseLinksOnly) {
  constexpr hypervec::idx_t count = 16;
  const auto data = RandomVectors(count, 3, 2006);
  hypervec::IndexHNSWFlat index(3, 4);
  index.Add(count, data.data());

  std::vector<float> self_distances(static_cast<size_t>(count), 0.0F);
  std::vector<hypervec::idx_t> self_labels(static_cast<size_t>(count));
  for (hypervec::idx_t node = 0; node < count; ++node) {
    self_labels[static_cast<size_t>(node)] = node;
  }
  index.InitLevel0FromKnngraph(1, self_distances.data(), self_labels.data());
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  const std::array<hypervec::HNSW::storage_idx_t, 2> points = {0, 4};
  const std::array<hypervec::HNSW::storage_idx_t, 2> nearests = {1, 5};

  index.InitLevel0FromEntryPoints(static_cast<int>(points.size()),
                                  points.data(), nearests.data());
  for (size_t input = 0; input < points.size(); ++input) {
    const auto point_edges = Level0Neighbors(index, points[input]);
    const auto nearest_edges = Level0Neighbors(index, nearests[input]);
    EXPECT_NE(
        std::find(point_edges.begin(), point_edges.end(), nearests[input]),
        point_edges.end());
    EXPECT_NE(
        std::find(nearest_edges.begin(), nearest_edges.end(), points[input]),
        nearest_edges.end());
  }
  for (hypervec::idx_t node = 0; node < count; ++node) {
    size_t level0_begin = 0;
    size_t level0_end = 0;
    index.hnsw.NeighborRange(node, 0, &level0_begin, &level0_end);
    const size_t node_end = index.hnsw.offsets[static_cast<size_t>(node) + 1];
    for (size_t offset = level0_end; offset < node_end; ++offset) {
      EXPECT_EQ(index.hnsw.neighbors[offset], neighbors_before[offset]);
    }
  }
}

TEST(IndexHNSWCorrectness, InitLevel0FromEntryPointsRollsBackRuntimeFailure) {
  constexpr hypervec::idx_t count = 12;
  const auto data = RandomVectors(count, 3, 2007);
  ThrowingReconstructStorage storage(3);
  hypervec::IndexHNSW index(&storage, 4);
  index.Add(count, data.data());
  std::vector<float> self_distances(static_cast<size_t>(count), 0.0F);
  std::vector<hypervec::idx_t> self_labels(static_cast<size_t>(count));
  for (hypervec::idx_t node = 0; node < count; ++node) {
    self_labels[static_cast<size_t>(node)] = node;
  }
  index.InitLevel0FromKnngraph(1, self_distances.data(), self_labels.data());
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  const std::array<hypervec::HNSW::storage_idx_t, 2> points = {0, 4};
  const std::array<hypervec::HNSW::storage_idx_t, 2> nearests = {1, 5};
  storage.fail_key = points[1];

  EXPECT_THROW(index.InitLevel0FromEntryPoints(static_cast<int>(points.size()),
                                               points.data(), nearests.data()),
               std::runtime_error);
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            neighbors_before);
}

TEST(IndexHNSWCorrectness, InitLevel0FromEntryPointsValidatesInputs) {
  constexpr hypervec::idx_t count = 8;
  const auto data = RandomVectors(count, 2, 2008);
  hypervec::IndexHNSWFlat index(2, 4);
  index.Add(count, data.data());
  hypervec::HNSW::storage_idx_t point = 0;
  hypervec::HNSW::storage_idx_t nearest = 1;

  EXPECT_NO_THROW(index.InitLevel0FromEntryPoints(0, nullptr, nullptr));
  EXPECT_THROW(index.InitLevel0FromEntryPoints(-1, nullptr, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.InitLevel0FromEntryPoints(1, nullptr, &nearest),
               hypervec::HypervecException);
  EXPECT_THROW(index.InitLevel0FromEntryPoints(1, &point, nullptr),
               hypervec::HypervecException);
  point = -1;
  EXPECT_THROW(index.InitLevel0FromEntryPoints(1, &point, &nearest),
               hypervec::HypervecException);
  point = 0;
  nearest = static_cast<hypervec::HNSW::storage_idx_t>(count);
  EXPECT_THROW(index.InitLevel0FromEntryPoints(1, &point, &nearest),
               hypervec::HypervecException);
  nearest = point;
  EXPECT_THROW(index.InitLevel0FromEntryPoints(1, &point, &nearest),
               hypervec::HypervecException);
  nearest = 1;
  index.hnsw.ef_construction = 0;
  EXPECT_THROW(index.InitLevel0FromEntryPoints(1, &point, &nearest),
               hypervec::HypervecException);
}

TEST(IndexHNSWCorrectness, LinkSingletonsRepairsZeroIncomingNodes) {
  constexpr hypervec::idx_t count = 12;
  constexpr hypervec::idx_t connected_count = 10;
  constexpr int k = 2;
  std::vector<float> data(static_cast<size_t>(count));
  for (hypervec::idx_t node = 0; node < count; ++node) {
    data[static_cast<size_t>(node)] = static_cast<float>(node);
  }
  hypervec::IndexHNSWFlat index(1, 4);
  index.Add(count, data.data());
  std::vector<float> distances(static_cast<size_t>(count) * k, 0.0F);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(count) * k);
  for (hypervec::idx_t node = 0; node < count; ++node) {
    const size_t row = static_cast<size_t>(node) * k;
    labels[row] = node;
    labels[row + 1] = node < connected_count ? (node + 1) % connected_count : 0;
    const float delta = data[static_cast<size_t>(node)] -
                        data[static_cast<size_t>(labels[row + 1])];
    distances[row + 1] = delta * delta;
  }
  index.InitLevel0FromKnngraph(k, distances.data(), labels.data());
  const auto incoming_before = Level0IncomingCounts(index);
  EXPECT_EQ(incoming_before[10], 0U);
  EXPECT_EQ(incoming_before[11], 0U);
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());

  index.LinkSingletons();
  const auto incoming_after = Level0IncomingCounts(index);
  EXPECT_GT(incoming_after[10], 0U);
  EXPECT_GT(incoming_after[11], 0U);
  for (hypervec::idx_t node = 0; node < count; ++node) {
    size_t level0_begin = 0;
    size_t level0_end = 0;
    index.hnsw.NeighborRange(node, 0, &level0_begin, &level0_end);
    const size_t node_end = index.hnsw.offsets[static_cast<size_t>(node) + 1];
    for (size_t offset = level0_end; offset < node_end; ++offset) {
      EXPECT_EQ(index.hnsw.neighbors[offset], neighbors_before[offset]);
    }
  }

  const std::vector<hypervec::HNSW::storage_idx_t> repaired_graph(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  index.LinkSingletons();
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            repaired_graph);
}

TEST(IndexHNSWCorrectness, LinkSingletonsConnectsAnEdgelessBaseLayer) {
  constexpr hypervec::idx_t count = 6;
  const auto data = RandomVectors(count, 3, 2009);
  hypervec::IndexHNSWFlat index(3, 4);
  index.Add(count, data.data());
  std::vector<float> self_distances(static_cast<size_t>(count), 0.0F);
  std::vector<hypervec::idx_t> self_labels(static_cast<size_t>(count));
  for (hypervec::idx_t node = 0; node < count; ++node) {
    self_labels[static_cast<size_t>(node)] = node;
  }
  index.InitLevel0FromKnngraph(1, self_distances.data(), self_labels.data());

  index.LinkSingletons();
  const auto incoming = Level0IncomingCounts(index);
  for (const size_t degree : incoming) {
    EXPECT_GT(degree, 0U);
  }

  hypervec::IndexHNSWFlat empty(3, 4);
  EXPECT_NO_THROW(empty.LinkSingletons());
  hypervec::IndexHNSWFlat one(3, 4);
  one.Add(1, data.data());
  EXPECT_NO_THROW(one.LinkSingletons());
}

TEST(IndexHNSWCorrectness, LinkSingletonsRollsBackRuntimeFailure) {
  constexpr hypervec::idx_t count = 8;
  constexpr int k = 2;
  const auto data = RandomVectors(count, 3, 2010);
  ThrowingReconstructStorage storage(3);
  hypervec::IndexHNSW index(&storage, 4);
  index.Add(count, data.data());
  std::vector<float> distances(static_cast<size_t>(count) * k, 0.0F);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(count) * k);
  for (hypervec::idx_t node = 0; node < count; ++node) {
    const size_t row = static_cast<size_t>(node) * k;
    labels[row] = node;
    labels[row + 1] = node + 1 < count ? (node + 1) % (count - 1) : 0;
  }
  index.InitLevel0FromKnngraph(k, distances.data(), labels.data());
  ASSERT_EQ(Level0IncomingCounts(index).back(), 0U);
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  storage.fail_key = count - 1;

  EXPECT_THROW(index.LinkSingletons(), std::runtime_error);
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            neighbors_before);
}

TEST(IndexHNSWCorrectness, PermuteEntriesRemapsFlatStorageAndGraph) {
  constexpr hypervec::idx_t d = 6;
  constexpr hypervec::idx_t count = 48;
  constexpr hypervec::idx_t query_count = 4;
  constexpr hypervec::idx_t k = 5;
  const auto data = RandomVectors(count, d, 2001);
  const auto permutation = MakePermutation(count);
  const auto inverse = InvertPermutation(permutation);

  hypervec::IndexHNSWFlat index(d, 8);
  index.Add(count, data.data());
  auto* flat = dynamic_cast<hypervec::IndexFlatL2*>(index.storage);
  ASSERT_NE(flat, nullptr);
  flat->SyncL2Norms();
  ASSERT_EQ(flat->cached_l2norms.size(), static_cast<size_t>(count));

  std::vector<float> distances_before(query_count * k);
  std::vector<hypervec::idx_t> labels_before(query_count * k);
  index.Search(query_count, data.data(), k, distances_before.data(),
               labels_before.data());

  const auto old_levels = index.hnsw.levels;
  const auto old_offsets = index.hnsw.offsets;
  const std::vector<hypervec::HNSW::storage_idx_t> old_neighbors(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());
  const auto old_entry_point = index.hnsw.entry_point;

  index.PermuteEntries(permutation.data());
  ExpectConsistentGraph(index);
  EXPECT_TRUE(flat->cached_l2norms.empty());
  EXPECT_EQ(index.hnsw.entry_point,
            inverse[static_cast<size_t>(old_entry_point)]);

  for (hypervec::idx_t new_id = 0; new_id < count; ++new_id) {
    const hypervec::idx_t old_id = permutation[static_cast<size_t>(new_id)];
    ExpectVector(index, new_id, data.data() + static_cast<size_t>(old_id * d));
    EXPECT_EQ(index.hnsw.levels[static_cast<size_t>(new_id)],
              old_levels[static_cast<size_t>(old_id)]);
    const size_t new_begin = index.hnsw.offsets[static_cast<size_t>(new_id)];
    const size_t new_end = index.hnsw.offsets[static_cast<size_t>(new_id) + 1];
    const size_t old_begin = old_offsets[static_cast<size_t>(old_id)];
    const size_t old_end = old_offsets[static_cast<size_t>(old_id) + 1];
    ASSERT_EQ(new_end - new_begin, old_end - old_begin);
    for (size_t offset = 0; offset < old_end - old_begin; ++offset) {
      const auto old_neighbor = old_neighbors[old_begin + offset];
      const auto expected_neighbor =
          old_neighbor < 0 ? old_neighbor
                           : static_cast<hypervec::HNSW::storage_idx_t>(
                                 inverse[static_cast<size_t>(old_neighbor)]);
      EXPECT_EQ(index.hnsw.neighbors[new_begin + offset], expected_neighbor);
    }
  }

  std::vector<float> distances_after(query_count * k);
  std::vector<hypervec::idx_t> labels_after(query_count * k);
  index.Search(query_count, data.data(), k, distances_after.data(),
               labels_after.data());
  for (size_t result = 0; result < labels_before.size(); ++result) {
    EXPECT_NEAR(distances_after[result], distances_before[result], 1e-5F);
    ASSERT_GE(labels_before[result], 0);
    EXPECT_EQ(labels_after[result],
              inverse[static_cast<size_t>(labels_before[result])]);
  }
}

TEST(IndexHNSWCorrectness, PermuteEntriesRejectsInvalidInputWithoutMutation) {
  constexpr hypervec::idx_t d = 4;
  constexpr hypervec::idx_t count = 16;
  const auto data = RandomVectors(count, d, 2002);
  hypervec::IndexHNSWFlat index(d, 4);
  index.Add(count, data.data());
  auto* flat = dynamic_cast<hypervec::IndexFlat*>(index.storage);
  ASSERT_NE(flat, nullptr);

  const std::vector<uint8_t> codes_before(
      flat->codes.data(), flat->codes.data() + flat->codes.size());
  const auto levels_before = index.hnsw.levels;
  const auto offsets_before = index.hnsw.offsets;
  const std::vector<hypervec::HNSW::storage_idx_t> neighbors_before(
      index.hnsw.neighbors.data(),
      index.hnsw.neighbors.data() + index.hnsw.neighbors.size());

  auto duplicate = MakePermutation(count);
  duplicate[1] = duplicate[0];
  EXPECT_THROW(index.PermuteEntries(duplicate.data()),
               hypervec::HypervecException);
  auto out_of_range = MakePermutation(count);
  out_of_range[0] = count;
  EXPECT_THROW(index.PermuteEntries(out_of_range.data()),
               hypervec::HypervecException);
  EXPECT_THROW(index.PermuteEntries(nullptr), hypervec::HypervecException);

  EXPECT_EQ(std::vector<uint8_t>(flat->codes.data(),
                                 flat->codes.data() + flat->codes.size()),
            codes_before);
  EXPECT_EQ(index.hnsw.levels, levels_before);
  EXPECT_EQ(index.hnsw.offsets, offsets_before);
  EXPECT_EQ(std::vector<hypervec::HNSW::storage_idx_t>(
                index.hnsw.neighbors.data(),
                index.hnsw.neighbors.data() + index.hnsw.neighbors.size()),
            neighbors_before);
}

TEST(IndexHNSWCorrectness, PermuteEntriesKeepsBuildScaffoldsAligned) {
  constexpr hypervec::idx_t d = 8;
  constexpr hypervec::idx_t initial_count = 64;
  constexpr hypervec::idx_t extra_count = 8;
  const auto data = RandomVectors(initial_count + extra_count, d, 2003);

  hypervec::IndexHNSWPQ pq(d, 2, 3, 8);
  pq.Train(initial_count, data.data());
  pq.Add(initial_count, data.data());
  ExpectCompressedPermutation(&pq, pq.raw_storage, data, initial_count,
                              extra_count);

  hypervec::IndexHNSWLVQ lvq(d, 4, 3, 8);
  lvq.Train(initial_count, data.data());
  lvq.Add(initial_count, data.data());
  ExpectCompressedPermutation(&lvq, lvq.raw_storage, data, initial_count,
                              extra_count);
}
