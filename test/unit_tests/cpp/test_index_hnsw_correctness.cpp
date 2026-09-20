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
