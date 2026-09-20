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

#include <array>
#include <cstdint>
#include <memory>
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
