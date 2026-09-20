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
#include <index/idmap/index_id_map.h>
#include <persistence/index_clone.h>
#include <utils/log/exception.h>

#include <memory>
#include <vector>

namespace {

TEST(IndexClone, CopiesFlatDataAndOwnsIndependentStorage) {
  hypervec::IndexFlatL2 source(2);
  const std::vector<float> source_vectors = {0.0F, 0.0F, 2.0F, 2.0F};
  source.Add(2, source_vectors.data());

  std::unique_ptr<hypervec::Index> clone(hypervec::clone_index(&source));
  auto* cloned_flat = dynamic_cast<hypervec::IndexFlatL2*>(clone.get());
  ASSERT_NE(cloned_flat, nullptr);
  EXPECT_EQ(cloned_flat->n_total, 2);

  const std::vector<float> query = {1.9F, 2.1F};
  float source_distance = 0.0F;
  float clone_distance = 0.0F;
  hypervec::idx_t source_label = -1;
  hypervec::idx_t clone_label = -1;
  source.Search(1, query.data(), 1, &source_distance, &source_label);
  clone->Search(1, query.data(), 1, &clone_distance, &clone_label);
  EXPECT_EQ(clone_label, source_label);
  EXPECT_FLOAT_EQ(clone_distance, source_distance);

  const std::vector<float> clone_only_vector = {1.9F, 2.1F};
  clone->Add(1, clone_only_vector.data());
  EXPECT_EQ(clone->n_total, 3);
  EXPECT_EQ(source.n_total, 2);

  clone->Search(1, query.data(), 1, &clone_distance, &clone_label);
  source.Search(1, query.data(), 1, &source_distance, &source_label);
  EXPECT_EQ(clone_label, 2);
  EXPECT_EQ(source_label, 1);
}

TEST(IndexClone, PreservesIdMapDataAndExternalLabels) {
  auto storage = std::make_unique<hypervec::IndexFlatL2>(2);
  hypervec::IndexIDMap source(storage.release());
  source.own_fields = true;
  const std::vector<float> vectors = {0.0F, 0.0F, 3.0F, 3.0F};
  const std::vector<hypervec::idx_t> ids = {101, 909};
  source.AddWithIds(2, vectors.data(), ids.data());

  std::unique_ptr<hypervec::Index> clone(hypervec::clone_index(&source));
  auto* cloned_map = dynamic_cast<hypervec::IndexIDMap*>(clone.get());
  ASSERT_NE(cloned_map, nullptr);
  EXPECT_TRUE(cloned_map->own_fields);
  EXPECT_NE(cloned_map->index, source.index);
  EXPECT_EQ(cloned_map->rev_map, ids);
  EXPECT_EQ(cloned_map->to_internal(101), 0);
  EXPECT_EQ(cloned_map->to_internal(909), 1);

  const std::vector<float> query = {2.9F, 3.1F};
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  clone->Search(1, query.data(), 1, &distance, &label);
  EXPECT_EQ(label, 909);
}

TEST(IndexClone, HnswEntryPointPreservesGraphAndStorage) {
  hypervec::IndexHNSWFlat source(2, 4, hypervec::kMetricL2);
  const std::vector<float> vectors = {0.0F, 0.0F, 1.0F, 1.0F,
                                      2.0F, 2.0F, 3.0F, 3.0F};
  source.Add(4, vectors.data());

  std::unique_ptr<hypervec::IndexHNSW> clone(
      hypervec::clone_IndexHNSW(&source));
  ASSERT_NE(dynamic_cast<hypervec::IndexHNSWFlat*>(clone.get()), nullptr);
  EXPECT_EQ(clone->n_total, source.n_total);
  EXPECT_EQ(clone->hnsw.entry_point, source.hnsw.entry_point);
  EXPECT_EQ(clone->hnsw.levels, source.hnsw.levels);
  EXPECT_EQ(clone->hnsw.offsets, source.hnsw.offsets);
  EXPECT_EQ(clone->hnsw.neighbors, source.hnsw.neighbors);
  EXPECT_NE(clone->storage, source.storage);

  const std::vector<float> query = {2.1F, 1.9F};
  float source_distance = 0.0F;
  float clone_distance = 0.0F;
  hypervec::idx_t source_label = -1;
  hypervec::idx_t clone_label = -1;
  source.Search(1, query.data(), 1, &source_distance, &source_label);
  clone->Search(1, query.data(), 1, &clone_distance, &clone_label);
  EXPECT_EQ(clone_label, source_label);
  EXPECT_FLOAT_EQ(clone_distance, source_distance);
}

TEST(IndexClone, RejectsNullInput) {
  EXPECT_THROW(hypervec::clone_index(nullptr), hypervec::HypervecException);
  EXPECT_THROW(hypervec::clone_IndexHNSW(nullptr), hypervec::HypervecException);
}

}  // namespace
