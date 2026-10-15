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
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>

TEST(IndexCapabilities, FlatReportsMutableExactOperations) {
  hypervec::IndexFlatL2 index(4);
  const auto capabilities = index.GetCapabilities();

  EXPECT_FALSE(capabilities.requires_training);
  EXPECT_FALSE(capabilities.supports_add_with_ids);
  EXPECT_TRUE(capabilities.supports_remove_ids);
  EXPECT_TRUE(capabilities.supports_range_search);
  EXPECT_TRUE(capabilities.supports_reconstruct);
  EXPECT_TRUE(capabilities.supports_merge);
}

TEST(IndexCapabilities, FlatRangeSearchSupportsExtraMetrics) {
  hypervec::IndexFlat index(2, hypervec::kMetricL1);
  const float database[] = {0.0F, 0.0F, 2.0F, 0.0F};
  const float query[] = {0.0F, 0.0F};
  index.Add(2, database);

  hypervec::RangeSearchResult result(1);
  index.RangeSearch(1, query, 1.5F, &result);

  ASSERT_EQ(result.lims[1], 1U);
  EXPECT_EQ(result.labels[0], 0);
  EXPECT_FLOAT_EQ(result.distances[0], 0.0F);
}

TEST(IndexCapabilities, IvfFlatRejectsUnsupportedMetrics) {
  EXPECT_THROW((hypervec::IndexIVFFlat(4, 2, hypervec::kMetricL1)),
               hypervec::HypervecException);
}

TEST(IndexCapabilities, FlatQuantizersReportTrainingAndReconstruction) {
  hypervec::IndexPQ pq(4, 2, 2);
  hypervec::IndexLVQ lvq(4, 2);

  for (const hypervec::Index* index : {static_cast<hypervec::Index*>(&pq),
                                       static_cast<hypervec::Index*>(&lvq)}) {
    const auto capabilities = index->GetCapabilities();
    EXPECT_TRUE(capabilities.requires_training);
    EXPECT_TRUE(capabilities.supports_reconstruct);
    EXPECT_FALSE(capabilities.supports_add_with_ids);
    EXPECT_FALSE(capabilities.supports_remove_ids);
    EXPECT_FALSE(capabilities.supports_range_search);
    EXPECT_FALSE(capabilities.supports_merge);
  }
}

TEST(IndexCapabilities, IvfVariantsDistinguishRangeSearchSupport) {
  hypervec::IndexIVFFlat flat(4, 2);
  hypervec::IndexIVFPQ pq(4, 2, 2, 2);
  hypervec::IndexIVFLVQ lvq(4, 2, 2);

  const auto flat_capabilities = flat.GetCapabilities();
  EXPECT_TRUE(flat_capabilities.requires_training);
  EXPECT_TRUE(flat_capabilities.supports_add_with_ids);
  EXPECT_TRUE(flat_capabilities.supports_range_search);
  EXPECT_TRUE(flat_capabilities.supports_reconstruct);

  const auto pq_capabilities = pq.GetCapabilities();
  EXPECT_TRUE(pq_capabilities.requires_training);
  EXPECT_TRUE(pq_capabilities.supports_add_with_ids);
  EXPECT_TRUE(pq_capabilities.supports_range_search);
  EXPECT_TRUE(pq_capabilities.supports_reconstruct);

  const auto lvq_capabilities = lvq.GetCapabilities();
  EXPECT_TRUE(lvq_capabilities.requires_training);
  EXPECT_TRUE(lvq_capabilities.supports_add_with_ids);
  EXPECT_TRUE(lvq_capabilities.supports_range_search);
  EXPECT_TRUE(lvq_capabilities.supports_reconstruct);
}

TEST(IndexCapabilities, HnswComposesStorageRequirements) {
  hypervec::IndexHNSWFlat flat(4, 8);
  hypervec::IndexHNSWPQ pq(4, 2, 2, 8);
  hypervec::IndexHNSWLVQ lvq(4, 2, 8);

  const auto flat_capabilities = flat.GetCapabilities();
  EXPECT_FALSE(flat_capabilities.requires_training);
  EXPECT_TRUE(flat_capabilities.supports_range_search);
  EXPECT_TRUE(flat_capabilities.supports_reconstruct);

  for (const hypervec::Index* index : {static_cast<hypervec::Index*>(&pq),
                                       static_cast<hypervec::Index*>(&lvq)}) {
    const auto capabilities = index->GetCapabilities();
    EXPECT_TRUE(capabilities.requires_training);
    EXPECT_TRUE(capabilities.supports_range_search);
    EXPECT_TRUE(capabilities.supports_reconstruct);
    EXPECT_FALSE(capabilities.supports_add_with_ids);
    EXPECT_FALSE(capabilities.supports_remove_ids);
  }
}

TEST(IndexCapabilities, IdMapAddsExternalIdsAndComposesStorageOperations) {
  hypervec::IndexFlatL2 flat_storage(4);
  hypervec::IndexIDMap flat(&flat_storage);
  const auto flat_capabilities = flat.GetCapabilities();
  EXPECT_TRUE(flat_capabilities.supports_add_with_ids);
  EXPECT_TRUE(flat_capabilities.supports_remove_ids);
  EXPECT_TRUE(flat_capabilities.supports_range_search);
  EXPECT_TRUE(flat_capabilities.supports_reconstruct);
  EXPECT_TRUE(flat_capabilities.supports_merge);

  hypervec::IndexPQ pq_storage(4, 2, 2);
  hypervec::IndexIDMap pq(&pq_storage);
  const auto pq_capabilities = pq.GetCapabilities();
  EXPECT_TRUE(pq_capabilities.requires_training);
  EXPECT_TRUE(pq_capabilities.supports_add_with_ids);
  EXPECT_FALSE(pq_capabilities.supports_remove_ids);
  EXPECT_FALSE(pq_capabilities.supports_range_search);
  EXPECT_TRUE(pq_capabilities.supports_reconstruct);
  EXPECT_FALSE(pq_capabilities.supports_merge);
}

TEST(IndexCapabilities, DefaultIsConservativeForOptionalOperations) {
  class MinimalIndex final : public hypervec::Index {
   public:
    void Add(hypervec::idx_t, const float*) final {}
    void Search(hypervec::idx_t, const float*, hypervec::idx_t, float*,
                hypervec::idx_t*,
                const hypervec::SearchParameters*) const final {}
    void Reset() final {}
  } index;

  const auto capabilities = index.GetCapabilities();
  EXPECT_FALSE(capabilities.requires_training);
  EXPECT_FALSE(capabilities.supports_add_with_ids);
  EXPECT_FALSE(capabilities.supports_remove_ids);
  EXPECT_FALSE(capabilities.supports_range_search);
  EXPECT_FALSE(capabilities.supports_reconstruct);
  EXPECT_FALSE(capabilities.supports_merge);
}
