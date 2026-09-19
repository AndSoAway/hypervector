/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <quantization/rabitq/index_ivf_rabitq.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/random.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

std::vector<float> RandomVectors(hypervec::idx_t count,
                                 hypervec::idx_t dimension, int64_t seed) {
  hypervec::RandomGenerator random(seed);
  std::vector<float> vectors(static_cast<size_t>(count * dimension));
  for (float& value : vectors) {
    value = 6.0F * random.rand_float() - 3.0F;
  }
  return vectors;
}

std::vector<float> SeparatedVectors(hypervec::idx_t count,
                                    hypervec::idx_t dimension) {
  std::vector<float> vectors(static_cast<size_t>(count * dimension), 0.0F);
  for (hypervec::idx_t row = 0; row < count; ++row) {
    vectors[static_cast<size_t>(row * dimension + row % dimension)] =
        5.0F + static_cast<float>(row / dimension);
  }
  return vectors;
}

void ExpectSortedValid(const std::vector<float>& distances,
                       const std::vector<hypervec::idx_t>& labels,
                       hypervec::idx_t query_count,
                       hypervec::idx_t neighbor_count,
                       hypervec::idx_t database_size) {
  for (hypervec::idx_t query = 0; query < query_count; ++query) {
    for (hypervec::idx_t rank = 0; rank < neighbor_count; ++rank) {
      const size_t offset = static_cast<size_t>(query * neighbor_count + rank);
      EXPECT_GE(labels[offset], 0);
      EXPECT_LT(labels[offset], database_size);
      EXPECT_TRUE(std::isfinite(distances[offset]));
      EXPECT_GE(distances[offset], 0.0F);
      if (rank > 0) {
        EXPECT_GE(distances[offset], distances[offset - 1]);
      }
    }
  }
}

float RecallAtK(const std::vector<hypervec::idx_t>& actual,
                const std::vector<hypervec::idx_t>& expected,
                hypervec::idx_t query_count, hypervec::idx_t neighbor_count) {
  size_t hits = 0;
  for (hypervec::idx_t query = 0; query < query_count; ++query) {
    for (hypervec::idx_t expected_rank = 0; expected_rank < neighbor_count;
         ++expected_rank) {
      const hypervec::idx_t expected_id =
          expected[static_cast<size_t>(query * neighbor_count + expected_rank)];
      for (hypervec::idx_t actual_rank = 0; actual_rank < neighbor_count;
           ++actual_rank) {
        if (actual[static_cast<size_t>(query * neighbor_count + actual_rank)] ==
            expected_id) {
          ++hits;
          break;
        }
      }
    }
  }
  return static_cast<float>(hits) /
         static_cast<float>(query_count * neighbor_count);
}

}  // namespace

TEST(IndexIVFRaBitQ, TrainAddAndSearchUsesCommonIvfPipeline) {
  constexpr hypervec::idx_t dimension = 16;
  constexpr hypervec::idx_t database_size = 256;
  constexpr hypervec::idx_t query_count = 8;
  constexpr hypervec::idx_t neighbor_count = 5;
  const std::vector<float> database =
      RandomVectors(database_size, dimension, 101);
  const std::vector<float> queries = RandomVectors(query_count, dimension, 202);

  hypervec::IndexIVFRaBitQ index(dimension, 8, 42, 3);
  EXPECT_FALSE(index.is_trained);
  EXPECT_TRUE(index.GetCapabilities().requires_training);
  EXPECT_TRUE(index.GetCapabilities().supports_range_search);
  EXPECT_TRUE(index.GetCapabilities().supports_reconstruct);
  index.Train(database_size, database.data());
  index.Add(database_size, database.data());
  EXPECT_TRUE(index.is_trained);
  EXPECT_EQ(index.n_total, database_size);
  ASSERT_NE(index.rabitq, nullptr);
  EXPECT_EQ(index.invlists->code_size, index.rabitq->CodeSize());

  hypervec::IVFSearchParameters parameters;
  parameters.nprobe = 4;
  std::vector<float> distances(
      static_cast<size_t>(query_count * neighbor_count));
  std::vector<hypervec::idx_t> labels(distances.size());
  index.Search(query_count, queries.data(), neighbor_count, distances.data(),
               labels.data(), &parameters);
  ExpectSortedValid(distances, labels, query_count, neighbor_count,
                    database_size);
}

TEST(IndexIVFRaBitQ, ExactRowsRemainNearestWithAllListsProbed) {
  constexpr hypervec::idx_t dimension = 32;
  constexpr hypervec::idx_t count = 24;
  const std::vector<float> vectors = SeparatedVectors(count, dimension);
  hypervec::IndexIVFRaBitQ index(dimension, 4, 9001, 3);
  index.Train(count, vectors.data());
  index.Add(count, vectors.data());

  hypervec::IVFSearchParameters parameters;
  parameters.nprobe = index.nlist;
  for (hypervec::idx_t expected = 0; expected < count; ++expected) {
    float distance = -1.0F;
    hypervec::idx_t label = -1;
    index.Search(1, vectors.data() + expected * dimension, 1, &distance, &label,
                 &parameters);
    EXPECT_EQ(label, expected);
    EXPECT_NEAR(distance, 0.0F, 2e-4F);
  }
}

TEST(IndexIVFRaBitQ, OneBitEstimatesRetainUsefulRecall) {
  constexpr hypervec::idx_t dimension = 64;
  constexpr hypervec::idx_t database_size = 512;
  constexpr hypervec::idx_t query_count = 32;
  constexpr hypervec::idx_t neighbor_count = 10;
  const std::vector<float> database =
      RandomVectors(database_size, dimension, 606);
  const std::vector<float> queries = RandomVectors(query_count, dimension, 707);

  hypervec::IndexFlatL2 exact(dimension);
  exact.Add(database_size, database.data());
  std::vector<float> exact_distances(
      static_cast<size_t>(query_count * neighbor_count));
  std::vector<hypervec::idx_t> exact_labels(exact_distances.size());
  exact.Search(query_count, queries.data(), neighbor_count,
               exact_distances.data(), exact_labels.data());

  hypervec::IndexIVFRaBitQ approximate(dimension, 16, 808, 3);
  approximate.Train(database_size, database.data());
  approximate.Add(database_size, database.data());
  hypervec::IVFSearchParameters parameters;
  parameters.nprobe = approximate.nlist;
  std::vector<float> approximate_distances(exact_distances.size());
  std::vector<hypervec::idx_t> approximate_labels(exact_distances.size());
  approximate.Search(query_count, queries.data(), neighbor_count,
                     approximate_distances.data(), approximate_labels.data(),
                     &parameters);

  EXPECT_GT(
      RecallAtK(approximate_labels, exact_labels, query_count, neighbor_count),
      0.20F);
}

TEST(IndexIVFRaBitQ, ScannerSupportsResidualRawRangeAndExternalIds) {
  constexpr hypervec::idx_t dimension = 16;
  constexpr hypervec::idx_t count = 64;
  constexpr hypervec::idx_t target = 11;
  const std::vector<float> vectors = RandomVectors(count, dimension, 303);
  std::vector<hypervec::idx_t> ids(static_cast<size_t>(count));
  for (hypervec::idx_t i = 0; i < count; ++i) {
    ids[static_cast<size_t>(i)] = 1000 + i;
  }

  for (const bool by_residual : {true, false}) {
    hypervec::IndexIVFRaBitQ index(dimension, 4, 77, 2);
    index.by_residual = by_residual;
    index.Train(count, vectors.data());
    index.AddWithIds(count, vectors.data(), ids.data());

    hypervec::IDSelectorRange selector(ids[target], ids[target] + 1);
    hypervec::IVFSearchParameters parameters;
    parameters.nprobe = index.nlist;
    parameters.sel = &selector;
    const float* query = vectors.data() + target * dimension;
    float distance = -1.0F;
    hypervec::idx_t label = -1;
    index.Search(1, query, 1, &distance, &label, &parameters);
    EXPECT_EQ(label, ids[target]) << by_residual;

    hypervec::RangeSearchResult result(1);
    index.RangeSearch(1, query, (std::numeric_limits<float>::infinity)(),
                      &result, &parameters);
    ASSERT_EQ(result.lims[1], 1U) << by_residual;
    EXPECT_EQ(result.labels[0], ids[target]) << by_residual;
    EXPECT_FLOAT_EQ(result.distances[0], distance) << by_residual;
  }
}

TEST(IndexIVFRaBitQ, ReconstructsAndSupportsRepeatedAdds) {
  constexpr hypervec::idx_t dimension = 16;
  const std::vector<float> vectors = RandomVectors(96, dimension, 404);
  hypervec::IndexIVFRaBitQ index(dimension, 4);
  index.Train(96, vectors.data());
  index.Add(48, vectors.data());
  index.Add(48, vectors.data() + 48 * dimension);
  EXPECT_EQ(index.n_total, 96);

  std::vector<float> reconstructed(static_cast<size_t>(dimension));
  index.Reconstruct(53, reconstructed.data());
  for (float value : reconstructed) {
    EXPECT_TRUE(std::isfinite(value));
  }
  EXPECT_THROW(index.Reconstruct(200, reconstructed.data()),
               hypervec::HypervecException);
}

TEST(IndexIVFRaBitQ, RejectsInvalidLifecycleWithoutMutation) {
  EXPECT_THROW(
      hypervec::IndexIVFRaBitQ(8, 2, 1, 3, hypervec::kMetricInnerProduct),
      hypervec::HypervecException);
  EXPECT_THROW(hypervec::IndexIVFRaBitQ(8, 2, 1, 0),
               hypervec::HypervecException);

  const std::vector<float> vectors = RandomVectors(32, 8, 505);
  hypervec::IndexIVFRaBitQ index(8, 2);
  EXPECT_THROW(index.Add(1, vectors.data()), hypervec::HypervecException);
  std::vector<float> invalid_training = vectors;
  invalid_training[3] = (std::numeric_limits<float>::quiet_NaN)();
  EXPECT_THROW(index.Train(32, invalid_training.data()),
               hypervec::HypervecException);
  EXPECT_FALSE(index.is_trained);
  index.Train(32, vectors.data());
  index.Add(2, vectors.data());
  const hypervec::idx_t original_total = index.n_total;
  std::vector<float> invalid(vectors.begin(), vectors.begin() + 8);
  invalid[3] = (std::numeric_limits<float>::quiet_NaN)();
  EXPECT_THROW(index.Add(1, invalid.data()), hypervec::HypervecException);
  EXPECT_THROW(index.Add(-1, vectors.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, original_total);
}
