/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/diskann/index_diskann.h>
#include <index/flat/index_flat.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace {

hypervec::DiskAnnIndexOptions ExhaustiveOptions() {
  hypervec::DiskAnnIndexOptions options;
  options.max_degree = 8;
  options.build_search_width = 8;
  options.candidate_pool_size = 16;
  options.build_passes = 2;
  options.random_seed = 42;
  options.search_width = 16;
  options.check_relative_distance = false;
  options.page_size = 64;
  options.cache_capacity_pages = 1;
  return options;
}

void ExpectSameSearch(const hypervec::Index& expected,
                      const hypervec::Index& actual,
                      const std::vector<float>& queries, hypervec::idx_t k) {
  const hypervec::idx_t count =
      static_cast<hypervec::idx_t>(queries.size()) / expected.d;
  std::vector<float> expected_distances(static_cast<size_t>(count * k));
  std::vector<float> actual_distances(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> expected_labels(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> actual_labels(static_cast<size_t>(count * k));
  expected.Search(count, queries.data(), k, expected_distances.data(),
                  expected_labels.data());
  actual.Search(count, queries.data(), k, actual_distances.data(),
                actual_labels.data());

  EXPECT_EQ(actual_labels, expected_labels);
  for (size_t result = 0; result < actual_distances.size(); ++result) {
    EXPECT_FLOAT_EQ(actual_distances[result], expected_distances[result]);
  }
}

}  // namespace

TEST(IndexDiskANN, BuildsPagedVamanaAndMatchesExhaustiveSearch) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  const std::vector<float> queries = {4.0F, 16.0F};
  hypervec::IndexFlatL2 expected(1);
  expected.Add(static_cast<hypervec::idx_t>(database.size()), database.data());
  hypervec::IndexDiskANNFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());

  index.Build(static_cast<hypervec::idx_t>(database.size()), database.data());

  EXPECT_EQ(index.n_total, 6);
  EXPECT_EQ(index.CodeStore().Size(), 6);
  ASSERT_NE(index.Layout(), nullptr);
  ASSERT_NE(index.Graph(), nullptr);
  EXPECT_EQ(index.Layout()->NodeCount(), 6U);
  EXPECT_EQ(index.Graph()->NodeCount(), 6U);
  EXPECT_EQ(index.BuildStats().passes_completed, 2U);
  EXPECT_EQ(index.ReadStats().read_operations, 0U);
  ExpectSameSearch(expected, index, queries, 3);
  EXPECT_GT(index.ReadStats().read_operations, 0U);
  EXPECT_GT(index.CacheStats().pages_loaded, 0U);
}

TEST(IndexDiskANN, RuntimeParametersFilterWithoutBlockingNavigation) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  hypervec::IndexDiskANNFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());
  index.Build(6, database.data());
  hypervec::IDSelectorRange selector(2, 5);
  hypervec::SearchParametersDiskANN params;
  params.search_width = 8;
  params.check_relative_distance = false;
  params.sel = &selector;
  std::array<float, 4> distances;
  std::array<hypervec::idx_t, 4> labels;
  const float query = 1.0F;

  index.Search(1, &query, 4, distances.data(), labels.data(), &params);

  EXPECT_EQ(labels, (std::array<hypervec::idx_t, 4>{2, 3, 4, -1}));
  EXPECT_TRUE(std::isinf(distances[3]));
  params.search_width = 0;
  EXPECT_THROW(
      index.Search(1, &query, 1, distances.data(), labels.data(), &params),
      hypervec::HypervecException);
}

TEST(IndexDiskANN, ReconstructsExactVectorsAndSupportsResetRebuild) {
  const std::array<float, 4> database = {1.0F, 2.0F, 3.0F, 4.0F};
  hypervec::DiskAnnIndexOptions options = ExhaustiveOptions();
  options.max_degree = 1;
  hypervec::IndexDiskANNFlat index(2, hypervec::kMetricL2, options);

  EXPECT_THROW(index.Add(2, database.data()), hypervec::HypervecException);
  index.Build(2, database.data());
  EXPECT_THROW(index.Build(2, database.data()), hypervec::HypervecException);
  std::array<float, 2> reconstructed;
  index.Reconstruct(1, reconstructed.data());
  EXPECT_EQ(reconstructed, (std::array<float, 2>{3.0F, 4.0F}));
  EXPECT_TRUE(index.GetCapabilities().supports_reconstruct);

  index.ResetIOStats();
  EXPECT_EQ(index.ReadStats().read_operations, 0U);
  EXPECT_EQ(index.CacheStats().pages_loaded, 0U);
  index.Reset();
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.Layout(), nullptr);
  EXPECT_EQ(index.Graph(), nullptr);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(index.BuildStats().nodes_processed, 0U);
  index.Build(2, database.data(), 1, database.data());
  EXPECT_EQ(index.n_total, 2);
}

TEST(IndexDiskANN, ValidatesConstructionBuildAndSearchInputs) {
  EXPECT_THROW((hypervec::IndexDiskANNFlat(2, hypervec::kMetricInnerProduct,
                                           ExhaustiveOptions())),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::IndexDiskANN(nullptr, ExhaustiveOptions())),
               hypervec::HypervecException);
  hypervec::DiskAnnIndexOptions bad = ExhaustiveOptions();
  bad.search_width = 0;
  EXPECT_THROW((hypervec::IndexDiskANNFlat(2, hypervec::kMetricL2, bad)),
               hypervec::HypervecException);
  bad = ExhaustiveOptions();
  bad.cache_capacity_pages = 0;
  EXPECT_THROW((hypervec::IndexDiskANNFlat(2, hypervec::kMetricL2, bad)),
               hypervec::HypervecException);
  bad = ExhaustiveOptions();
  bad.page_size = 16;
  EXPECT_THROW((hypervec::IndexDiskANNFlat(2, hypervec::kMetricL2, bad)),
               hypervec::HypervecException);

  hypervec::IndexDiskANNFlat index(2, hypervec::kMetricL2, ExhaustiveOptions());
  const std::array<float, 4> database = {1.0F, 2.0F, 3.0F, 4.0F};
  EXPECT_THROW(index.Build(0, database.data()), hypervec::HypervecException);
  EXPECT_THROW(index.Build(2, nullptr), hypervec::HypervecException);
  EXPECT_THROW(index.Build(2, database.data(), -1, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.Build(2, database.data(), 1, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.Search(-1, nullptr, 1, nullptr, nullptr),
               hypervec::HypervecException);

  float distance = 0.0F;
  hypervec::idx_t label = 0;
  index.Search(1, database.data(), 1, &distance, &label);
  EXPECT_TRUE(std::isinf(distance));
  EXPECT_EQ(label, -1);
}

TEST(IndexDiskANN, FailedBuildDoesNotPublishPartialStorage) {
  const std::array<float, 3> database = {
      0.0F, (std::numeric_limits<float>::quiet_NaN)(), 2.0F};
  hypervec::IndexDiskANNFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());

  EXPECT_THROW(index.Build(3, database.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.CodeStore().Size(), 0);
  EXPECT_EQ(index.Layout(), nullptr);
  EXPECT_EQ(index.Graph(), nullptr);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(index.BuildStats().nodes_processed, 0U);
}
