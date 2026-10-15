/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/graph/graph_validation.h>
#include <index/vamana/index_vamana.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace {

hypervec::VamanaIndexOptions ExhaustiveOptions() {
  hypervec::VamanaIndexOptions options;
  options.max_degree = 8;
  options.build_search_width = 8;
  options.candidate_pool_size = 16;
  options.alpha = 1.2F;
  options.build_passes = 2;
  options.random_seed = 42;
  options.search_width = 16;
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

TEST(IndexVamana, FlatL2BuildsFixedGraphAndMatchesExhaustiveSearch) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  const std::vector<float> queries = {4.0F, 16.0F};
  hypervec::IndexFlatL2 expected(1);
  expected.Add(static_cast<hypervec::idx_t>(database.size()), database.data());
  hypervec::IndexVamanaFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());

  index.Build(static_cast<hypervec::idx_t>(database.size()), database.data());

  EXPECT_EQ(index.n_total, 6);
  EXPECT_EQ(index.CodeStore().Size(), 6);
  EXPECT_EQ(index.Graph().NodeCount(), 6U);
  EXPECT_EQ(index.Graph().MaxDegree(), ExhaustiveOptions().max_degree);
  EXPECT_EQ(index.EntryPoint(), 3);
  EXPECT_EQ(index.BuildStats().passes_completed, 2U);
  EXPECT_EQ(index.BuildStats().nodes_processed, 12U);
  const auto report =
      hypervec::ValidateGraph(index.Graph(), index.EntryPoint());
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, 6U);
  ExpectSameSearch(expected, index, queries, 3);
}

TEST(IndexVamana, BuildsDuplicateVectorsWithoutBreakingReachability) {
  std::vector<float> database(20, 0.0F);
  hypervec::VamanaIndexOptions options = ExhaustiveOptions();
  options.max_degree = 4;
  options.build_search_width = 8;
  options.candidate_pool_size = 12;
  hypervec::IndexVamanaFlat index(1, hypervec::kMetricL2, options);

  index.Build(static_cast<hypervec::idx_t>(database.size()), database.data());

  const auto report =
      hypervec::ValidateGraph(index.Graph(), index.EntryPoint());
  EXPECT_EQ(report.reachable_nodes, database.size());
}

TEST(IndexVamana, ApproximateGraphRetainsHighRecall) {
  constexpr size_t kCount = 128;
  constexpr hypervec::idx_t kQueryCount = 16;
  constexpr hypervec::idx_t k = 5;
  std::vector<float> database(kCount * 2);
  for (size_t node = 0; node < kCount; ++node) {
    database[node * 2] = static_cast<float>((node * 37) % kCount);
    database[node * 2 + 1] =
        static_cast<float>((node * 53) % (kCount - 1)) * 0.01F;
  }
  std::vector<float> queries(static_cast<size_t>(kQueryCount) * 2);
  for (hypervec::idx_t query = 0; query < kQueryCount; ++query) {
    const size_t source = static_cast<size_t>(query * 7) % kCount;
    queries[static_cast<size_t>(query) * 2] = database[source * 2] + 0.2F;
    queries[static_cast<size_t>(query) * 2 + 1] =
        database[source * 2 + 1] - 0.15F;
  }

  hypervec::IndexFlatL2 expected(2);
  expected.Add(kCount, database.data());
  hypervec::VamanaIndexOptions options;
  options.max_degree = 8;
  options.build_search_width = 20;
  options.candidate_pool_size = 40;
  options.random_seed = 42;
  options.search_width = 24;
  hypervec::IndexVamanaFlat index(2, hypervec::kMetricL2, options);
  index.Build(kCount, database.data());

  std::vector<float> expected_distances(kQueryCount * k);
  std::vector<float> actual_distances(kQueryCount * k);
  std::vector<hypervec::idx_t> expected_labels(kQueryCount * k);
  std::vector<hypervec::idx_t> actual_labels(kQueryCount * k);
  expected.Search(kQueryCount, queries.data(), k, expected_distances.data(),
                  expected_labels.data());
  index.Search(kQueryCount, queries.data(), k, actual_distances.data(),
               actual_labels.data());

  size_t matches = 0;
  for (hypervec::idx_t query = 0; query < kQueryCount; ++query) {
    const auto expected_begin =
        expected_labels.begin() + static_cast<size_t>(query * k);
    for (hypervec::idx_t result = 0; result < k; ++result) {
      const hypervec::idx_t actual =
          actual_labels[static_cast<size_t>(query * k + result)];
      matches +=
          static_cast<size_t>(std::find(expected_begin, expected_begin + k,
                                        actual) != expected_begin + k);
    }
  }
  EXPECT_GE(static_cast<double>(matches) / static_cast<double>(kQueryCount * k),
            0.90);
}

TEST(IndexVamana, RuntimeParametersFilterResults) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  hypervec::IndexVamanaFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());
  index.Build(6, database.data());

  hypervec::IDSelectorRange selector(2, 5);
  hypervec::SearchParametersVamana params;
  params.search_width = 8;
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

TEST(IndexVamana, EnforcesStaticLifecycleAndSupportsResetRebuild) {
  const std::array<float, 4> database = {1.0F, 2.0F, 3.0F, 4.0F};
  hypervec::IndexVamanaFlat index(2, hypervec::kMetricL2, ExhaustiveOptions());

  EXPECT_THROW(index.Add(2, database.data()), hypervec::HypervecException);
  index.Build(2, database.data());
  EXPECT_THROW(index.Build(2, database.data()), hypervec::HypervecException);
  std::array<float, 2> reconstructed;
  index.Reconstruct(1, reconstructed.data());
  EXPECT_EQ(reconstructed, (std::array<float, 2>{3.0F, 4.0F}));
  EXPECT_TRUE(index.GetCapabilities().supports_reconstruct);

  index.Reset();
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.Graph().NodeCount(), 0U);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(index.BuildStats().nodes_processed, 0U);
  index.Build(2, database.data(), 1, database.data());
  EXPECT_EQ(index.n_total, 2);
}

TEST(IndexVamana, RestoreStateValidatesBeforeReplacingLiveIndex) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  const std::vector<float> queries = {4.0F, 16.0F};
  const hypervec::VamanaIndexOptions options = ExhaustiveOptions();
  hypervec::IndexVamanaFlat source(1, hypervec::kMetricL2, options);
  source.Build(static_cast<hypervec::idx_t>(database.size()), database.data());
  hypervec::IndexVamanaFlat restored(1, hypervec::kMetricL2, options);

  restored.RestoreState(source.CodeStore(), source.Graph(),
                        source.EntryPoint());

  EXPECT_EQ(restored.n_total, source.n_total);
  EXPECT_EQ(restored.BuildStats().nodes_processed, 0U);
  ExpectSameSearch(source, restored, queries, 3);

  hypervec::FixedDegreeGraph unreachable(database.size(), options.max_degree);
  EXPECT_THROW(restored.RestoreState(source.CodeStore(), unreachable,
                                     source.EntryPoint()),
               hypervec::HypervecException);
  EXPECT_EQ(restored.n_total, source.n_total);
  ExpectSameSearch(source, restored, queries, 3);
}

TEST(IndexVamana, ValidatesConstructionBuildAndSearchInputs) {
  EXPECT_THROW((hypervec::IndexVamanaFlat(2, hypervec::kMetricInnerProduct,
                                          ExhaustiveOptions())),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::IndexVamana(nullptr, ExhaustiveOptions())),
               hypervec::HypervecException);
  hypervec::VamanaIndexOptions zero_search = ExhaustiveOptions();
  zero_search.search_width = 0;
  EXPECT_THROW((hypervec::IndexVamanaFlat(2, hypervec::kMetricL2, zero_search)),
               hypervec::HypervecException);

  hypervec::IndexVamanaFlat index(2, hypervec::kMetricL2, ExhaustiveOptions());
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

TEST(IndexVamana, FailedBuildDoesNotPublishPartialState) {
  const std::array<float, 3> database = {
      0.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
  hypervec::IndexVamanaFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());

  EXPECT_THROW(index.Build(3, database.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.CodeStore().Size(), 0);
  EXPECT_EQ(index.Graph().NodeCount(), 0U);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(index.BuildStats().nodes_processed, 0U);
}
