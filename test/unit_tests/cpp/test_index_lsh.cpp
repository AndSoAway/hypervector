/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/lsh/index_lsh.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

hypervec::LSHIndexOptions ExhaustiveOptions() {
  hypervec::LSHIndexOptions options;
  options.table_count = 4;
  options.bits_per_table = 1;
  options.probe_count = 2;
  options.random_seed = 42;
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
  EXPECT_EQ(actual_distances, expected_distances);
}

}  // namespace

TEST(IndexLSH, ExhaustiveProbesPreserveInnerProductResults) {
  const std::vector<float> database = {
      1.0F, 0.0F, 0.0F, 2.0F, 2.0F, 1.0F, 3.0F, 4.0F, 1.0F, 3.0F, 6.0F, 2.0F,
  };
  const std::vector<float> queries = {1.0F, 1.0F, 2.0F, 0.5F};
  hypervec::IndexFlatIP expected(2);
  expected.Add(6, database.data());
  hypervec::IndexLSH index(2, hypervec::kMetricInnerProduct,
                           ExhaustiveOptions());

  index.Add(3, database.data());
  index.Add(3, database.data() + 6);

  EXPECT_EQ(index.n_total, 6);
  EXPECT_EQ(index.CodeStore().Size(), 6);
  ExpectSameSearch(expected, index, queries, 3);
  std::array<float, 2> reconstructed;
  index.Reconstruct(4, reconstructed.data());
  EXPECT_EQ(reconstructed, (std::array<float, 2>{1.0F, 3.0F}));
  const auto capabilities = index.GetCapabilities();
  EXPECT_TRUE(capabilities.supports_reconstruct);
  EXPECT_FALSE(capabilities.requires_training);
}

TEST(IndexLSH, DeterministicHyperplanesAndAdaptiveProbes) {
  hypervec::LSHIndexOptions options;
  options.table_count = 3;
  options.bits_per_table = 5;
  options.probe_count = 3;
  options.random_seed = 123;
  hypervec::IndexLSH first(4, hypervec::kMetricInnerProduct, options);
  hypervec::IndexLSH second(4, hypervec::kMetricInnerProduct, options);
  EXPECT_EQ(first.Hyperplanes(), second.Hyperplanes());

  options.random_seed = 124;
  hypervec::IndexLSH different(4, hypervec::kMetricInnerProduct, options);
  EXPECT_NE(first.Hyperplanes(), different.Hyperplanes());

  const std::vector<float> database = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  };
  first.Add(4, database.data());
  std::array<float, 4> distances;
  std::array<hypervec::idx_t, 4> labels;
  hypervec::SearchParametersLSH params;
  params.probe_count = 6;
  first.Search(1, database.data(), 4, distances.data(), labels.data(), &params);
  EXPECT_EQ(labels[0], 0);
  EXPECT_FLOAT_EQ(distances[0], 1.0F);
}

TEST(IndexLSH, SelectorAndCandidateLimitControlExactReranking) {
  const std::vector<float> database = {1, 0, 2, 0, 3, 0, 4, 0};
  hypervec::IndexLSH index(2, hypervec::kMetricInnerProduct,
                           ExhaustiveOptions());
  index.Add(4, database.data());

  hypervec::IDSelectorRange selector(1, 4);
  hypervec::SearchParametersLSH params;
  params.probe_count = 2;
  params.candidate_limit = 2;
  params.sel = &selector;
  std::array<float, 3> distances;
  std::array<hypervec::idx_t, 3> labels;
  index.Search(1, database.data(), 3, distances.data(), labels.data(), &params);
  EXPECT_EQ(labels[0], 2);
  EXPECT_EQ(labels[1], 1);
  EXPECT_EQ(labels[2], -1);
  EXPECT_TRUE(std::isinf(distances[2]));
  EXPECT_LT(distances[2], 0.0F);
}

TEST(IndexLSH, RuntimeZeroCandidateLimitDisablesConfiguredLimit) {
  hypervec::LSHIndexOptions options = ExhaustiveOptions();
  options.candidate_limit = 1;
  const std::vector<float> database = {1, 0, 2, 0, 3, 0};
  hypervec::IndexLSH index(2, hypervec::kMetricInnerProduct, options);
  index.Add(3, database.data());
  std::array<float, 3> distances;
  std::array<hypervec::idx_t, 3> labels;

  hypervec::SearchParametersLSH inherited;
  index.Search(1, database.data(), 3, distances.data(), labels.data(),
               &inherited);
  EXPECT_EQ(labels[1], -1);

  hypervec::SearchParametersLSH unlimited;
  unlimited.candidate_limit = 0;
  index.Search(1, database.data(), 3, distances.data(), labels.data(),
               &unlimited);
  EXPECT_EQ(labels, (std::array<hypervec::idx_t, 3>{2, 1, 0}));
}

TEST(IndexLSH, FailedAddIsTransactionalAndResetKeepsHashFamily) {
  hypervec::IndexLSH index(2, hypervec::kMetricInnerProduct,
                           ExhaustiveOptions());
  const std::array<float, 2> valid = {1.0F, 2.0F};
  index.Add(1, valid.data());
  const std::vector<float> hyperplanes = index.Hyperplanes();
  const std::array<float, 2> invalid = {std::numeric_limits<float>::quiet_NaN(),
                                        3.0F};
  EXPECT_THROW(index.Add(1, invalid.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(index.CodeStore().Size(), 1);

  index.Reset();
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.CodeStore().Size(), 0);
  EXPECT_EQ(index.Hyperplanes(), hyperplanes);
  index.Add(1, valid.data());
  EXPECT_EQ(index.n_total, 1);
}

TEST(IndexLSH, ValidatesOptionsMetricsAndInputs) {
  hypervec::LSHIndexOptions options = ExhaustiveOptions();
  options.table_count = 0;
  EXPECT_THROW((hypervec::IndexLSH(2, hypervec::kMetricInnerProduct, options)),
               hypervec::HypervecException);
  options = ExhaustiveOptions();
  options.bits_per_table = 64;
  EXPECT_THROW((hypervec::IndexLSH(2, hypervec::kMetricInnerProduct, options)),
               hypervec::HypervecException);
  options = ExhaustiveOptions();
  options.probe_count = 3;
  EXPECT_THROW((hypervec::IndexLSH(2, hypervec::kMetricInnerProduct, options)),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::IndexLSH(2, hypervec::kMetricL2)),
               hypervec::HypervecException);

  hypervec::IndexLSH index(2, hypervec::kMetricInnerProduct,
                           ExhaustiveOptions());
  std::array<float, 2> vector = {1.0F, 2.0F};
  EXPECT_THROW(index.Add(-1, nullptr), hypervec::HypervecException);
  EXPECT_THROW(index.Add(1, nullptr), hypervec::HypervecException);
  index.Add(1, vector.data());
  EXPECT_THROW(index.Train(1, vector.data()), hypervec::HypervecException);
  std::array<float, 1> distances;
  std::array<hypervec::idx_t, 1> labels;
  EXPECT_THROW(index.Search(1, nullptr, 1, distances.data(), labels.data()),
               hypervec::HypervecException);
  hypervec::SearchParametersLSH params;
  params.probe_count = 3;
  EXPECT_THROW(index.Search(1, vector.data(), 1, distances.data(),
                            labels.data(), &params),
               hypervec::HypervecException);
  EXPECT_THROW(index.Reconstruct(1, vector.data()),
               hypervec::HypervecException);
}
