/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/refine/index_rerank.h>
#include <utils/log/exception.h>

#include <array>

namespace {

class DeliberatelyMisranked final : public hypervec::Index {
 public:
  explicit DeliberatelyMisranked(hypervec::MetricType metric)
      : Index(1, metric) {
    n_total = 3;
  }

  void Add(hypervec::idx_t, const float*) override {}
  void Reset() override {}

  void Search(hypervec::idx_t n, const float*, hypervec::idx_t k,
              float* distances, hypervec::idx_t* labels,
              const hypervec::SearchParameters*) const override {
    for (hypervec::idx_t row = 0; row < n; ++row) {
      for (hypervec::idx_t i = 0; i < k; ++i) {
        labels[row * k + i] = i;
        distances[row * k + i] = static_cast<float>(i);
      }
    }
  }
};

}  // namespace

TEST(IndexRerank, CorrectsApproximateOrderForL2AndInnerProduct) {
  const std::array<float, 3> base = {2.0F, 9.0F, 4.0F};
  const std::array<float, 2> queries = {3.0F, 8.0F};
  for (const auto metric :
       {hypervec::kMetricL2, hypervec::kMetricInnerProduct}) {
    DeliberatelyMisranked approximate(metric);
    hypervec::IndexRerank refined(approximate, base.data(), 3, 3);
    std::array<float, 4> distances{};
    std::array<hypervec::idx_t, 4> labels{};
    refined.Search(2, queries.data(), 2, distances.data(), labels.data());
    if (metric == hypervec::kMetricL2) {
      EXPECT_EQ(labels, (std::array<hypervec::idx_t, 4>{0, 2, 1, 2}));
      EXPECT_EQ(distances, (std::array<float, 4>{1, 1, 1, 16}));
    } else {
      EXPECT_EQ(labels, (std::array<hypervec::idx_t, 4>{1, 2, 1, 2}));
      EXPECT_EQ(distances, (std::array<float, 4>{27, 12, 72, 32}));
    }
    EXPECT_THROW(
        refined.Search(2, queries.data(), 4, distances.data(), labels.data()),
        hypervec::HypervecException);
    EXPECT_THROW(refined.Add(1, base.data()), hypervec::HypervecException);
  }
}

TEST(IndexRerank, RejectsIncompatibleBaseAndCandidateCount) {
  DeliberatelyMisranked approximate(hypervec::kMetricL2);
  const std::array<float, 3> base = {1, 2, 3};
  EXPECT_THROW(hypervec::IndexRerank(approximate, base.data(), 2, 3),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::IndexRerank(approximate, base.data(), 3, 0),
               hypervec::HypervecException);
}
