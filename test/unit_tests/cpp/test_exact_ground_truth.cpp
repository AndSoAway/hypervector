/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/exact_ground_truth.h>
#include <gtest/gtest.h>
#include <utils/log/exception.h>

#include <limits>
#include <vector>

TEST(SemanticMetric, MapsWorkloadSemanticsAndRejectsIndexMismatch) {
  EXPECT_EQ(hypervec::SemanticMetricName(hypervec::SemanticMetric::kL2), "l2");
  EXPECT_EQ(hypervec::SemanticMetricName(hypervec::SemanticMetric::kCosine),
            "cosine");
  EXPECT_EQ(hypervec::IndexMetricForSemanticMetric(
                hypervec::SemanticMetric::kInnerProduct),
            hypervec::kMetricInnerProduct);
  EXPECT_EQ(
      hypervec::IndexMetricForSemanticMetric(hypervec::SemanticMetric::kCosine),
      hypervec::kMetricInnerProduct);
  EXPECT_NO_THROW(hypervec::ValidateSemanticMetricIndex(
      hypervec::kMetricInnerProduct, hypervec::SemanticMetric::kCosine));
  EXPECT_THROW(hypervec::ValidateSemanticMetricIndex(
                   hypervec::kMetricL2, hypervec::SemanticMetric::kCosine),
               hypervec::HypervecException);
}

TEST(ExactGroundTruth, ComputesL2NeighborsWithStableTies) {
  const hypervec::FloatVectorDataset base{
      4, 2, {0.0F, 0.0F, 2.0F, 0.0F, 2.0F, 0.0F, 0.0F, 3.0F}};
  const hypervec::FloatVectorDataset queries{2, 2, {1.9F, 0.0F, 0.0F, 2.8F}};

  const hypervec::IntegerVectorDataset result =
      hypervec::ComputeExactGroundTruth(base, queries, 3,
                                        hypervec::GroundTruthMetric::kL2);

  EXPECT_EQ(result.vector_count, 2);
  EXPECT_EQ(result.dimension, 3);
  EXPECT_EQ(result.values, (std::vector<hypervec::idx_t>{1, 2, 0, 3, 0, 1}));
}

TEST(ExactGroundTruth, ComputesInnerProductAndNormalizedCosineNeighbors) {
  const hypervec::FloatVectorDataset inner_product_base{
      3, 2, {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F}};
  const hypervec::FloatVectorDataset inner_product_queries{1, 2, {2.0F, 1.0F}};
  const auto inner_product = hypervec::ComputeExactGroundTruth(
      inner_product_base, inner_product_queries, 3,
      hypervec::GroundTruthMetric::kInnerProduct);
  EXPECT_EQ(inner_product.values, (std::vector<hypervec::idx_t>{2, 0, 1}));

  constexpr float kInverseSqrtTwo = 0.70710677F;
  const hypervec::FloatVectorDataset cosine_base{
      3, 2, {1.0F, 0.0F, 0.0F, 1.0F, kInverseSqrtTwo, kInverseSqrtTwo}};
  const hypervec::FloatVectorDataset cosine_queries{1, 2, {1.0F, 0.0F}};
  const auto cosine = hypervec::ComputeExactGroundTruth(
      cosine_base, cosine_queries, 3, hypervec::GroundTruthMetric::kCosine);
  EXPECT_EQ(cosine.values, (std::vector<hypervec::idx_t>{0, 2, 1}));
}

TEST(ExactGroundTruth, RejectsInvalidInputsAndUnnormalizedCosine) {
  const hypervec::FloatVectorDataset base{2, 2, {1.0F, 0.0F, 0.0F, 1.0F}};
  const hypervec::FloatVectorDataset queries{1, 2, {1.0F, 0.0F}};
  EXPECT_THROW(hypervec::ComputeExactGroundTruth(
                   base, queries, 0, hypervec::GroundTruthMetric::kL2),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::ComputeExactGroundTruth(
                   base, queries, 3, hypervec::GroundTruthMetric::kL2),
               hypervec::HypervecException);

  const hypervec::FloatVectorDataset wrong_dimension{1, 1, {1.0F}};
  EXPECT_THROW(hypervec::ComputeExactGroundTruth(
                   base, wrong_dimension, 1, hypervec::GroundTruthMetric::kL2),
               hypervec::HypervecException);
  const hypervec::FloatVectorDataset unnormalized{1, 2, {2.0F, 0.0F}};
  EXPECT_THROW(hypervec::ComputeExactGroundTruth(
                   base, unnormalized, 1, hypervec::GroundTruthMetric::kCosine),
               hypervec::HypervecException);
  const hypervec::FloatVectorDataset non_finite{
      1, 2, {(std::numeric_limits<float>::infinity)(), 0.0F}};
  EXPECT_THROW(
      hypervec::ComputeExactGroundTruth(
          base, non_finite, 1, hypervec::GroundTruthMetric::kInnerProduct),
      hypervec::HypervecException);
  EXPECT_THROW(
      hypervec::ComputeExactGroundTruth(
          base, queries, 1, static_cast<hypervec::GroundTruthMetric>(99)),
      hypervec::HypervecException);
}
