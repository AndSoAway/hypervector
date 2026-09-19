/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <transform/vector_transform.h>
#include <utils/log/exception.h>

#include <limits>
#include <vector>

namespace {

TEST(VectorTransform, IdentitySupportsAllocatedAndInPlaceBatches) {
  hypervec::LinearTransform linear(3, 3);
  EXPECT_FALSE(linear.RequiresTraining());
  EXPECT_FALSE(linear.IsReversible());
  linear.SetIdentity();
  EXPECT_TRUE(linear.IsReversible());
  const std::vector<float> input = {1.0F, 2.0F, 3.0F, -4.0F, 5.0F, 6.0F};

  EXPECT_EQ(linear.Apply(2, input.data()), input);
  EXPECT_EQ(linear.ReverseTransform(2, input.data()), input);

  std::vector<float> in_place = input;
  linear.Apply(2, in_place.data(), in_place.data());
  EXPECT_EQ(in_place, input);
  linear.ReverseTransform(2, in_place.data(), in_place.data());
  EXPECT_EQ(in_place, input);
}

TEST(VectorTransform, OrthogonalAffineTransformRoundtrips) {
  hypervec::LinearTransform linear(2, 2);
  linear.SetTransform({0.0F, -1.0F, 1.0F, 0.0F}, {3.0F, -2.0F}, true);
  const std::vector<float> input = {1.0F, 2.0F, -4.0F, 5.0F};

  const std::vector<float> rotated = linear.Apply(2, input.data());
  EXPECT_EQ(rotated, (std::vector<float>{1.0F, -1.0F, -2.0F, -6.0F}));

  const std::vector<float> restored =
      linear.ReverseTransform(2, rotated.data());
  ASSERT_EQ(restored.size(), input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(restored[i], input[i], 1e-6F);
  }
}

TEST(VectorTransform, GeneralLinearTransformHasNoImplicitInverse) {
  hypervec::LinearTransform linear(2, 1);
  linear.SetTransform({2.0F, -1.0F});
  const std::vector<float> input = {3.0F, 4.0F, -2.0F, 5.0F};

  EXPECT_EQ(linear.Apply(2, input.data()), (std::vector<float>{2.0F, -9.0F}));
  const std::vector<float> projected = linear.Apply(2, input.data());
  float output[4] = {};
  EXPECT_THROW(linear.ReverseTransform(2, projected.data(), output),
               hypervec::HypervecException);
}

TEST(VectorTransform, RejectedUpdatePreservesActiveState) {
  hypervec::LinearTransform linear(2, 2);
  linear.SetIdentity();
  const std::vector<float> input = {2.0F, -3.0F};
  const float nan = std::numeric_limits<float>::quiet_NaN();

  EXPECT_THROW(linear.SetTransform({1.0F, 0.0F, 0.0F, nan}, {}, true),
               hypervec::HypervecException);
  EXPECT_THROW(linear.SetTransform({1.0F, 0.0F, 0.0F, 2.0F}, {}, true),
               hypervec::HypervecException);
  EXPECT_EQ(linear.Apply(1, input.data()), input);
  EXPECT_TRUE(linear.is_orthonormal);
}

TEST(VectorTransform, ValidatesDimensionsStateAndBuffers) {
  EXPECT_THROW(hypervec::LinearTransform(0, 2), hypervec::HypervecException);
  EXPECT_THROW(hypervec::LinearTransform(2, -1), hypervec::HypervecException);

  hypervec::LinearTransform linear(2, 2);
  const float input[2] = {1.0F, 2.0F};
  float output[2] = {};
  EXPECT_THROW(linear.Apply(1, input, output), hypervec::HypervecException);
  EXPECT_THROW(linear.SetTransform({1.0F, 0.0F, 0.0F}),
               hypervec::HypervecException);
  EXPECT_THROW(linear.SetTransform({1.0F, 0.0F, 0.0F, 1.0F}, {1.0F}),
               hypervec::HypervecException);

  linear.SetIdentity();
  EXPECT_THROW(linear.Apply(-1, input, output), hypervec::HypervecException);
  EXPECT_THROW(linear.Apply(1, nullptr, output), hypervec::HypervecException);
  EXPECT_THROW(linear.Apply(1, input, nullptr), hypervec::HypervecException);
  EXPECT_NO_THROW(linear.Apply(0, nullptr, nullptr));
}

}  // namespace
