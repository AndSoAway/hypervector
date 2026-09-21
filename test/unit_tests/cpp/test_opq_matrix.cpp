/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <quantization/pq/pq.h>
#include <transform/opq_matrix.h>
#include <utils/log/exception.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr size_t kDimension = 8;

std::vector<float> CorrelatedTrainingData(hypervec::idx_t n) {
  std::vector<float> data(static_cast<size_t>(n) * kDimension);
  uint32_t state = 0x12345678U;
  auto sample = [&state]() {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>((state >> 8) & 0xFFFFU) / 32767.5F - 1.0F;
  };
  for (hypervec::idx_t i = 0; i < n; ++i) {
    std::array<float, kDimension> latent;
    for (size_t j = 0; j < kDimension; ++j) {
      latent[j] = sample() * std::pow(0.35F, static_cast<float>(j) / 2.0F);
    }
    for (size_t row = 0; row < kDimension; ++row) {
      double value = 0.0;
      for (size_t column = 0; column < kDimension; ++column) {
        const float sign =
            (std::popcount(static_cast<unsigned>(row & column)) & 1) != 0
                ? -1.0F
                : 1.0F;
        value += sign * latent[column];
      }
      data[static_cast<size_t>(i) * kDimension + row] =
          static_cast<float>(value / std::sqrt(8.0));
    }
  }
  return data;
}

double QuantizationMse(const hypervec::ProductQuantizer& pq, hypervec::idx_t n,
                       const float* data) {
  std::vector<uint8_t> codes(static_cast<size_t>(n) * pq.code_size);
  std::vector<float> reconstructed(static_cast<size_t>(n * pq.d));
  pq.ComputeCodes(n, data, codes.data());
  pq.DecodeBatch(n, codes.data(), reconstructed.data());
  double error = 0.0;
  for (size_t i = 0; i < reconstructed.size(); ++i) {
    const double difference = data[i] - reconstructed[i];
    error += difference * difference;
  }
  return error / static_cast<double>(reconstructed.size());
}

TEST(OPQMatrix, LearnsDeterministicOrthonormalRotation) {
  constexpr hypervec::idx_t n = 256;
  const std::vector<float> data = CorrelatedTrainingData(n);
  hypervec::OPQMatrix first(8, 4, 2);
  hypervec::OPQMatrix second(8, 4, 2);
  first.parameters.iterations = 4;
  second.parameters = first.parameters;
  first.parameters.pq_parameters.niter = 12;
  second.parameters.pq_parameters = first.parameters.pq_parameters;
  EXPECT_TRUE(first.RequiresTraining());
  EXPECT_FALSE(first.IsReversible());

  first.Train(n, data.data());
  second.Train(n, data.data());

  EXPECT_TRUE(first.is_trained);
  EXPECT_TRUE(first.is_orthonormal);
  EXPECT_TRUE(first.IsReversible());
  ASSERT_EQ(first.matrix.size(), second.matrix.size());
  for (size_t i = 0; i < first.matrix.size(); ++i) {
    EXPECT_NEAR(first.matrix[i], second.matrix[i], 1e-6F);
  }

  const std::vector<float> rotated = first.Apply(n, data.data());
  const std::vector<float> restored = first.ReverseTransform(n, rotated.data());
  ASSERT_EQ(restored.size(), data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_NEAR(restored[i], data[i], 2e-4F);
  }
}

TEST(OPQMatrix, ReducesProductQuantizationErrorOnCorrelatedData) {
  constexpr hypervec::idx_t n = 512;
  constexpr hypervec::idx_t d = 8;
  const std::vector<float> data = CorrelatedTrainingData(n);
  hypervec::PQParameters pq_parameters;
  pq_parameters.niter = 20;

  hypervec::ProductQuantizer plain(d, 4, 2);
  plain.Train(n, data.data(), pq_parameters);
  const double plain_error = QuantizationMse(plain, n, data.data());

  hypervec::OPQMatrix opq(d, 4, 2);
  opq.parameters.iterations = 6;
  opq.parameters.pq_parameters = pq_parameters;
  opq.Train(n, data.data());
  const std::vector<float> rotated = opq.Apply(n, data.data());
  hypervec::ProductQuantizer optimized(d, 4, 2);
  optimized.Train(n, rotated.data(), pq_parameters);
  const double optimized_error = QuantizationMse(optimized, n, rotated.data());

  EXPECT_LT(optimized_error, plain_error * 0.95);
}

TEST(OPQMatrix, BoundedTrainingSamplesDeterministicallyAndValidatesAllRows) {
  const std::vector<float> data = CorrelatedTrainingData(256);
  hypervec::OPQMatrix first(8, 4, 2);
  first.parameters.iterations = 2;
  first.parameters.pq_parameters.niter = 5;
  first.parameters.max_training_rows = 32;
  hypervec::OPQMatrix second(8, 4, 2);
  second.parameters = first.parameters;
  first.Train(256, data.data());
  second.Train(256, data.data());
  EXPECT_EQ(first.matrix, second.matrix);
  std::vector<float> invalid = data;
  invalid.back() = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(first.Train(256, invalid.data()), hypervec::HypervecException);
  EXPECT_EQ(first.matrix, second.matrix);

  first.parameters.max_training_rows = -1;
  EXPECT_THROW(first.Train(256, data.data()), hypervec::HypervecException);
  first.parameters.max_training_rows = 2;
  EXPECT_THROW(first.Train(256, data.data()), hypervec::HypervecException);
}

TEST(OPQMatrix, FailedRetrainingPreservesUsableState) {
  constexpr hypervec::idx_t n = 64;
  std::vector<float> data = CorrelatedTrainingData(n);
  hypervec::OPQMatrix opq(8, 4, 2);
  opq.parameters.iterations = 2;
  opq.parameters.pq_parameters.niter = 5;
  opq.Train(n, data.data());
  const std::vector<float> previous_matrix = opq.matrix;

  data[7] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(opq.Train(n, data.data()), hypervec::HypervecException);
  EXPECT_EQ(opq.matrix, previous_matrix);
  EXPECT_TRUE(opq.is_trained);
}

TEST(OPQMatrix, ValidatesConstructionAndTrainingOptions) {
  EXPECT_THROW(hypervec::OPQMatrix(8, 0, 2), hypervec::HypervecException);
  EXPECT_THROW(hypervec::OPQMatrix(7, 4, 2), hypervec::HypervecException);
  EXPECT_THROW(hypervec::OPQMatrix(8, 4, 0), hypervec::HypervecException);
  EXPECT_GE(hypervec::OPQMatrix(8, 4, 16).parameters.max_training_rows, 65536);

  hypervec::OPQMatrix opq(8, 4, 2);
  opq.parameters.iterations = 0;
  const std::vector<float> data = CorrelatedTrainingData(16);
  EXPECT_THROW(opq.Train(16, data.data()), hypervec::HypervecException);
  opq.parameters.iterations = 1;
  EXPECT_THROW(opq.Train(0, data.data()), hypervec::HypervecException);
  EXPECT_THROW(opq.Train(16, nullptr), hypervec::HypervecException);
}

}  // namespace
