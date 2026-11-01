/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <quantization/rabitq/rabitq_quantizer.h>
#include <utils/log/exception.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {

std::vector<float> MakeVectors(hypervec::idx_t count,
                               hypervec::idx_t dimension) {
  std::vector<float> vectors(static_cast<size_t>(count * dimension));
  for (hypervec::idx_t row = 0; row < count; ++row) {
    for (hypervec::idx_t column = 0; column < dimension; ++column) {
      vectors[static_cast<size_t>(row * dimension + column)] =
          static_cast<float>(((row + 3) * (column + 5) + column * column) % 31 -
                             15) /
          7.0F;
    }
  }
  return vectors;
}

float SquaredNorm(const float* vector, size_t dimension) {
  float norm = 0.0F;
  for (size_t i = 0; i < dimension; ++i) {
    norm += vector[i] * vector[i];
  }
  return norm;
}

TEST(RaBitQQuantizer, RotationRoundtripsPaddedDimensions) {
  hypervec::RaBitQQuantizer quantizer(6, 2026, 3);
  EXPECT_EQ(quantizer.RotatedDimension(), 8U);
  EXPECT_EQ(quantizer.BitBytes(), 1U);
  EXPECT_EQ(quantizer.CodeSize(), 1U + 2 * sizeof(float));
  EXPECT_FALSE(quantizer.NeedsTraining());
  EXPECT_TRUE(quantizer.IsTrained());

  const std::vector<float> vector = {1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 6.0F};
  std::vector<float> rotated(quantizer.RotatedDimension());
  std::vector<float> restored(vector.size());
  quantizer.Transform(vector.data(), rotated.data());
  quantizer.InverseTransform(rotated.data(), restored.data());
  for (size_t i = 0; i < vector.size(); ++i) {
    EXPECT_NEAR(restored[i], vector[i], 2e-5F);
  }
  EXPECT_NEAR(SquaredNorm(rotated.data(), rotated.size()),
              SquaredNorm(vector.data(), vector.size()), 2e-4F);
}

TEST(RaBitQQuantizer, CodesAreDeterministicAndCompact) {
  constexpr hypervec::idx_t dimension = 64;
  const std::vector<float> vectors = MakeVectors(3, dimension);
  hypervec::RaBitQQuantizer first(dimension, 42, 3);
  hypervec::RaBitQQuantizer same(dimension, 42, 3);
  hypervec::RaBitQQuantizer different(dimension, 43, 3);
  EXPECT_EQ(first.CodeSize(), 16U);
  EXPECT_LT(first.CodeSize(), static_cast<size_t>(dimension) * sizeof(float));

  std::vector<uint8_t> first_codes(3 * first.CodeSize());
  std::vector<uint8_t> same_codes(first_codes.size());
  std::vector<uint8_t> different_codes(first_codes.size());
  first.Encode(3, vectors.data(), first_codes.data());
  same.Encode(3, vectors.data(), same_codes.data());
  different.Encode(3, vectors.data(), different_codes.data());
  EXPECT_EQ(first_codes, same_codes);
  EXPECT_NE(first_codes, different_codes);
}

TEST(RaBitQQuantizer, AsymmetricDistanceIsExactForSelfQueries) {
  constexpr hypervec::idx_t dimension = 32;
  constexpr hypervec::idx_t count = 8;
  const std::vector<float> vectors = MakeVectors(count, dimension);
  hypervec::RaBitQQuantizer quantizer(dimension, 1234, 3);
  std::vector<uint8_t> codes(static_cast<size_t>(count) * quantizer.CodeSize());
  quantizer.Encode(count, vectors.data(), codes.data());
  auto distance = quantizer.CreateDistanceComputer(
      hypervec::EncodedVectorView(codes.data(), count, quantizer.CodeSize()));

  for (hypervec::idx_t row = 0; row < count; ++row) {
    distance->SetQuery(vectors.data() + row * dimension);
    EXPECT_NEAR((*distance)(row), 0.0F, 2e-4F);
  }

  distance->SetQuery(vectors.data());
  EXPECT_THROW((*distance)(count), hypervec::HypervecException);
  const float lhs_rhs = distance->symmetric_dis(0, 1);
  const float rhs_lhs = distance->symmetric_dis(1, 0);
  EXPECT_GE(lhs_rhs, 0.0F);
  EXPECT_FLOAT_EQ(lhs_rhs, rhs_lhs);
}

TEST(RaBitQQuantizer, DecodePreservesNormForPowerOfTwoDimension) {
  constexpr hypervec::idx_t dimension = 16;
  const std::vector<float> vectors = MakeVectors(4, dimension);
  hypervec::RaBitQQuantizer quantizer(dimension);
  std::vector<uint8_t> codes(4 * quantizer.CodeSize());
  std::vector<float> decoded(vectors.size());
  quantizer.Encode(4, vectors.data(), codes.data());
  quantizer.Decode(4, codes.data(), decoded.data());

  for (size_t row = 0; row < 4; ++row) {
    EXPECT_NEAR(SquaredNorm(decoded.data() + row * dimension, dimension),
                SquaredNorm(vectors.data() + row * dimension, dimension),
                2e-4F);
  }
}

TEST(RaBitQQuantizer, PreservesNearestNeighborOnSeparatedFixture) {
  constexpr hypervec::idx_t dimension = 64;
  constexpr hypervec::idx_t count = 16;
  std::vector<float> vectors(static_cast<size_t>(count * dimension), 0.0F);
  for (hypervec::idx_t row = 0; row < count; ++row) {
    vectors[static_cast<size_t>(row * dimension + row)] = 4.0F;
  }
  hypervec::RaBitQQuantizer quantizer(dimension, 9001, 3);
  std::vector<uint8_t> codes(static_cast<size_t>(count) * quantizer.CodeSize());
  quantizer.Encode(count, vectors.data(), codes.data());
  auto distance = quantizer.CreateDistanceComputer(
      hypervec::EncodedVectorView(codes.data(), count, quantizer.CodeSize()));

  std::vector<float> query(static_cast<size_t>(dimension), 0.01F);
  for (hypervec::idx_t expected = 0; expected < count; ++expected) {
    query[static_cast<size_t>(expected)] += 4.0F;
    distance->SetQuery(query.data());
    hypervec::idx_t nearest = -1;
    float nearest_distance = (std::numeric_limits<float>::max)();
    for (hypervec::idx_t candidate = 0; candidate < count; ++candidate) {
      const float estimate = (*distance)(candidate);
      if (estimate < nearest_distance) {
        nearest_distance = estimate;
        nearest = candidate;
      }
    }
    EXPECT_EQ(nearest, expected);
    query[static_cast<size_t>(expected)] -= 4.0F;
  }
}

TEST(RaBitQQuantizer, PreparedLookupMatchesScalarEstimates) {
  for (hypervec::idx_t dimension : {1, 5, 16, 65, 300}) {
    hypervec::RaBitQQuantizer quantizer(dimension, 2026, 3);
    const std::vector<float> vectors = MakeVectors(17, dimension);
    std::vector<uint8_t> codes(17 * quantizer.CodeSize());
    quantizer.Encode(17, vectors.data(), codes.data());
    for (hypervec::idx_t query = 0; query < 5; ++query) {
      std::vector<float> rotated(quantizer.RotatedDimension());
      const float* input = vectors.data() + query * dimension;
      quantizer.Transform(input, rotated.data());
      hypervec::RaBitQQuantizer::DistanceLut lut;
      quantizer.PrepareDistanceLut(rotated.data(), &lut);
      const float norm = SquaredNorm(input, static_cast<size_t>(dimension));
      for (size_t row = 0; row < 17; ++row) {
        const uint8_t* code = codes.data() + row * quantizer.CodeSize();
        EXPECT_NEAR(
            quantizer.EstimateSquaredDistance(rotated.data(), norm, code),
            quantizer.EstimateSquaredDistanceWithLut(norm, code, lut),
            0.0001F * (1.0F + norm))
            << "dimension=" << dimension << " row=" << row;
      }
    }
  }
}

TEST(RaBitQQuantizer, DistanceComputerMatchesScalarAcrossQueries) {
  constexpr hypervec::idx_t dimension = 65;
  constexpr hypervec::idx_t count = 17;
  hypervec::RaBitQQuantizer quantizer(dimension, 2026, 3);
  const auto vectors = MakeVectors(count, dimension);
  std::vector<uint8_t> codes(static_cast<size_t>(count) * quantizer.CodeSize());
  quantizer.Encode(count, vectors.data(), codes.data());
  auto distance = quantizer.CreateDistanceComputer(
      hypervec::EncodedVectorView(codes.data(), count, quantizer.CodeSize()));

  for (hypervec::idx_t query = 0; query < 5; ++query) {
    const float* input = vectors.data() + query * dimension;
    std::vector<float> rotated(quantizer.RotatedDimension());
    quantizer.Transform(input, rotated.data());
    distance->SetQuery(input);
    for (hypervec::idx_t row = 0; row < count; ++row) {
      const float expected = quantizer.EstimateSquaredDistance(
          rotated.data(), SquaredNorm(input, dimension),
          codes.data() + static_cast<size_t>(row) * quantizer.CodeSize());
      EXPECT_NEAR((*distance)(row), expected,
                  0.0001F * (1.0F + SquaredNorm(input, dimension)));
    }
  }
}

TEST(RaBitQQuantizer, HandlesZeroVectorsAndRejectsInvalidState) {
  EXPECT_THROW(hypervec::RaBitQQuantizer(0), hypervec::HypervecException);
  EXPECT_THROW(hypervec::RaBitQQuantizer(4, 1, 0), hypervec::HypervecException);
  EXPECT_THROW(hypervec::RaBitQQuantizer(4, 1, 17),
               hypervec::HypervecException);

  hypervec::RaBitQQuantizer quantizer(4);
  const float zero[4] = {};
  std::vector<uint8_t> code(quantizer.CodeSize());
  float decoded[4] = {};
  quantizer.Encode(1, zero, code.data());
  quantizer.Decode(1, code.data(), decoded);
  for (float value : decoded) {
    EXPECT_FLOAT_EQ(value, 0.0F);
  }

  float invalid[4] = {0.0F, 1.0F, 2.0F,
                      (std::numeric_limits<float>::infinity)()};
  EXPECT_THROW(quantizer.Encode(1, invalid, code.data()),
               hypervec::HypervecException);
  EXPECT_THROW(quantizer.CreateDistanceComputer(hypervec::EncodedVectorView(
                   code.data(), 1, quantizer.CodeSize() + 1)),
               hypervec::HypervecException);

  const float nan = (std::numeric_limits<float>::quiet_NaN)();
  std::memcpy(code.data() + quantizer.BitBytes(), &nan, sizeof(nan));
  std::vector<float> rotated(quantizer.RotatedDimension());
  quantizer.Transform(zero, rotated.data());
  EXPECT_THROW(
      quantizer.EstimateSquaredDistance(rotated.data(), 0.0F, code.data()),
      hypervec::HypervecException);
}

}  // namespace
