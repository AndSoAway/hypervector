/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/rabitq/rabitq_quantizer.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

size_t NextPowerOfTwo(idx_t dimension) {
  HYPERVEC_THROW_IF_NOT_MSG(dimension > 0,
                            "RaBitQQuantizer: dimension must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<uint64_t>(dimension) <=
          static_cast<uint64_t>((std::numeric_limits<size_t>::max)()),
      "RaBitQQuantizer: dimension does not fit in size_t");
  const size_t target = static_cast<size_t>(dimension);
  size_t result = 1;
  while (result < target) {
    HYPERVEC_THROW_IF_NOT_MSG(
        result <= (std::numeric_limits<size_t>::max)() / 2,
        "RaBitQQuantizer: padded dimension overflows size_t");
    result *= 2;
  }
  return result;
}

uint64_t SplitMix64(uint64_t* state) {
  *state += UINT64_C(0x9E3779B97F4A7C15);
  uint64_t value = *state;
  value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
  return value ^ (value >> 31);
}

void Hadamard(float* values, size_t size) {
  for (size_t half = 1; half < size; half *= 2) {
    const size_t span = half * 2;
    for (size_t begin = 0; begin < size; begin += span) {
      for (size_t offset = 0; offset < half; ++offset) {
        const float lhs = values[begin + offset];
        const float rhs = values[begin + half + offset];
        values[begin + offset] = lhs + rhs;
        values[begin + half + offset] = lhs - rhs;
      }
    }
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(size));
  for (size_t i = 0; i < size; ++i) {
    values[i] *= scale;
  }
}

float SquaredNorm(const float* vector, idx_t dimension, const char* operation) {
  double norm = 0.0;
  for (idx_t i = 0; i < dimension; ++i) {
    HYPERVEC_THROW_IF_NOT_FMT(std::isfinite(vector[i]),
                              "%s: non-finite value at offset %zu", operation,
                              static_cast<size_t>(i));
    norm += static_cast<double>(vector[i]) * vector[i];
  }
  HYPERVEC_THROW_IF_NOT_FMT(
      std::isfinite(norm) &&
          norm <= static_cast<double>((std::numeric_limits<float>::max)()),
      "%s: squared norm is not representable", operation);
  return static_cast<float>(norm);
}

class RaBitQDistanceComputer final : public DistanceComputer {
 public:
  RaBitQDistanceComputer(const RaBitQQuantizer& quantizer,
                         EncodedVectorView store)
      : quantizer_(quantizer),
        store_(store),
        rotated_query_(quantizer.RotatedDimension()),
        decoded_lhs_(static_cast<size_t>(quantizer.Dimension())),
        decoded_rhs_(static_cast<size_t>(quantizer.Dimension())) {}

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                              "RaBitQQuantizer: query must not be null");
    query_norm_squared_ =
        SquaredNorm(query, quantizer_.Dimension(), "RaBitQ query");
    quantizer_.Transform(query, rotated_query_.data());
    quantizer_.PrepareDistanceLut(rotated_query_.data(), &distance_lut_);
    query_ready_ = true;
  }

  float operator()(idx_t index) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ready_,
        "RaBitQQuantizer: SetQuery must be called before distance evaluation");
    return quantizer_.EstimateSquaredDistanceWithLut(
        query_norm_squared_, store_.Code(index), distance_lut_);
  }

  float symmetric_dis(idx_t lhs, idx_t rhs) override {
    quantizer_.Decode(1, store_.Code(lhs), decoded_lhs_.data());
    quantizer_.Decode(1, store_.Code(rhs), decoded_rhs_.data());
    double distance = 0.0;
    for (idx_t i = 0; i < quantizer_.Dimension(); ++i) {
      const double difference =
          static_cast<double>(decoded_lhs_[static_cast<size_t>(i)]) -
          decoded_rhs_[static_cast<size_t>(i)];
      distance += difference * difference;
    }
    return static_cast<float>(distance);
  }

 private:
  const RaBitQQuantizer& quantizer_;
  EncodedVectorView store_;
  std::vector<float> rotated_query_;
  RaBitQQuantizer::DistanceLut distance_lut_;
  std::vector<float> decoded_lhs_;
  std::vector<float> decoded_rhs_;
  float query_norm_squared_ = 0.0F;
  bool query_ready_ = false;
};

}  // namespace

RaBitQQuantizer::RaBitQQuantizer(idx_t dimension, uint64_t seed,
                                 int rotation_rounds)
    : dimension_(dimension),
      rotated_dimension_(NextPowerOfTwo(dimension)),
      bit_bytes_((rotated_dimension_ + 7) / 8),
      code_size_(
          add_no_overflow(bit_bytes_, 2 * sizeof(float), "RaBitQ code size")),
      seed_(seed),
      rotation_rounds_(rotation_rounds) {
  HYPERVEC_THROW_IF_NOT_MSG(
      rotation_rounds > 0 && rotation_rounds <= 16,
      "RaBitQQuantizer: rotation_rounds must be in [1, 16]");
  const size_t sign_count =
      mul_no_overflow(static_cast<size_t>(rotation_rounds_), rotated_dimension_,
                      "RaBitQ rotation signs");
  rotation_signs_.resize(sign_count);
  uint64_t state = seed_;
  for (int8_t& sign : rotation_signs_) {
    sign = (SplitMix64(&state) & 1U) == 0 ? int8_t{-1} : int8_t{1};
  }
}

void RaBitQQuantizer::ApplyRotation(float* values) const {
  for (int round = 0; round < rotation_rounds_; ++round) {
    const int8_t* signs = rotation_signs_.data() +
                          static_cast<size_t>(round) * rotated_dimension_;
    for (size_t i = 0; i < rotated_dimension_; ++i) {
      values[i] *= signs[i];
    }
    Hadamard(values, rotated_dimension_);
  }
}

void RaBitQQuantizer::ApplyInverseRotation(std::vector<float>* values) const {
  for (int round = rotation_rounds_; round-- > 0;) {
    Hadamard(values->data(), rotated_dimension_);
    const int8_t* signs = rotation_signs_.data() +
                          static_cast<size_t>(round) * rotated_dimension_;
    for (size_t i = 0; i < rotated_dimension_; ++i) {
      (*values)[i] *= signs[i];
    }
  }
}

void RaBitQQuantizer::Transform(const float* vector, float* rotated) const {
  HYPERVEC_THROW_IF_NOT_MSG(vector != nullptr && rotated != nullptr,
                            "RaBitQQuantizer::Transform: buffers are null");
  SquaredNorm(vector, dimension_, "RaBitQQuantizer::Transform");
  std::memmove(rotated, vector,
               static_cast<size_t>(dimension_) * sizeof(float));
  std::fill(rotated + dimension_, rotated + rotated_dimension_, 0.0F);
  ApplyRotation(rotated);
}

void RaBitQQuantizer::InverseTransform(const float* rotated,
                                       float* vector) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      rotated != nullptr && vector != nullptr,
      "RaBitQQuantizer::InverseTransform: buffers are null");
  std::vector<float> values(rotated, rotated + rotated_dimension_);
  for (size_t i = 0; i < values.size(); ++i) {
    HYPERVEC_THROW_IF_NOT_FMT(
        std::isfinite(values[i]),
        "RaBitQQuantizer::InverseTransform: non-finite value at offset %zu", i);
  }
  ApplyInverseRotation(&values);
  std::copy(values.begin(), values.begin() + dimension_, vector);
}

float RaBitQQuantizer::ReadFactor(const uint8_t* code, size_t offset) const {
  float value;
  std::memcpy(&value, code + bit_bytes_ + offset, sizeof(value));
  return value;
}

void RaBitQQuantizer::WriteFactor(float value, size_t offset,
                                  uint8_t* code) const {
  std::memcpy(code + bit_bytes_ + offset, &value, sizeof(value));
}

void RaBitQQuantizer::TrainImpl(idx_t /*count*/, const float* /*vectors*/) {}

void RaBitQQuantizer::EncodeImpl(idx_t count, const float* vectors,
                                 uint8_t* codes) const {
  std::vector<float> rotated(rotated_dimension_);
  for (idx_t row = 0; row < count; ++row) {
    const float* vector = vectors + row * dimension_;
    uint8_t* code = codes + static_cast<size_t>(row) * code_size_;
    std::memset(code, 0, code_size_);
    const float norm_squared =
        SquaredNorm(vector, dimension_, "RaBitQQuantizer::Encode");
    // SquaredNorm above already validated the input; reuse the batch scratch
    // instead of allocating a padded vector and validating it again per row.
    std::copy(vector, vector + dimension_, rotated.begin());
    std::fill(rotated.begin() + dimension_, rotated.end(), 0.0F);
    ApplyRotation(rotated.data());

    double absolute_sum = 0.0;
    for (size_t i = 0; i < rotated_dimension_; ++i) {
      if (rotated[i] >= 0.0F) {
        code[i / 8] |= static_cast<uint8_t>(1U << (i % 8));
      }
      absolute_sum += std::abs(static_cast<double>(rotated[i]));
    }
    const float scale =
        norm_squared == 0.0F
            ? 0.0F
            : static_cast<float>(static_cast<double>(norm_squared) /
                                 absolute_sum);
    HYPERVEC_THROW_IF_NOT_MSG(
        std::isfinite(scale),
        "RaBitQQuantizer::Encode: correction factor is not finite");
    WriteFactor(norm_squared, 0, code);
    WriteFactor(scale, sizeof(float), code);
  }
}

void RaBitQQuantizer::DecodeImpl(idx_t count, const uint8_t* codes,
                                 float* vectors) const {
  std::vector<float> rotated(rotated_dimension_);
  const float inverse_root_dimension =
      1.0F / std::sqrt(static_cast<float>(rotated_dimension_));
  for (idx_t row = 0; row < count; ++row) {
    const uint8_t* code = codes + static_cast<size_t>(row) * code_size_;
    const float norm_squared = ReadFactor(code, 0);
    HYPERVEC_THROW_IF_NOT_MSG(
        std::isfinite(norm_squared) && norm_squared >= 0.0F,
        "RaBitQQuantizer::Decode: invalid squared norm");
    const float magnitude = std::sqrt(norm_squared) * inverse_root_dimension;
    for (size_t i = 0; i < rotated_dimension_; ++i) {
      const bool positive = (code[i / 8] & (1U << (i % 8))) != 0;
      rotated[i] = positive ? magnitude : -magnitude;
    }
    InverseTransform(rotated.data(), vectors + row * dimension_);
  }
}

float RaBitQQuantizer::EstimateSquaredDistance(const float* rotated_query,
                                               float query_norm_squared,
                                               const uint8_t* code) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      rotated_query != nullptr && code != nullptr,
      "RaBitQQuantizer::EstimateSquaredDistance: buffers are null");
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(query_norm_squared) && query_norm_squared >= 0.0F,
      "RaBitQQuantizer::EstimateSquaredDistance: invalid query norm");
  const float vector_norm_squared = ReadFactor(code, 0);
  const float scale = ReadFactor(code, sizeof(float));
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(vector_norm_squared) && vector_norm_squared >= 0.0F &&
          std::isfinite(scale) && scale >= 0.0F,
      "RaBitQQuantizer::EstimateSquaredDistance: invalid code factors");

  double signed_sum = 0.0;
  for (size_t i = 0; i < rotated_dimension_; ++i) {
    HYPERVEC_THROW_IF_NOT_FMT(
        std::isfinite(rotated_query[i]),
        "RaBitQQuantizer::EstimateSquaredDistance: non-finite query value at "
        "offset %zu",
        i);
    const bool positive = (code[i / 8] & (1U << (i % 8))) != 0;
    signed_sum += positive ? rotated_query[i] : -rotated_query[i];
  }
  const double estimated = static_cast<double>(query_norm_squared) +
                           vector_norm_squared -
                           2.0 * static_cast<double>(scale) * signed_sum;
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(estimated),
      "RaBitQQuantizer::EstimateSquaredDistance: estimate is not finite");
  return static_cast<float>(std::max(0.0, estimated));
}

void RaBitQQuantizer::PrepareDistanceLut(const float* rotated_query,
                                         DistanceLut* lut) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      rotated_query != nullptr && lut != nullptr,
      "RaBitQQuantizer::PrepareDistanceLut: buffers are null");
  lut->resize(bit_bytes_ * 2);
  for (size_t nibble = 0; nibble < lut->size(); ++nibble) {
    double values[4] = {};
    for (size_t bit = 0; bit < 4; ++bit) {
      const size_t offset = nibble * 4 + bit;
      if (offset < rotated_dimension_) {
        HYPERVEC_THROW_IF_NOT_FMT(std::isfinite(rotated_query[offset]),
                                  "RaBitQQuantizer::PrepareDistanceLut: "
                                  "non-finite query value at offset %zu",
                                  offset);
        values[bit] = rotated_query[offset];
      }
    }
    for (size_t bits = 0; bits < 16; ++bits) {
      double sum = 0.0;
      for (size_t bit = 0; bit < 4; ++bit) {
        sum += (bits & (size_t{1} << bit)) != 0 ? values[bit] : -values[bit];
      }
      (*lut)[nibble][bits] = sum;
    }
  }
}

float RaBitQQuantizer::EstimateSquaredDistanceWithLut(
    float query_norm_squared, const uint8_t* code,
    const DistanceLut& lut) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      code != nullptr && lut.size() == bit_bytes_ * 2,
      "RaBitQQuantizer::EstimateSquaredDistanceWithLut: invalid buffers");
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(query_norm_squared) && query_norm_squared >= 0.0F,
      "RaBitQQuantizer::EstimateSquaredDistanceWithLut: invalid query norm");
  const float vector_norm_squared = ReadFactor(code, 0);
  const float scale = ReadFactor(code, sizeof(float));
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(vector_norm_squared) && vector_norm_squared >= 0.0F &&
          std::isfinite(scale) && scale >= 0.0F,
      "RaBitQQuantizer::EstimateSquaredDistanceWithLut: invalid code factors");
  double signed_sum = 0.0;
  for (size_t byte = 0; byte < bit_bytes_; ++byte) {
    const uint8_t bits = code[byte];
    signed_sum += lut[byte * 2][bits & 15U];
    signed_sum += lut[byte * 2 + 1][bits >> 4];
  }
  const double estimated = static_cast<double>(query_norm_squared) +
                           vector_norm_squared -
                           2.0 * static_cast<double>(scale) * signed_sum;
  HYPERVEC_THROW_IF_NOT_MSG(std::isfinite(estimated),
                            "RaBitQQuantizer::EstimateSquaredDistanceWithLut: "
                            "estimate is not finite");
  return static_cast<float>(std::max(0.0, estimated));
}

std::unique_ptr<DistanceComputer> RaBitQQuantizer::CreateDistanceComputerImpl(
    EncodedVectorView store) const {
  return std::make_unique<RaBitQDistanceComputer>(*this, std::move(store));
}

}  // namespace hypervec
