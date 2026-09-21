/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/quantizer.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace hypervec {

#define HYPERVEC_RABITQ_DEFAULT_SEED 1234
#define HYPERVEC_RABITQ_DEFAULT_ROTATION_ROUNDS 3

/** CPU one-bit randomized quantizer for squared-L2 estimation.
 *
 * Vectors are zero-padded to a power of two and transformed by deterministic
 * randomized Walsh-Hadamard rounds. A code stores one sign bit per rotated
 * dimension plus two float correction factors. For a rotated query q and an
 * encoded vector x, the asymmetric inner-product estimate is
 *
 *   ||x||^2 / ||R*x||_1 * sum_i(sign((R*x)_i) * (R*q)_i).
 *
 * The estimate is exact when q == x. No dataset-dependent training is needed;
 * the seed and round count fully determine the shared orthogonal transform.
 */
class RaBitQQuantizer final : public Quantizer {
 public:
  explicit RaBitQQuantizer(
      idx_t dimension, uint64_t seed = HYPERVEC_RABITQ_DEFAULT_SEED,
      int rotation_rounds = HYPERVEC_RABITQ_DEFAULT_ROTATION_ROUNDS);

  std::string_view TypeName() const noexcept override { return "rabitq"; }
  idx_t Dimension() const noexcept override { return dimension_; }
  MetricType Metric() const noexcept override { return kMetricL2; }
  bool NeedsTraining() const noexcept override { return false; }
  bool IsTrained() const noexcept override { return true; }
  size_t CodeSize() const noexcept override { return code_size_; }

  size_t RotatedDimension() const noexcept { return rotated_dimension_; }
  size_t BitBytes() const noexcept { return bit_bytes_; }
  uint64_t Seed() const noexcept { return seed_; }
  int RotationRounds() const noexcept { return rotation_rounds_; }

  /** Apply the shared orthogonal transform to one input vector. */
  void Transform(const float* vector, float* rotated) const;

  /** Reverse one transformed vector and remove the zero-padding. */
  void InverseTransform(const float* rotated, float* vector) const;

  /** Estimate squared L2 using an already transformed query. */
  float EstimateSquaredDistance(const float* rotated_query,
                                float query_norm_squared,
                                const uint8_t* code) const;

  /** Prepare per-query signed sums once instead of revisiting every float for
   * every code in an inverted list. A nibble also handles padded dimensions. */
  using DistanceLut = std::vector<std::array<double, 16>>;
  void PrepareDistanceLut(const float* rotated_query, DistanceLut* lut) const;
  float EstimateSquaredDistanceWithLut(float query_norm_squared,
                                       const uint8_t* code,
                                       const DistanceLut& lut) const;

 protected:
  void TrainImpl(idx_t count, const float* vectors) override;
  void EncodeImpl(idx_t count, const float* vectors,
                  uint8_t* codes) const override;
  void DecodeImpl(idx_t count, const uint8_t* codes,
                  float* vectors) const override;
  std::unique_ptr<DistanceComputer> CreateDistanceComputerImpl(
      EncodedVectorView store) const override;

 private:
  void ApplyRotation(std::vector<float>* values) const;
  void ApplyInverseRotation(std::vector<float>* values) const;
  float ReadFactor(const uint8_t* code, size_t offset) const;
  void WriteFactor(float value, size_t offset, uint8_t* code) const;

  idx_t dimension_;
  size_t rotated_dimension_;
  size_t bit_bytes_;
  size_t code_size_;
  uint64_t seed_;
  int rotation_rounds_;
  std::vector<int8_t> rotation_signs_;
};

}  // namespace hypervec
