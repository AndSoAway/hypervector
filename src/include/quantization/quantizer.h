/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <utils/distances/distance_computer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace hypervec {

/** Non-owning view over fixed-width encoded vectors. */
class EncodedVectorView {
 public:
  EncodedVectorView(const uint8_t* data, idx_t count, size_t code_size);

  const uint8_t* Data() const noexcept { return data_; }
  idx_t Size() const noexcept { return count_; }
  size_t CodeSize() const noexcept { return code_size_; }
  const uint8_t* Code(idx_t index) const;

 private:
  const uint8_t* data_;
  idx_t count_;
  size_t code_size_;
};

/** Owning, append-only store for fixed-width encoded vectors. */
class InMemoryCodeStore {
 public:
  explicit InMemoryCodeStore(size_t code_size);

  const uint8_t* Data() const noexcept { return bytes_.data(); }
  uint8_t* Data() noexcept { return bytes_.data(); }
  idx_t Size() const noexcept;
  size_t CodeSize() const noexcept { return code_size_; }

  const uint8_t* Code(idx_t index) const;
  uint8_t* Code(idx_t index);
  EncodedVectorView View() const;

  void Append(idx_t count, const uint8_t* codes);
  void Reset() noexcept { bytes_.clear(); }

 private:
  size_t code_size_;
  std::vector<uint8_t> bytes_;
};

/** Common lifecycle, codec, and random-access distance contract. */
class Quantizer {
 public:
  virtual ~Quantizer() = default;

  virtual std::string_view TypeName() const noexcept = 0;
  virtual idx_t Dimension() const noexcept = 0;
  virtual MetricType Metric() const noexcept = 0;
  virtual bool NeedsTraining() const noexcept = 0;
  virtual bool IsTrained() const noexcept = 0;
  virtual size_t CodeSize() const noexcept = 0;

  void Train(idx_t count, const float* vectors);
  void Encode(idx_t count, const float* vectors, uint8_t* codes) const;
  void Decode(idx_t count, const uint8_t* codes, float* vectors) const;
  std::unique_ptr<DistanceComputer> CreateDistanceComputer(
      EncodedVectorView store) const;

 protected:
  virtual void TrainImpl(idx_t count, const float* vectors) = 0;
  virtual void EncodeImpl(idx_t count, const float* vectors,
                          uint8_t* codes) const = 0;
  virtual void DecodeImpl(idx_t count, const uint8_t* codes,
                          float* vectors) const = 0;
  virtual std::unique_ptr<DistanceComputer> CreateDistanceComputerImpl(
      EncodedVectorView store) const = 0;
};

/** Lossless float32 codec implementing the Quantizer protocol. */
class FlatQuantizer final : public Quantizer {
 public:
  FlatQuantizer(idx_t dimension, MetricType metric, float metric_arg = 0.0F);

  std::string_view TypeName() const noexcept override { return "flat"; }
  idx_t Dimension() const noexcept override { return dimension_; }
  MetricType Metric() const noexcept override { return metric_; }
  bool NeedsTraining() const noexcept override { return false; }
  bool IsTrained() const noexcept override { return true; }
  size_t CodeSize() const noexcept override { return code_size_; }

 protected:
  void TrainImpl(idx_t count, const float* vectors) override;
  void EncodeImpl(idx_t count, const float* vectors,
                  uint8_t* codes) const override;
  void DecodeImpl(idx_t count, const uint8_t* codes,
                  float* vectors) const override;
  std::unique_ptr<DistanceComputer> CreateDistanceComputerImpl(
      EncodedVectorView store) const override;

 private:
  idx_t dimension_;
  MetricType metric_;
  float metric_arg_;
  size_t code_size_;
};

}  // namespace hypervec
