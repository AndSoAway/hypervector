/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/pq/pq.h>
#include <quantization/quantizer.h>

#include <memory>
#include <string_view>

namespace hypervec {

/** Quantizer protocol adapter borrowing an existing ProductQuantizer.
 *
 * The model must outlive this adapter and every DistanceComputer created by
 * it. The const-model overload supports encoding, decoding, and distance
 * evaluation, but rejects Train().
 */
class ProductQuantizerAdapter final : public Quantizer {
 public:
  explicit ProductQuantizerAdapter(ProductQuantizer& pq,
                                   PQParameters parameters = {});
  explicit ProductQuantizerAdapter(const ProductQuantizer& pq);

  std::string_view TypeName() const noexcept override { return "pq"; }
  idx_t Dimension() const noexcept override { return pq_.d; }
  MetricType Metric() const noexcept override { return kMetricL2; }
  bool NeedsTraining() const noexcept override { return true; }
  bool IsTrained() const noexcept override { return pq_.is_trained; }
  size_t CodeSize() const noexcept override { return pq_.code_size; }

 protected:
  void TrainImpl(idx_t count, const float* vectors) override;
  void EncodeImpl(idx_t count, const float* vectors,
                  uint8_t* codes) const override;
  void DecodeImpl(idx_t count, const uint8_t* codes,
                  float* vectors) const override;
  std::unique_ptr<DistanceComputer> CreateDistanceComputerImpl(
      EncodedVectorView store) const override;

 private:
  const ProductQuantizer& pq_;
  ProductQuantizer* mutable_pq_;
  PQParameters parameters_;
};

}  // namespace hypervec
