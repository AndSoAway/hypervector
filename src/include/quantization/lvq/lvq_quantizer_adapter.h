/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/lvq/lvq.h>
#include <quantization/quantizer.h>

#include <memory>
#include <string_view>

namespace hypervec {

/** Quantizer protocol adapter for an existing LocalVectorQuantizer.
 *
 * The adapter borrows `lvq`; the LocalVectorQuantizer must outlive the adapter
 * and every DistanceComputer created through it. An adapter built from a const
 * model supports encode/decode/distance operations but rejects Train().
 */
class LocalVectorQuantizerAdapter final : public Quantizer {
 public:
  explicit LocalVectorQuantizerAdapter(LocalVectorQuantizer& lvq,
                                       LVQParameters parameters = {});
  explicit LocalVectorQuantizerAdapter(const LocalVectorQuantizer& lvq);

  std::string_view TypeName() const noexcept override { return "lvq"; }
  idx_t Dimension() const noexcept override { return lvq_.d; }
  MetricType Metric() const noexcept override { return kMetricL2; }
  bool NeedsTraining() const noexcept override { return true; }
  bool IsTrained() const noexcept override { return lvq_.is_trained; }
  size_t CodeSize() const noexcept override { return lvq_.code_size; }

 protected:
  void TrainImpl(const DenseDatasetView& data) override;
  void EncodeImpl(idx_t count, const float* vectors,
                  uint8_t* codes) const override;
  void DecodeImpl(idx_t count, const uint8_t* codes,
                  float* vectors) const override;
  std::unique_ptr<DistanceComputer> CreateDistanceComputerImpl(
      EncodedVectorView store) const override;

 private:
  const LocalVectorQuantizer& lvq_;
  LocalVectorQuantizer* mutable_lvq_;
  LVQParameters parameters_;
};

}  // namespace hypervec
