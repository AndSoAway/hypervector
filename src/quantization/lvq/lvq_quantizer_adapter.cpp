/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/lvq/lvq_distance_computer.h>
#include <quantization/lvq/lvq_quantizer_adapter.h>
#include <utils/log/assert.h>

#include <memory>
#include <utility>

namespace hypervec {

LocalVectorQuantizerAdapter::LocalVectorQuantizerAdapter(
    LocalVectorQuantizer& lvq, LVQParameters parameters)
    : lvq_(lvq), mutable_lvq_(&lvq), parameters_(std::move(parameters)) {}

LocalVectorQuantizerAdapter::LocalVectorQuantizerAdapter(
    const LocalVectorQuantizer& lvq)
    : lvq_(lvq), mutable_lvq_(nullptr) {}

void LocalVectorQuantizerAdapter::TrainImpl(idx_t count, const float* vectors) {
  HYPERVEC_THROW_IF_NOT_MSG(
      mutable_lvq_ != nullptr,
      "LocalVectorQuantizerAdapter: cannot train a read-only quantizer");
  mutable_lvq_->Train(count, vectors, parameters_);
}

void LocalVectorQuantizerAdapter::EncodeImpl(idx_t count, const float* vectors,
                                             uint8_t* codes) const {
  lvq_.ComputeCodes(count, vectors, codes);
}

void LocalVectorQuantizerAdapter::DecodeImpl(idx_t count, const uint8_t* codes,
                                             float* vectors) const {
  lvq_.DecodeBatch(count, codes, vectors);
}

std::unique_ptr<DistanceComputer>
LocalVectorQuantizerAdapter::CreateDistanceComputerImpl(
    EncodedVectorView store) const {
  return std::make_unique<LVQDistanceComputer>(lvq_, store.Data(),
                                               store.CodeSize());
}

}  // namespace hypervec
