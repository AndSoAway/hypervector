/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/pq/pq_distance_computer.h>
#include <quantization/pq/pq_quantizer_adapter.h>
#include <utils/log/assert.h>

#include <memory>
#include <utility>

namespace hypervec {

ProductQuantizerAdapter::ProductQuantizerAdapter(ProductQuantizer& pq,
                                                 PQParameters parameters)
    : pq_(pq), mutable_pq_(&pq), parameters_(std::move(parameters)) {}

ProductQuantizerAdapter::ProductQuantizerAdapter(const ProductQuantizer& pq)
    : pq_(pq), mutable_pq_(nullptr) {}

void ProductQuantizerAdapter::TrainImpl(idx_t count, const float* vectors) {
  HYPERVEC_THROW_IF_NOT_MSG(
      mutable_pq_ != nullptr,
      "ProductQuantizerAdapter: cannot train a read-only quantizer");
  mutable_pq_->Train(count, vectors, parameters_);
}

void ProductQuantizerAdapter::EncodeImpl(idx_t count, const float* vectors,
                                         uint8_t* codes) const {
  pq_.ComputeCodes(count, vectors, codes);
}

void ProductQuantizerAdapter::DecodeImpl(idx_t count, const uint8_t* codes,
                                         float* vectors) const {
  pq_.DecodeBatch(count, codes, vectors);
}

std::unique_ptr<DistanceComputer>
ProductQuantizerAdapter::CreateDistanceComputerImpl(
    EncodedVectorView store) const {
  return std::make_unique<PQDistanceComputer>(pq_, store.Data(),
                                              store.CodeSize());
}

}  // namespace hypervec
