/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <quantization/lvq/lvq.h>
#include <utils/structures/maybe_owned_vector.h>

#include <cstdint>

namespace hypervec {

struct IndexLVQ : Index {
  LocalVectorQuantizer lvq;
  MaybeOwnedVector<uint8_t> codes;

  IndexLVQ();
  IndexLVQ(idx_t d, idx_t nlocal, int nbits, MetricType metric = kMetricL2);

  void Train(idx_t n, const float* x) override;
  void Add(idx_t n, const float* x) override;
  void Search(idx_t n, const float* x, idx_t k, float* distances,
              idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Reset() override;
  void Reconstruct(idx_t key, float* recons) const override;
  DistanceComputer* GetDistanceComputer() const override;
  size_t SaCodeSize() const override;
  void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;
  void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;
};

}  // namespace hypervec
