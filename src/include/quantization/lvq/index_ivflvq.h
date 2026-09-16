/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/ivf/index_ivf.h>
#include <quantization/lvq/lvq.h>

#include <memory>

namespace hypervec {

struct IndexIVFLVQ : IndexIVF {
  LocalVectorQuantizer lvq;
  bool by_residual = true;

  IndexIVFLVQ();
  IndexIVFLVQ(idx_t d, idx_t nlist, idx_t nlocal, int nbits,
              MetricType metric = kMetricL2);

  std::string_view TypeName() const noexcept override { return "ivf_lvq"; }

  IndexCapabilities Capabilities() const noexcept override {
    return IndexCapability::NeedsTrain | IndexCapability::BatchBuild |
           IndexCapability::IncrementalAdd | IndexCapability::KnnSearch |
           IndexCapability::RangeSearch | IndexCapability::Filter |
           IndexCapability::ConcurrentSearch | IndexCapability::Serialize |
           IndexCapability::Clone;
  }

  /** KNN and range search share the base IVF traversal and an LVQ-aware
   *  per-thread scanner. Range distances and the radius comparison use the
   *  same approximate ADC values as KNN, not exact vector-space distances. */

  void Train(idx_t n, const float* x) override;
  void EncodeVectors(idx_t n, const float* x, uint8_t* codes) const override;
  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;
  void Reconstruct(idx_t key, float* recons) const override;

 protected:
  std::unique_ptr<InvertedListScanner> CreateInvertedListScanner()
      const override;
};

}  // namespace hypervec
