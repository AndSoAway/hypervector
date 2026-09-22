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

namespace hypervec {

struct IndexIVFLVQ : IndexIVF {
  LocalVectorQuantizer lvq;
  bool by_residual = true;

  IndexIVFLVQ();
  IndexIVFLVQ(idx_t d, idx_t nlist, int nbits, MetricType metric = kMetricL2);

  IndexCapabilities GetCapabilities() const override;

  void Train(idx_t n, const float* x) override;
  void EncodeVectors(idx_t n, const float* x, uint8_t* codes) const override;
  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;
  void Reconstruct(idx_t key, float* recons) const override;

 protected:
  InvertedListScannerPtr CreateInvertedListScanner() const override;
};

}  // namespace hypervec
