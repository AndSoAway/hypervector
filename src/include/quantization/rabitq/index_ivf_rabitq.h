/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/ivf/index_ivf.h>
#include <quantization/rabitq/rabitq_quantizer.h>

#include <cstdint>
#include <memory>

namespace hypervec {

/** IVF index whose list entries use CPU one-bit RaBitQ codes. */
struct IndexIVFRaBitQ : IndexIVF {
  std::unique_ptr<RaBitQQuantizer> rabitq;
  bool by_residual = true;

  IndexIVFRaBitQ();
  IndexIVFRaBitQ(idx_t d, idx_t nlist,
                 uint64_t seed = HYPERVEC_RABITQ_DEFAULT_SEED,
                 int rotation_rounds = HYPERVEC_RABITQ_DEFAULT_ROTATION_ROUNDS,
                 MetricType metric = kMetricL2);

  IndexCapabilities GetCapabilities() const override;

  void Train(idx_t n, const float* x) override;
  void EncodeVectors(idx_t n, const float* x, uint8_t* codes) const override;
  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;
  void Reconstruct(idx_t key, float* recons) const override;

 protected:
  InvertedListScannerPtr CreateInvertedListScanner() const override;
};

}  // namespace hypervec
