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
  IndexIVFLVQ(idx_t d, idx_t nlist, idx_t nlocal, int nbits,
              MetricType metric = kMetricL2);

  void Train(idx_t n, const float* x) override;
  void EncodeVectors(idx_t n, const float* x, uint8_t* codes) const override;
  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;
  void SearchPreassigned(idx_t n, const float* x, idx_t k,
                         const idx_t* list_ids, const float* centroid_dis,
                         float* distances, idx_t* labels, idx_t nprobe_actual,
                         const IDSelector* sel) const override;
  void Reconstruct(idx_t key, float* recons) const override;
};

}  // namespace hypervec
