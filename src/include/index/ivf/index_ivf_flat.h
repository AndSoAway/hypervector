/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/ivf/index_ivf.h>

namespace hypervec {

/** IVF index that stores raw float vectors (exact per-cell distance).
 *
 * code_size = d * sizeof(float).  Distances within each inverted list are
 * computed exactly using fvec_L2sqr / fvec_inner_product.  This is the
 * simplest IVF variant and serves as the base for quantized variants. */
struct IndexIVFFlat : IndexIVF {
  explicit IndexIVFFlat(idx_t d = 0, idx_t nlist = 0,
                        MetricType metric = kMetricL2);

  /** Encode: store raw floats as bytes (memcpy). */
  void EncodeVectors(idx_t n, const float* x, uint8_t* codes) const override;

  /** Exact distance search within pre-assigned inverted lists. */
  void SearchPreassigned(idx_t n, const float* x, idx_t k,
                         const idx_t* list_ids, const float* centroid_dis,
                         float* distances, idx_t* labels, idx_t nprobe_actual,
                         const IDSelector* sel) const override;

  /** Reconstruct a vector by scanning all inverted lists for the given ID. */
  void Reconstruct(idx_t key, float* recons) const override;
};

}  // namespace hypervec
