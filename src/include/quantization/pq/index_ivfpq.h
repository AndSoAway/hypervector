/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/ivf/index_ivf.h>
#include <quantization/pq/pq.h>

#include <vector>

namespace hypervec {

/** Inverted file with Product-Quantized residuals.
 *
 *  Each vector x is assigned to a coarse cell c_i (IVF base behaviour); its
 *  residual r = x - c_i is then PQ-encoded and stored in the inverted list
 *  for cell i. Search probes nprobe cells: per (query, probed list) it
 *  computes a distance table over the residual query (q - c_i), looks up
 *  per-code partial sums, and merges the top-k via a max-heap.
 *
 *  T1+T2 scope: kMetricL2 only. Two search paths share the same disk
 *  format and final results:
 *    - use_precomputed_table = 0: per-(query, probe) ADC table from scratch
 *      (ComputeDistanceTable at scan time). Lower memory, ~dsub× more flops
 *      per probed list.
 *    - use_precomputed_table = 1: per-(coarse-cell, m, k) term cached at
 *      training time; per-query inner-product table; per-probe combine via
 *      one subtraction. Faster but adds nlist*M*ksub floats of state.
 *
 *  Persistence 4cc: "IVPQ".
 */
struct IndexIVFPQ : IndexIVF {
  /// Embedded product quantizer (trained on residuals when by_residual=true).
  ProductQuantizer pq;

  /// If true, PQ encodes residuals (x - c_i); if false, raw x. Default true.
  bool by_residual = true;

  /// 0 = compute ADC table per-probe at scan time; 1 = use the precomputed
  /// table cache. Set this BEFORE Train() (or call PrecomputeTable() after
  /// flipping it on a trained index). Default 0.
  int use_precomputed_table = 0;

  /// Cache of size nlist * M * ksub floats. Populated by PrecomputeTable().
  /// Layout: precomputed_table[(i * M + m) * ksub + k] = ||p_{m,k}||² + 2*<c_i, p_{m,k}>
  std::vector<float> precomputed_table;

  /// Default constructor — for deserialization only.
  IndexIVFPQ();

  /** @param d       vector dimension (must be divisible by M)
   *  @param nlist   number of inverted lists (coarse cells)
   *  @param M       number of subquantizers
   *  @param nbits   bits per code (1..HYPERVEC_PQ_MAX_NBITS)
   *  @param metric  distance metric (T1+T2: kMetricL2 only) */
  IndexIVFPQ(idx_t d, idx_t nlist, idx_t M, int nbits,
             MetricType metric = kMetricL2);

  /** Train the coarse quantizer on x, then train the PQ on residuals
   *  (if by_residual) or on x directly. If use_precomputed_table is set,
   *  populate the cache as well. */
  void Train(idx_t n, const float* x) override;

  /** Encode raw vectors. For by_residual=true this re-runs the coarse
   *  assignment internally because the IndexIVF base contract for
   *  EncodeVectors does not pass list ids in. AddWithIds is overridden to
   *  avoid this redundancy. */
  void EncodeVectors(idx_t n, const float* x, uint8_t* codes) const override;

  /** Override AddWithIds to thread coarse assignments through to PQ
   *  encoding without re-running FindNearestCentroids. */
  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;

  /** Per-query, per-probe ADC scan; max-heap top-k. Selects between the
   *  precomputed-table fast path and the per-probe table path based on
   *  use_precomputed_table. */
  void SearchPreassigned(idx_t n, const float* x, idx_t k,
                         const idx_t* list_ids, const float* centroid_dis,
                         float* distances, idx_t* labels, idx_t nprobe_actual,
                         const IDSelector* sel) const override;

  /** Reconstruct an indexed vector by id: decode the PQ code and add the
   *  coarse centroid back if by_residual. O(n_total) — scans all lists. */
  void Reconstruct(idx_t key, float* recons) const override;

  /** (Re)build precomputed_table from the current coarse + PQ centroids.
   *  Called automatically by Train() when use_precomputed_table != 0; can
   *  also be called manually after toggling the flag on a trained index. */
  void PrecomputeTable();
};

}  // namespace hypervec
