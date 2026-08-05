/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/rabitq/rabitq.h>
#include <index/flat/index_flat.h>
#include <index/ivf/index_ivf.h>

#include <vector>

namespace hypervec {

/** IVF index with RaBitQ-compressed vectors and optional error-bound-based
 *  re-ranking.
 *
 *  Inherits the IVF framework (k-means clustering + inverted lists) from
 *  IndexIVF and implements EncodeVectors / SearchPreassigned using the
 *  RaBitQ quantizer.
 *
 *  Per-list storage layout (one entry):
 *    [code (code_size bytes)] [inner_product (4 bytes)] [norm (4 bytes)]
 *
 *  Re-ranking (optional, see SearchWithRerank) uses raw vectors stored
 *  in `raw_storage` to compute exact L2 distances for candidates that
 *  cannot be safely pruned by the RaBitQ error bound.
 */
struct IndexIVFRaBitQ : IndexIVF {
    RaBitQQuantizer rabitq;           ///< RaBitQ quantizer
    bool by_residual = true;          ///< encode residual o-c (true) or raw o (false)

    /// Raw vector storage for re-ranking.  Populated during Add().
    /// May be left empty (SearchWithRerank falls back to normal Search).
    IndexFlatL2* raw_storage = nullptr;
    bool own_raw_storage = false;

    /** Construct IVF+RaBitQ index.
     *
     *  @param quantizer  coarse quantizer (k-means centroids); ownership passed
     *  @param d          vector dimension
     *  @param nlist      number of inverted lists (clusters)
     *  @param B          bits per dimension for RaBitQ
     *  @param metric     distance metric
     */
    IndexIVFRaBitQ(Index* quantizer, size_t d, size_t nlist, int B,
                   MetricType metric = kMetricL2);

    ~IndexIVFRaBitQ() override;

    // -----------------------------------------------------------------------
    //  IndexIVF interface
    // -----------------------------------------------------------------------

    void Train(idx_t n, const float* x) override;

    /** Encode n raw vectors into RaBitQ codes for inverted-list storage.
     *
     *  For each vector o_r with assigned centroid c:
     *    1. Normalize: o = (o_r - c) / ∥o_r - c∥
     *    2. Transform: o' = P^{-1} · o
     *    3. Encode: code = sign_bits(o')  (B=1) or quantize_B(o')  (B>1)
     *    4. Precompute ⟨ō, o⟩ = ⟨P·code, o⟩ and ∥o_r - c∥
     */
    void EncodeVectors(idx_t n, const float* x, uint8_t* codes) const override;

    /** Override AddWithIds to guarantee centroid_id consistency between
     *  encoding and list insertion. */
    void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;

    /** Encode a single vector with a given centroid (shared helper). */
    void EncodeOneVector(const float* xv, const float* c,
                         uint8_t* code_out, size_t code_sz,
                         size_t d_sz) const;

    /** Standard search — no re-ranking (inherited from IndexIVF). */
    void SearchPreassigned(idx_t n, const float* x, idx_t k,
                           const idx_t* list_ids,
                           const float* centroid_dis,
                           float* distances, idx_t* labels,
                           idx_t nprobe_actual,
                           const IDSelector* sel) const override;

    // -----------------------------------------------------------------------
    //  Re-ranking search (RaBitQ-specific, paper Section 4)
    // -----------------------------------------------------------------------

    /** Search with error-bound-based re-ranking.
     *
     *  1. Run standard IVF+RaBitQ search to collect a pool of candidates.
     *  2. Re-rank candidates using exact L2 distances from raw_storage.
     *  3. Prune candidates whose lower bound (est - error_bound) exceeds
     *     the current k-th best exact distance.  This guarantees the true
     *     NN is re-ranked with high probability (controlled by ε₀).
     *
     *  Requires raw_storage to be populated (call EnableReranking first).
     *  Falls back to normal Search if raw_storage is null.
     *
     *  @param n          number of queries
     *  @param x          query vectors, size n * d
     *  @param k          neighbours per query (final output size)
     *  @param pool_mult  pool size = pool_mult * k  (e.g. pool_mult=3 → collect 3k candidates)
     *  @param distances  output distances, size n * k
     *  @param labels     output labels, size n * k
     *  @param nprobe     IVF probes
     */
    void SearchWithRerank(idx_t n, const float* x, idx_t k, int pool_mult,
                          float* distances, idx_t* labels,
                          idx_t nprobe) const;

    /** Enable re-ranking by storing raw vectors.
     *
     *  When called, all future Add() calls will also store the raw vectors
     *  in an internal IndexFlatL2 so that SearchWithRerank can access them.
     */
    void EnableReranking();

    // -----------------------------------------------------------------------
    //  Progressive pruning (Extended RaBitQ B>1 — stub for future)
    // -----------------------------------------------------------------------

    void SearchWithProgressivePruning(
        idx_t n, const float* x, idx_t k,
        float* distances, idx_t* labels,
        idx_t nprobe,
        const SearchParameters* params = nullptr) const;

    // -----------------------------------------------------------------------
    //  Utilities
    // -----------------------------------------------------------------------

    size_t compute_per_vector_code_size() const;
    IVFSearchParameters* create_default_search_params() const;
};

}  // namespace hypervec