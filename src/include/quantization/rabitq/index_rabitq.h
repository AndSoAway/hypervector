/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/rabitq/rabitq.h>
#include <utils/structures/maybe_owned_vector.h>

#include <cstdint>
#include <vector>

namespace hypervec {

/** Flat (non-IVF) index over RaBitQ-compressed codes.
 *
 *  Storage: a contiguous code array of size n_total * rabitq.code_size bytes,
 *  plus per-vector metadata:
 *    inner_products[i] = ⟨ō, o⟩  (unbiased estimator denominator)
 *    norms[i]          = ∥o_r∥  (distance from origin)
 *
 *  Search: per-query bit-plane decomposition → linear scan of all codes using
 *  bitwise popcount distance estimation (formula 20) → top-k max-heap.
 *
 *  Memory cost is ⌈B·D/8⌉ bytes per vector for B=1 → 16 bytes per 128-dim
 *  vector (32× compression).
 *
 *  Primarily used as the `storage` of HNSW+RaBitQ (IndexHNSWRaBitQ).
 */
struct IndexRaBitQ : Index {
    /// Embedded RaBitQ quantizer (owns the random orthogonal matrix).
    RaBitQQuantizer rabitq;

    /// Encoded dataset, size n_total * rabitq.code_size.
    std::vector<uint8_t> codes;

    /// ⟨ō, o⟩ per vector (weight denominator), size n_total.
    std::vector<float> inner_products;

    /// ∥o_r - c∥ per vector (formula (2) term), size n_total.
    std::vector<float> norms;

    /// Global centroid c used for centering before normalization.
    /// Computed in Train(); used by Add/Reconstruct/DistanceComputer.
    std::vector<float> centroid;

    /// Default constructor for deserialization.
    IndexRaBitQ();

    /** @param d       vector dimension
     *  @param B       bits per dimension (1 = original RaBitQ)
     *  @param metric  distance metric; T1 requires kMetricL2 */
    IndexRaBitQ(idx_t d, int B, MetricType metric = kMetricL2);

    /** Train the underlying quantizer (sample random orthogonal matrix). */
    void Train(idx_t n, const float* x) override;

    /** Encode and append the n vectors to the code array. */
    void Add(idx_t n, const float* x) override;

    /** Brute-force search via bitwise distance estimation. */
    void Search(idx_t n, const float* x, idx_t k, float* distances,
                idx_t* labels,
                const SearchParameters* params = nullptr) const override;

    /// Drop all stored codes and metadata; the trained quantizer is kept.
    void Reset() override;

    /// Lossy reconstruction by decoding stored code [key].
    void Reconstruct(idx_t key, float* recons) const override;

    /** Random-access distance computer over `codes`. The returned object
     *  borrows `rabitq` and `codes`; caller owns and must delete. */
    DistanceComputer* GetDistanceComputer() const override;

    /// Standalone codec interface (mirrors RaBitQQuantizer's API).
    size_t SaCodeSize() const override;
    void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;
    void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;
};

}  // namespace hypervec