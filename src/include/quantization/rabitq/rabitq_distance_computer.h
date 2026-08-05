/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/rabitq/rabitq.h>
#include <utils/distances/distance_computer.h>

#include <cstdint>

namespace hypervec {

/** Random-access distance computer for RaBitQ-encoded vectors.
 *
 *  Used by IndexRaBitQ and HNSW+RaBitQ for on-the-fly distance estimation
 *  during graph traversal (HNSW) or exhaustive search.
 *
 *  Stores pointers (borrowed, not owned) to the code array and per-vector
 *  metadata.
 */
struct RaBitQDistanceComputer : DistanceComputer {
    const RaBitQQuantizer* quantizer;   ///< quantizer (owns rot matrix + codec)
    const uint8_t* codes;               ///< encoded data, size n * code_size
    const float* inner_products;        ///< precomputed ⟨ō, o⟩, size n
    const float* norms;                 ///< precomputed ∥o_r - c∥, size n
    const float* centroid;              ///< global centroid c (may be nullptr)

    // Query-side cached state (rebuilt on each SetQuery call)
    uint8_t* q_quantized;               ///< quantized query (d bytes)
    uint8_t* bit_planes[HYPERVEC_RABITQ_DEFAULT_BQ];  ///< B_q bit planes
    float v_l;                          ///< min quantized value
    float delta;                        ///< quantization step
    int sum_q;                          ///< sum of q_quantized entries
    float query_norm;                   ///< ∥q_r∥
    float dot_offset;                   ///< constant from formula (2)

    RaBitQDistanceComputer(const RaBitQQuantizer* quantizer,
                           const uint8_t* codes,
                           const float* inner_products,
                           const float* norms,
                           const float* centroid = nullptr);

    ~RaBitQDistanceComputer() override;

    /** Set a new query vector and precompute all query-side data. */
    void SetQuery(const float* x) override;

    /** Estimated distance between query and data vector i.
     *
     *  Uses the unbiased estimator ⟨ō, q⟩/⟨ō, o⟩ and formula (2)
     *  to produce an estimate of ∥o_r - q_r∥².
     */
    float operator()(idx_t i) override;

    /** Symmetric distance between two encoded vectors (used in HNSW build).
     *
     *  Decodes both codes and computes exact L2 between the reconstructed
     *  unit vectors scaled by their norms.
     */
    float symmetric_dis(idx_t i, idx_t j) override;

    /** Switch the code/ip/norm buffers (for external list iteration). */
    void SetBuffers(const uint8_t* new_codes,
                    const float* new_inner_products,
                    const float* new_norms);
};

}  // namespace hypervec