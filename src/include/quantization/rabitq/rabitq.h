/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/rabitq/random_orthogonal_matrix.h>

#include <cstdint>
#include <vector>

namespace hypervec {

/// Default number of bits for query vector uniform scalar quantization.
#define HYPERVEC_RABITQ_DEFAULT_BQ 4

/// Default epsilon_0 parameter for error bound confidence interval.
#define HYPERVEC_RABITQ_DEFAULT_EPSILON0 1.9f

/// Maximum bits per dimension supported for extended RaBitQ (B>1).
#define HYPERVEC_RABITQ_MAX_B 8

/// Default random seed for orthogonal matrix generation.
#define HYPERVEC_RABITQ_DEFAULT_SEED 1234

/** RaBitQ quantizer: randomized quantization of high-dimensional vectors.
 *
 *  Original RaBitQ (B=1) quantizes a D-dimensional vector into a D-bit string
 *  using sign(P^{-1}·x). Extended RaBitQ (B>1) quantizes into B*D bits using
 *  a shifted/normalized/rotated integer grid as the codebook.
 *
 *  The quantizer provides:
 *    - Encoding:  ComputeCode(s) — data vector → quantization code
 *    - Decoding:  Decode(s)      — code → reconstructed unit vector
 *    - Distance estimation for ANN via ComputeDistanceTable + ApplyDistanceTable
 *      (SIMD batch mode) or ComputeSingleCode (single-code bitwise mode)
 */
struct RaBitQQuantizer {
    // -----------------------------------------------------------------------
    //  Configuration
    // -----------------------------------------------------------------------
    int d = 0;                     ///< vector dimension
    int B = 1;                     ///< bits per dimension (1 = original, >=1)
    size_t code_size = 0;          ///< code bytes = (B * d + 7) / 8
    bool is_trained = false;       ///< true after Train()

    // -----------------------------------------------------------------------
    //  Codebook parameters (Extended RaBitQ B>1)
    // -----------------------------------------------------------------------
    float shift = 0.0f;            ///< integer-code shift: -(2^{B-1} - 0.5) / sqrt(D)
    float scale = 0.0f;            ///< scaling factor for unit-norm mapping

    // -----------------------------------------------------------------------
    //  Random rotation
    // -----------------------------------------------------------------------
    RandomOrthogonalMatrix rot;    ///< random orthogonal matrix P (JLT)

    RaBitQQuantizer() = default;

    /** Construct an untrained RaBitQ quantizer.
     *
     *  @param d   vector dimension
     *  @param B   bits per dimension (1 = original RaBitQ, larger = finer)
     */
    RaBitQQuantizer(int d, int B);

    // -----------------------------------------------------------------------
    //  Training
    // -----------------------------------------------------------------------

    /** Sample the random orthogonal matrix. No data-dependent training needed.
     *  @param n  number of training vectors (ignored)
     *  @param x  training vectors (ignored)
     */
    void Train(idx_t n, const float* x);

    // -----------------------------------------------------------------------
    //  Encoding / Decoding
    // -----------------------------------------------------------------------

    /** Encode a single vector into `code` (size code_size bytes). */
    void ComputeCode(const float* x, uint8_t* code) const;

    /** Encode n vectors.  Parallelized via OpenMP. */
    void ComputeCodes(idx_t n, const float* x, uint8_t* codes) const;

    /** Decode a single code back to a d-dimensional unit vector. Lossy. */
    void Decode(const uint8_t* code, float* x) const;

    /** Decode n codes. */
    void DecodeBatch(idx_t n, const uint8_t* codes, float* x) const;

    // -----------------------------------------------------------------------
    //  Distance estimation — single-code bitwise path (original RaBitQ)
    // -----------------------------------------------------------------------

    /** Preprocess a raw query vector.
     *
     *  Computes q_transformed = P^{-1} · q_normalized  and extracts constants
     *  needed for distance estimation (formula (2) in the RaBitQ paper).
     *
     *  @param[in]  q               raw query vector, size d
     *  @param[out] q_transformed   P^{-1} · q_normalized, size d
     *  @param[out] query_norm      ∥q_r - c∥ (norm from centroid)
     *  @param[out] dot_offset      constant part of formula (2)
     */
    void PreprocessQuery(const float* q, float* q_transformed,
                         float& query_norm, float& dot_offset) const;

    /** Uniform scalar quantization (with randomized rounding) of q_transformed.
     *
     *  Output: q_quantized  (B_q-bit unsigned integers, one per dimension)
     *          v_l          min value (for formula (20))
     *          delta        quantization step size
     *
     *  @param q_transformed  input, size d
     *  @param q_quantized    output, size d (each element in [0, 2^B_q-1])
     *  @param v_l            output: minimum of q_transformed
     *  @param delta          output: quantization step
     *  @param B_q            bits for query quantization (default 4)
     */
    void QuantizeQuery(const float* q_transformed, uint8_t* q_quantized,
                       float& v_l, float& delta, int B_q = HYPERVEC_RABITQ_DEFAULT_BQ) const;

    /** Decompose a B_q-bit unsigned integer vector into B_q bit planes.
     *
     *  q_bits[j][i] = j-th bit of q_quantized[i], packed as D-bit string.
     *
     *  @param q_quantized  input, size d
     *  @param q_bits       output array of B_q pointers, each pointing to a
     *                      D-bit string stored in ((d+7)/8) bytes
     *  @param d            vector dimension
     */
    void ComputeBitPlanes(const uint8_t* q_quantized, uint8_t** q_bits,
                          int d) const;

    /** Estimate ⟨ō, q⟩ for a single data code using bitwise operations.
     *
     *  Implements formula (20) from the RaBitQ paper:
     *    ⟨ō, q⟩ = (2Δ/√D)·⟨x̄_b, q̄_u⟩ + (2v_l/√D)·popcount(x̄_b)
     *             - (Δ/√D)·sum(q̄_u) - √D·v_l
     *
     *  @param code          data quantization code, size code_size bytes
     *  @param q_bits        B_q bit planes of the quantized query
     *  @param sum_q         precomputed sum of all q_quantized entries
     *  @param v_l           min value from QuantizeQuery
     *  @param delta         quantization step from QuantizeQuery
     *  @param popcnt_code   precomputed popcount of the code
     *  @return              estimated ⟨ō, q⟩
     */
    float ComputeSingleCode(const uint8_t* code, const uint8_t** q_bits,
                            int sum_q, float v_l, float delta,
                            int popcnt_code) const;

    /** Precompute popcounts for a batch of codes (used for ⟨ō, o⟩ denominator). */
    void PrecomputePopcounts(const uint8_t* codes, idx_t n, int* popcnts) const;

    // -----------------------------------------------------------------------
    //  Distance estimation — SIMD batch path (FastScan-style)
    // -----------------------------------------------------------------------

    /** Build per-subsegment LUTs for SIMD batch distance estimation.
     *
     *  Splits the quantized query into d/4 subsegments of 4 bits each,
     *  and builds a 16-entry LUT per subsegment (like PQx4fs).
     *
     *  @param q          raw query vector, size d
     *  @param dis_table  output table, size (d/4) * 16 floats
     */
    void ComputeDistanceTable(const float* q, float* dis_table) const;

    /** Apply precomputed distance table to a single code (SIMD-style). */
    float ApplyDistanceTable(const float* dis_table, const uint8_t* code) const;

    // -----------------------------------------------------------------------
    //  Combined estimation
    // -----------------------------------------------------------------------

    /** Full inner-product estimation from code + precomputed constants.
     *
     *  @param code            quantization code
     *  @param inner_product   ⟨ō, o⟩ (precomputed during indexing)
     *  @param precomputed     structure holding all query-side constants
     *  @return                estimated ⟨o, q⟩
     */
    float EstimateInnerProduct(const uint8_t* code, float inner_product,
                               const void* precomputed) const;

    /** Convert estimated inner product to squared L2 distance (formula (2)). */
    float EstimateDistance(float ip_est, float norm_o, float norm_q,
                           float dot_offset) const;

    /** Compute the sharp error bound for a data vector (formula (14)). */
    float ComputeErrorBound(float inner_product_o_o) const;

    // -----------------------------------------------------------------------
    //  Utilities
    // -----------------------------------------------------------------------

    /** Recompute derived fields after changing d or B. */
    void SetDerivedValues();
};

}  // namespace hypervec