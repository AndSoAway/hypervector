/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/rabitq/rabitq.h>

#include <utils/log/exception.h>
#include <utils/distances/distances.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace hypervec {

// ===========================================================================
//  Construction and utilities
// ===========================================================================

RaBitQQuantizer::RaBitQQuantizer(int d, int B)
    : d(d), B(B) {
    HYPERVEC_THROW_IF_NOT_MSG(d > 0,
        "RaBitQQuantizer: dimension must be positive");
    HYPERVEC_THROW_IF_NOT_FMT(B >= 1 && B <= HYPERVEC_RABITQ_MAX_B,
        "RaBitQQuantizer: bits per dimension must be in [1, %d]",
        HYPERVEC_RABITQ_MAX_B);
    SetDerivedValues();
}

void RaBitQQuantizer::SetDerivedValues() {
    code_size = (static_cast<size_t>(B) * static_cast<size_t>(d) + 7) / 8;

    // Extended RaBitQ (B>1): map integer codes to unit-norm vectors.
    // B=1 uses sign-bit encoding; shift/scale are computed for consistency.
    float sqrt_d = std::sqrt(static_cast<float>(d));
    float half_range = std::pow(2.0f, static_cast<float>(B - 1));
    shift = -(half_range - 0.5f) / sqrt_d;
    scale = 1.0f / sqrt_d;
}

// ===========================================================================
//  Training
// ===========================================================================

void RaBitQQuantizer::Train(idx_t n, const float* x) {
    // RaBitQ does not need data-dependent training; just sample the random
    // orthogonal matrix.
    (void)n;
    (void)x;
    if (is_trained) {
        return;
    }
    RandomGenerator rng(HYPERVEC_RABITQ_DEFAULT_SEED);
    rot = RandomOrthogonalMatrix(d, rng);
    is_trained = true;
}

// ===========================================================================
//  Encoding / Decoding
// ===========================================================================

void RaBitQQuantizer::ComputeCode(const float* x, uint8_t* code) const {
    HYPERVEC_THROW_IF_NOT_MSG(is_trained,
        "RaBitQQuantizer not trained");

    if (B == 1) {
        // Original RaBitQ: sign(P^{-1} · x)
        rot.ComputeSignBitsOne(x, code);
    } else {
        // Extended RaBitQ: quantize each dimension of P^{-1}·x to B bits
        // 1. Inverse transform
        std::vector<float> transformed(static_cast<size_t>(d));
        rot.InverseTransform(1, x, transformed.data());

        // 2. Quantize each dimension
        const float inv_scale = 1.0f / scale;
        const int max_val = (1 << B) - 1;
        for (int j = 0; j < d; j++) {
            float val = (transformed[static_cast<size_t>(j)] - shift) * inv_scale;
            // Clamp to [0, max_val] and round
            int q = static_cast<int>(std::round(val));
            q = std::max(0, std::min(q, max_val));
            // Pack B bits into code (little-endian bit order within each byte)
            int bit_pos = j * B;
            for (int b = 0; b < B; b++) {
                if (q & (1 << b)) {
                    int byte_idx = (bit_pos + b) >> 3;
                    int bit_idx = (bit_pos + b) & 7;
                    code[byte_idx] |= (1 << bit_idx);
                }
            }
        }
    }
}

void RaBitQQuantizer::ComputeCodes(idx_t n, const float* x,
                                    uint8_t* codes) const {
#pragma omp parallel for if (n > 100)
    for (idx_t i = 0; i < n; i++) {
        ComputeCode(x + i * d, codes + i * code_size);
    }
}

void RaBitQQuantizer::Decode(const uint8_t* code, float* x) const {
    HYPERVEC_THROW_IF_NOT_MSG(is_trained,
        "RaBitQQuantizer not trained");

    std::vector<float> raw(static_cast<size_t>(d));

    if (B == 1) {
        // Each bit -> ±1/√D
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const size_t nbytes = (static_cast<size_t>(d) + 7) / 8;
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            int byte_idx = j >> 3;
            int bit_idx = j & 7;
            bool bit = (byte_idx < nbytes) ?
                       ((code[byte_idx] >> bit_idx) & 1) : 0;
            raw[j] = bit ? inv_sqrt_d : -inv_sqrt_d;
        }
    } else {
        // Extract B-bit values, then reverse: x = shift + scale * q
        for (int j = 0; j < d; j++) {
            int q = 0;
            int bit_pos = j * B;
            for (int b = 0; b < B; b++) {
                int byte_idx = (bit_pos + b) >> 3;
                int bit_idx = (bit_pos + b) & 7;
                if (code[byte_idx] & (1 << bit_idx)) {
                    q |= (1 << b);
                }
            }
            raw[static_cast<size_t>(j)] = shift + scale * static_cast<float>(q);
        }
    }

    // Rotate back: x = P · raw
    rot.Transform(1, raw.data(), x);
}

void RaBitQQuantizer::DecodeBatch(idx_t n, const uint8_t* codes,
                                   float* x) const {
#pragma omp parallel for if (n > 100)
    for (idx_t i = 0; i < n; i++) {
        Decode(codes + i * code_size, x + i * d);
    }
}

// ===========================================================================
//  Distance estimation — single-code bitwise path (original RaBitQ, B=1)
// ===========================================================================

void RaBitQQuantizer::PreprocessQuery(
        const float* q, float* q_transformed,
        float& query_norm, float& dot_offset) const {
    // 1. Compute norm
    query_norm = std::sqrt(fvec_norm_L2sqr(q, static_cast<size_t>(d)));

    // 2. Normalize and inverse transform
    if (query_norm > 1e-10f) {
        float inv_norm = 1.0f / query_norm;
        std::vector<float> q_normalized(static_cast<size_t>(d));
        for (int j = 0; j < d; j++) {
            q_normalized[static_cast<size_t>(j)] = q[j] * inv_norm;
        }
        rot.InverseTransform(1, q_normalized.data(), q_transformed);
    } else {
        // Zero vector: just transform zero
        std::vector<float> zeros(static_cast<size_t>(d), 0.0f);
        rot.InverseTransform(1, zeros.data(), q_transformed);
    }

    // 3. dot_offset for formula (2): this is the squared norm of the query,
    //    to be added back when combining with data norms.
    dot_offset = query_norm * query_norm;
}

void RaBitQQuantizer::QuantizeQuery(
        const float* q_transformed, uint8_t* q_quantized,
        float& v_l, float& delta, int B_q) const {
    if (B_q <= 0 || B_q > 8) {
        B_q = HYPERVEC_RABITQ_DEFAULT_BQ;
    }

    // 1. Find min and max
    float v_r = q_transformed[0];
    v_l = q_transformed[0];
    for (int j = 1; j < d; j++) {
        float val = q_transformed[j];
        if (val < v_l) v_l = val;
        if (val > v_r) v_r = val;
    }

    // 2. Compute delta
    float range = v_r - v_l;
    int nlevels = (1 << B_q) - 1;
    delta = (range > 1e-20f) ? (range / static_cast<float>(nlevels)) : 1e-10f;
    float inv_delta = 1.0f / delta;

    // 3. Quantize with randomized rounding
    //    Use a fixed, well-distributed seed to ensure true reproducibility
    //    and avoid bias from stack-address-derived seeds.
    RandomGenerator rng(HYPERVEC_RABITQ_DEFAULT_SEED);
    for (int j = 0; j < d; j++) {
        float offset = (q_transformed[j] - v_l) * inv_delta;
        float u = rng.rand_float();  // Uniform[0,1] for randomized rounding
        int q = static_cast<int>(std::floor(offset + u));
        q = std::max(0, std::min(q, nlevels));
        q_quantized[j] = static_cast<uint8_t>(q);
    }
}

void RaBitQQuantizer::ComputeBitPlanes(
        const uint8_t* q_quantized, uint8_t** q_bits, int d) const {
    int B_q = HYPERVEC_RABITQ_DEFAULT_BQ;
    size_t nbytes = (static_cast<size_t>(d) + 7) / 8;

    for (int bp = 0; bp < B_q; bp++) {
        std::memset(q_bits[bp], 0, nbytes);
        for (int j = 0; j < d; j++) {
            if (q_quantized[j] & (1 << bp)) {
                q_bits[bp][static_cast<size_t>(j) >> 3] |=
                    (1 << (static_cast<size_t>(j) & 7));
            }
        }
    }
}

float RaBitQQuantizer::ComputeSingleCode(
        const uint8_t* code, const uint8_t** q_bits,
        int sum_q, float v_l, float delta, int popcnt_code) const {
    // Formula (20) from RaBitQ paper:
    //   ⟨ō, q⟩ = (2Δ/√D)·⟨x̄_b, q̄_u⟩ + (2v_l/√D)·popcount(x̄_b)
    //            - (Δ/√D)·sum(q̄_u) - √D·v_l
    //
    // where ⟨x̄_b, q̄_u⟩ = Σ_j 2^j · popcount(x̄_b & q̄_u^(j))

    const size_t nbytes = (static_cast<size_t>(d) + 7) / 8;
    const int B_q = HYPERVEC_RABITQ_DEFAULT_BQ;

    // Compute ⟨x̄_b, q̄_u⟩ = Σ_j 2^j · popcount(x̄_b & q_bits[j])
    int inner = 0;
    for (int bp = 0; bp < B_q; bp++) {
        int cnt = 0;
        const uint8_t* bp_data = q_bits[bp];
        for (size_t b = 0; b < nbytes; b++) {
            cnt += __builtin_popcount(static_cast<unsigned int>(
                code[b] & bp_data[b]));
        }
        inner += cnt * (1 << bp);
    }

    float sqrt_d = std::sqrt(static_cast<float>(d));
    float inv_sqrt_d = 1.0f / sqrt_d;

    // Compute ⟨ō, q⟩ using formula (20)
    float result = (2.0f * delta * inv_sqrt_d) * static_cast<float>(inner) +
                   (2.0f * v_l * inv_sqrt_d) * static_cast<float>(popcnt_code) -
                   (delta * inv_sqrt_d) * static_cast<float>(sum_q) -
                   sqrt_d * v_l;

    return result;
}

void RaBitQQuantizer::PrecomputePopcounts(
        const uint8_t* codes, idx_t n, int* popcnts) const {
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        int cnt = 0;
        const uint8_t* c = codes + i * code_size;
        for (size_t b = 0; b < code_size; b++) {
            cnt += __builtin_popcount(static_cast<unsigned int>(c[b]));
        }
        popcnts[i] = cnt;
    }
}

// ===========================================================================
//  Distance estimation — SIMD batch path (FastScan-style)
// ===========================================================================

void RaBitQQuantizer::ComputeDistanceTable(
        const float* q, float* dis_table) const {
    // Build per-4-bit-subsegment LUTs, analogous to PQx4fs.
    // Called once per query; ApplyDistanceTable uses the table for fast
    // per-code estimation via SIMD shuffle instructions.

    // Preprocess query
    float query_norm, dot_offset;
    std::vector<float> q_transformed(static_cast<size_t>(d));
    PreprocessQuery(q, q_transformed.data(), query_norm, dot_offset);

    // Quantize
    float v_l, delta;
    std::vector<uint8_t> q_quantized(static_cast<size_t>(d));
    QuantizeQuery(q_transformed.data(), q_quantized.data(), v_l, delta);

    // Build LUT: for each 4-bit subsegment, precompute the contribution
    // of each possible 4-bit pattern.
    //
    // The contribution of a 4-bit pattern p at subsegment m is:
    //   (2*Δ/√D) · Σ_{bit=0..3} (bit(p, bit) * 2^bit * q_quantized_sub[m])
    // But we need to split this by the bit-plane decomposition.
    //
    // Actually, for the batch path we adopt the PQx4fs strategy:
    // After quantizing the query to 4-bit, we split the d-dimensional
    // quantized vector into d/4 groups of 4 bits each. For each group,
    // we precompute the dot-product contribution for all 16 possible
    // 4-bit code patterns. This allows SIMD shuffle-based lookup.

    const float sqrt_d = std::sqrt(static_cast<float>(d));
    const float coeff = 2.0f * delta / sqrt_d;
    const int n_sub = d / 4;

    for (int m = 0; m < n_sub; m++) {
        for (int p = 0; p < 16; p++) {
            float sum = 0.0f;
            for (int bit = 0; bit < 4; bit++) {
                if (p & (1 << bit)) {
                    int idx = m * 4 + bit;
                    int q_val = q_quantized[idx];
                    sum += static_cast<float>((1 << bit) * q_val);
                }
            }
            dis_table[m * 16 + p] = coeff * sum;
        }
    }
    (void)v_l;
    (void)dot_offset;
}

float RaBitQQuantizer::ApplyDistanceTable(
        const float* dis_table, const uint8_t* code) const {
    // Apply precomputed distance table to one code.
    // For each 4-bit subsegment, extract the 4-bit value from the code
    // and look up the contribution.
    const int n_sub = d / 4;
    float acc = 0.0f;

    for (int m = 0; m < n_sub; m++) {
        // Extract 4-bit value from code at position m
        int byte_idx = (m * 4) >> 3;
        int bit_off = (m * 4) & 7;
        int nibble = (code[byte_idx] >> bit_off) & 0x0F;
        acc += dis_table[m * 16 + nibble];
    }

    return acc;
}

// ===========================================================================
//  Combined estimation
// ===========================================================================

float RaBitQQuantizer::EstimateInnerProduct(
        const uint8_t* code, float inner_product,
        const void* precomputed) const {
    // Placeholder: called from RaBitQDistanceComputer / IndexIVFRaBitQ
    // which manage the query-side precomputed state themselves.
    (void)code;
    (void)inner_product;
    (void)precomputed;
    return 0.0f;
}

float RaBitQQuantizer::EstimateDistance(
        float ip_est, float norm_o, float norm_q,
        float dot_offset) const {
    // Formula (2) from RaBitQ paper:
    //   ∥o_r - q_r∥² = ∥o_r - c∥² + ∥q_r - c∥²
    //                  - 2·∥o_r - c∥·∥q_r - c∥·⟨ō, q⟩/⟨ō, o⟩
    //
    // ip_est = ⟨ō, q⟩ / ⟨ō, o⟩  (the estimated inner product)
    // norm_o = ∥o_r - c∥
    // norm_q = ∥q_r - c∥
    // dot_offset = ∥q_r - c∥² (already included as norm_q²)
    (void)dot_offset;
    return norm_o * norm_o + norm_q * norm_q -
           2.0f * norm_o * norm_q * ip_est;
}

float RaBitQQuantizer::ComputeErrorBound(float inner_product_o_o) const {
    // Formula (14) from RaBitQ paper:
    //   error_bound = √((1 - ip²) / ip²) · ε₀ / √(D - 1)
    // where ip = ⟨ō, o⟩
    float ip = inner_product_o_o;
    if (ip <= 1e-10f) {
        return 10.0f;  // conservative fallback
    }
    float ip_sq = ip * ip;
    float factor = std::sqrt((1.0f - ip_sq) / ip_sq);
    float eps0 = HYPERVEC_RABITQ_DEFAULT_EPSILON0;
    float sqrt_d_minus_1 = std::sqrt(static_cast<float>(d) - 1.0f);
    return factor * eps0 / sqrt_d_minus_1;
}

}  // namespace hypervec