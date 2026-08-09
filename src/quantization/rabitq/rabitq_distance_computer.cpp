/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/rabitq/rabitq_distance_computer.h>

#include <utils/distances/distances.h>
#include <utils/log/exception.h>

#include <cstring>
#include <vector>

namespace hypervec {

// ===========================================================================
//  Construction / Destruction
// ===========================================================================

RaBitQDistanceComputer::RaBitQDistanceComputer(
        const RaBitQQuantizer* quantizer,
        const uint8_t* codes,
        const float* inner_products,
        const float* norms,
        const float* centroid)
    : quantizer(quantizer), codes(codes),
      inner_products(inner_products), norms(norms),
      centroid(centroid),
      v_l(0.0f), delta(0.0f), sum_q(0),
      query_norm(0.0f), dot_offset(0.0f) {
    // Allocate query-side buffers (sized for the quantizer's dimension)
    const size_t d_sz = static_cast<size_t>(quantizer->d);
    q_quantized = new uint8_t[d_sz];
    const size_t nbytes = (d_sz + 7) / 8;
    for (int bp = 0; bp < HYPERVEC_RABITQ_DEFAULT_BQ; bp++) {
        bit_planes[bp] = new uint8_t[nbytes];
    }
}

RaBitQDistanceComputer::~RaBitQDistanceComputer() {
    delete[] q_quantized;
    for (int bp = 0; bp < HYPERVEC_RABITQ_DEFAULT_BQ; bp++) {
        delete[] bit_planes[bp];
    }
}

// ===========================================================================
//  Query setup
// ===========================================================================

void RaBitQDistanceComputer::SetQuery(const float* x) {
    const size_t d_sz = static_cast<size_t>(quantizer->d);

    // 1. Center query: q_c = x - c  (c = global centroid, or 0 if none)
    std::vector<float> q_centered(d_sz);
    for (size_t j = 0; j < d_sz; j++) {
        q_centered[j] = x[j] - (centroid ? centroid[j] : 0.0f);
    }

    // 2. Norm of the centered query ∥q_r - c∥
    query_norm = std::sqrt(fvec_norm_L2sqr(q_centered.data(), d_sz));

    // 3. Normalize and inverse transform: q' = P^{-1} · q_normalized
    std::vector<float> q_transformed(d_sz);
    if (query_norm > 1e-10f) {
        float inv_norm = 1.0f / query_norm;
        for (size_t j = 0; j < d_sz; j++) {
            q_centered[j] *= inv_norm;
        }
    }
    quantizer->rot.InverseTransform(1, q_centered.data(),
                                    q_transformed.data());
    dot_offset = query_norm * query_norm;

    // 2. Quantize to B_q bits
    quantizer->QuantizeQuery(q_transformed.data(), q_quantized,
                             v_l, delta);

    // 3. Decompose into bit planes
    quantizer->ComputeBitPlanes(q_quantized, bit_planes,
                                static_cast<int>(d_sz));

    // 4. Sum of quantized values
    sum_q = 0;
    for (size_t j = 0; j < d_sz; j++) {
        sum_q += q_quantized[j];
    }
}

// ===========================================================================
//  Distance computation
// ===========================================================================

float RaBitQDistanceComputer::operator()(idx_t i) {
    // Read code + metadata for vector i
    const size_t code_sz = quantizer->code_size;
    const uint8_t* code = codes + static_cast<size_t>(i) * code_sz;
    const float inner_product = inner_products[static_cast<size_t>(i)];
    const float norm_o = norms[static_cast<size_t>(i)];

    // Popcount of the code
    const size_t nbytes = (static_cast<size_t>(quantizer->d) + 7) / 8;
    int popcnt = 0;
    for (size_t b = 0; b < nbytes; b++) {
        popcnt += __builtin_popcount(static_cast<unsigned int>(code[b]));
    }

    // Estimate ⟨ō, q⟩ via bitwise ops
    float ip_q_o = quantizer->ComputeSingleCode(
        code, const_cast<const uint8_t**>(bit_planes),
        sum_q, v_l, delta, popcnt);

    // Unbiased estimator: ⟨ō, q⟩ / ⟨ō, o⟩
    float ip_est = (inner_product > 1e-10f)
        ? (ip_q_o / inner_product) : 0.0f;

    // Formula (2): estimated squared L2 distance
    return quantizer->EstimateDistance(ip_est, norm_o, query_norm,
                                       dot_offset);
}

float RaBitQDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    // Decode both codes and compute exact L2 between reconstructed
    // (unit) vectors, scaled by norms.
    const size_t d_sz = static_cast<size_t>(quantizer->d);
    std::vector<float> xi(d_sz), xj(d_sz);
    const size_t code_sz = quantizer->code_size;
    quantizer->Decode(codes + static_cast<size_t>(i) * code_sz, xi.data());
    quantizer->Decode(codes + static_cast<size_t>(j) * code_sz, xj.data());

    // Scale to approximate original vectors
    const float ni = norms[static_cast<size_t>(i)];
    const float nj = norms[static_cast<size_t>(j)];
    float acc = 0.0f;
    for (size_t k = 0; k < d_sz; k++) {
        float diff = xi[k] * ni - xj[k] * nj;
        acc += diff * diff;
    }
    return acc;
}

// ===========================================================================
//  Buffer switching
// ===========================================================================

void RaBitQDistanceComputer::SetBuffers(
        const uint8_t* new_codes,
        const float* new_inner_products,
        const float* new_norms) {
    codes = new_codes;
    inner_products = new_inner_products;
    norms = new_norms;
}

}  // namespace hypervec