/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/rabitq/index_rabitq.h>
#include <quantization/rabitq/rabitq_distance_computer.h>

#include <utils/distances/distances.h>
#include <utils/log/exception.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace hypervec {

// ===========================================================================
//  Construction
// ===========================================================================

IndexRaBitQ::IndexRaBitQ() : Index(0, kMetricL2) {
    is_trained = false;
}

IndexRaBitQ::IndexRaBitQ(idx_t d, int B, MetricType metric)
    : Index(d, metric), rabitq(static_cast<int>(d), B) {
    HYPERVEC_THROW_IF_NOT_FMT(metric == kMetricL2,
                              "IndexRaBitQ: T1 supports kMetricL2 only, got "
                              "metric=%d", static_cast<int>(metric));
    is_trained = false;
    centroid.assign(static_cast<size_t>(d > 0 ? d : 0), 0.0f);
}

// ===========================================================================
//  Training
// ===========================================================================

void IndexRaBitQ::Train(idx_t n, const float* x) {
    rabitq.Train(n, x);

    // If training data is provided, compute the global centroid for centering.
    if (n > 0 && x != nullptr) {
        if (centroid.size() != static_cast<size_t>(d)) {
            centroid.assign(static_cast<size_t>(d), 0.0f);
        }
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < d; j++) {
                centroid[static_cast<size_t>(j)] += x[i * d + j];
            }
        }
        float inv_n = 1.0f / static_cast<float>(n);
        for (size_t j = 0; j < centroid.size(); j++) {
            centroid[j] *= inv_n;
        }
    }

    is_trained = rabitq.is_trained;
}

// ===========================================================================
//  Add
// ===========================================================================

void IndexRaBitQ::Add(idx_t n, const float* x) {
    HYPERVEC_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    const idx_t d_sz = d;
    const size_t code_sz = rabitq.code_size;

    // Per-vector metadata: norm against centroid, unit vector, encoded code
    std::vector<uint8_t> new_codes(static_cast<size_t>(n) * code_sz);
    std::vector<float> new_ips(static_cast<size_t>(n));
    std::vector<float> new_norms(static_cast<size_t>(n));
    std::vector<float> centered(d_sz);
    std::vector<float> decoded(d_sz);
    const float* cent = centroid.empty() ? nullptr : centroid.data();

    for (idx_t i = 0; i < n; i++) {
        const float* xv = x + i * d;

        // Center: o_c = x - c
        for (idx_t j = 0; j < d_sz; j++) {
            centered[static_cast<size_t>(j)] =
                xv[static_cast<size_t>(j)] - (cent ? cent[static_cast<size_t>(j)] : 0.0f);
        }

        // Norm of centered vector
        float norm = std::sqrt(fvec_norm_L2sqr(centered.data(),
                                               static_cast<size_t>(d_sz)));
        new_norms[static_cast<size_t>(i)] = norm;

        // Normalize and encode
        if (norm > 1e-10f) {
            float inv_norm = 1.0f / norm;
            for (idx_t j = 0; j < d_sz; j++) {
                centered[static_cast<size_t>(j)] *= inv_norm;
            }
        }
        rabitq.rot.ComputeSignBitsOne(centered.data(),
                                      new_codes.data() + static_cast<size_t>(i) * code_sz);

        // Decode code → unit vector ō
        rabitq.Decode(new_codes.data() + static_cast<size_t>(i) * code_sz,
                      decoded.data());

        // Inner product ⟨ō, o⟩ with the normalized centered vector (o = centered after /norm)
        float ip = 0.0f;
        for (idx_t j = 0; j < d_sz; j++) {
            ip += decoded[static_cast<size_t>(j)] * centered[static_cast<size_t>(j)];
        }
        new_ips[static_cast<size_t>(i)] = ip;
    }

    // Append
    codes.insert(codes.end(), new_codes.begin(), new_codes.end());
    inner_products.insert(inner_products.end(), new_ips.begin(), new_ips.end());
    norms.insert(norms.end(), new_norms.begin(), new_norms.end());
    n_total += n;
}

// ===========================================================================
//  Search (brute force via bitwise distance estimation)
// ===========================================================================

void IndexRaBitQ::Search(idx_t n, const float* x, idx_t k, float* distances,
                         idx_t* labels, const SearchParameters* params) const {
    HYPERVEC_THROW_IF_NOT(is_trained);
    HYPERVEC_THROW_IF_NOT_MSG(
        params == nullptr || params->sel == nullptr,
        "IndexRaBitQ::Search does not support IDSelector yet");

    const size_t d_sz = static_cast<size_t>(d);
    const size_t code_sz = rabitq.code_size;
    const size_t nbytes_code = (d_sz + 7) / 8;
    const size_t n_total_sz = static_cast<size_t>(n_total);

#pragma omp parallel
    {
        std::vector<float> q_transformed(d_sz);
        std::vector<uint8_t> q_quantized(d_sz);
    uint8_t* bit_planes[HYPERVEC_RABITQ_DEFAULT_BQ];
        std::vector<uint8_t> bp_storage(HYPERVEC_RABITQ_DEFAULT_BQ * ((d_sz + 7) / 8));
        for (int bp = 0; bp < HYPERVEC_RABITQ_DEFAULT_BQ; bp++) {
            bit_planes[bp] = bp_storage.data() + bp * ((d_sz + 7) / 8);
        }

#pragma omp for
        for (idx_t qi = 0; qi < n; qi++) {
            const float* xq = x + qi * d;
            float* heap_dis = distances + qi * k;
            idx_t* heap_ids = labels + qi * k;
            heap_heapify<CMax<float, idx_t>>(static_cast<size_t>(k),
                                             heap_dis, heap_ids);

            // Query preprocessing: center with global centroid, then normalize
            const float* cent = centroid.empty() ? nullptr : centroid.data();
            std::vector<float> q_centered(d_sz);
            for (size_t j = 0; j < d_sz; j++) {
                q_centered[j] = xq[j] - (cent ? cent[j] : 0.0f);
            }
            float q_norm = std::sqrt(fvec_norm_L2sqr(q_centered.data(), d_sz));
            if (q_norm > 1e-10f) {
                float inv_norm = 1.0f / q_norm;
                for (size_t j = 0; j < d_sz; j++) {
                    q_centered[j] *= inv_norm;
                }
            }
            rabitq.rot.InverseTransform(1, q_centered.data(),
                                        q_transformed.data());

            float v_l, delta;
            rabitq.QuantizeQuery(q_transformed.data(), q_quantized.data(),
                                 v_l, delta);
            rabitq.ComputeBitPlanes(q_quantized.data(), bit_planes,
                                    static_cast<int>(d_sz));
            int sum_q = 0;
            for (size_t j = 0; j < d_sz; j++) {
                sum_q += q_quantized[j];
            }

            // Scan all codes
            float threshold = heap_dis[0];
            for (idx_t i = 0; i < n_total; i++) {
                const uint8_t* code = codes.data() +
                                      static_cast<size_t>(i) * code_sz;
                float inner_product = inner_products[static_cast<size_t>(i)];
                float norm_o = norms[static_cast<size_t>(i)];

                int popcnt = 0;
                for (size_t b = 0; b < nbytes_code; b++) {
                    popcnt += __builtin_popcount(
                        static_cast<unsigned int>(code[b]));
                }

                float ip_q_o = rabitq.ComputeSingleCode(
                    code, const_cast<const uint8_t**>(bit_planes),
                    sum_q, v_l, delta, popcnt);

                float ip_est = (inner_product > 1e-10f)
                    ? (ip_q_o / inner_product) : 0.0f;

                float dist_est = rabitq.EstimateDistance(
                    ip_est, norm_o, q_norm, 0.0f);

                if (CMax<float, idx_t>::cmp(threshold, dist_est)) {
                    heap_replace_top<CMax<float, idx_t>>(
                        static_cast<size_t>(k), heap_dis, heap_ids,
                        dist_est, i);
                    threshold = heap_dis[0];
                }
            }

            heap_reorder<CMax<float, idx_t>>(static_cast<size_t>(k),
                                             heap_dis, heap_ids);
        }
    }
}

// ===========================================================================
//  Reset / Reconstruct
// ===========================================================================

void IndexRaBitQ::Reset() {
    codes.clear();
    inner_products.clear();
    norms.clear();
    n_total = 0;
}

void IndexRaBitQ::Reconstruct(idx_t key, float* recons) const {
    HYPERVEC_THROW_IF_NOT(is_trained);
    HYPERVEC_THROW_IF_NOT(key >= 0 && key < n_total);
    // Decode to unit vector, then scale by norm approximation
    rabitq.Decode(codes.data() + static_cast<size_t>(key) * rabitq.code_size,
                  recons);
    float norm_o = norms[static_cast<size_t>(key)];
    const float* cent = centroid.empty() ? nullptr : centroid.data();
    for (idx_t j = 0; j < d; j++) {
        recons[j] = recons[j] * norm_o + (cent ? cent[static_cast<size_t>(j)] : 0.0f);
    }
}

// ===========================================================================
//  DistanceComputer
// ===========================================================================

DistanceComputer* IndexRaBitQ::GetDistanceComputer() const {
    HYPERVEC_THROW_IF_NOT(is_trained);
    auto* dc = new RaBitQDistanceComputer(
        &rabitq, codes.data(), inner_products.data(), norms.data(),
        centroid.empty() ? nullptr : centroid.data());
    return dc;
}

// ===========================================================================
//  Standalone codec interface
// ===========================================================================

size_t IndexRaBitQ::SaCodeSize() const {
    HYPERVEC_THROW_IF_NOT(is_trained);
    return rabitq.code_size;
}

void IndexRaBitQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
    HYPERVEC_THROW_IF_NOT(is_trained);
    rabitq.ComputeCodes(n, x, bytes);
}

void IndexRaBitQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
    HYPERVEC_THROW_IF_NOT(is_trained);
    rabitq.DecodeBatch(n, bytes, x);
}

}  // namespace hypervec