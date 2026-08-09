/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/rabitq/index_ivfrabitq.h>

#include <index/flat/index_flat.h>
#include <invlists/inverted_lists.h>
#include <utils/distances/distances.h>
#include <utils/log/exception.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace hypervec {

// ===========================================================================
//  Construction / Destruction
// ===========================================================================

IndexIVFRaBitQ::IndexIVFRaBitQ(Index* quantizer, size_t d, size_t nlist,
                                 int B, MetricType metric)
    : IndexIVF(static_cast<idx_t>(d), static_cast<idx_t>(nlist),
               /*code_size (temporary)*/ 0, metric) {
    rabitq = RaBitQQuantizer(static_cast<int>(d), B);

    size_t per_vec = rabitq.code_size + sizeof(float) * 2;
    if (own_invlists) {
        delete invlists;
    }
    invlists = new ArrayInvertedLists(nlist, per_vec);
    own_invlists = true;

    (void)quantizer;
}

IndexIVFRaBitQ::~IndexIVFRaBitQ() {
    if (own_raw_storage) {
        delete raw_storage;
    }
}

// ===========================================================================
//  Training
// ===========================================================================

void IndexIVFRaBitQ::Train(idx_t n, const float* x) {
    IndexIVF::Train(n, x);
    rabitq.Train(n, x);
    is_trained = true;
}

// ===========================================================================
//  EncodeVectors — fallback (real work is in AddWithIds)
// ===========================================================================

void IndexIVFRaBitQ::EncodeVectors(idx_t n, const float* x,
                                    uint8_t* codes) const {
    HYPERVEC_THROW_IF_NOT_MSG(is_trained,
        "IndexIVFRaBitQ not trained");

    const size_t d_sz = static_cast<size_t>(d);
    const size_t code_sz = rabitq.code_size;
    const size_t per_vec = code_sz + sizeof(float) * 2;

    std::vector<float> centroid_dis(static_cast<size_t>(n));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
    FindNearestCentroids(n, x, 1, centroid_dis.data(), centroid_ids.data());

    for (idx_t i = 0; i < n; i++) {
        EncodeOneVector(x + i * d,
                        centroids.data() + static_cast<size_t>(centroid_ids[i]) * d_sz,
                        codes + i * per_vec, code_sz, d_sz);
    }
}

// ===========================================================================
//  AddWithIds — override to ensure centroid_id consistency
// ===========================================================================

void IndexIVFRaBitQ::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
    HYPERVEC_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    // Find nearest centroid for each vector
    std::vector<float> centroid_dis(static_cast<size_t>(n));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
    FindNearestCentroids(n, x, 1, centroid_dis.data(), centroid_ids.data());

    const size_t d_sz = static_cast<size_t>(d);
    const size_t code_sz = rabitq.code_size;
    const size_t per_vec = code_sz + sizeof(float) * 2;

    // Encode each vector using its assigned centroid
    std::vector<uint8_t> codes(static_cast<size_t>(n) * per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* c = centroids.data() +
                         static_cast<size_t>(centroid_ids[i]) * d_sz;
        EncodeOneVector(x + i * d, c,
                        codes.data() + static_cast<size_t>(i) * per_vec,
                        code_sz, d_sz);
    }

    // Insert into inverted lists
    for (idx_t i = 0; i < n; i++) {
        const idx_t id = (xids != nullptr) ? xids[i] : n_total + i;
        const idx_t list_no = centroid_ids[static_cast<size_t>(i)];
        invlists->add_entry(static_cast<size_t>(list_no), id,
                            codes.data() +
                              static_cast<size_t>(i) * per_vec);
    }

    n_total += n;

    // Also store raw vectors if re-ranking is enabled
    if (raw_storage != nullptr) {
        raw_storage->Add(n, x);
    }
}

// ===========================================================================
//  EncodeOneVector — shared helper
// ===========================================================================

void IndexIVFRaBitQ::EncodeOneVector(const float* xv, const float* c,
                                     uint8_t* code_out, size_t code_sz,
                                     size_t d_sz) const {
    std::vector<float> residual(d_sz);
    for (size_t j = 0; j < d_sz; j++) {
        residual[j] = xv[j] - c[j];
    }

    float norm = std::sqrt(fvec_norm_L2sqr(residual.data(), d_sz));
    float norm_o = norm > 1e-10f ? norm : 1.0f;

    if (norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        for (size_t j = 0; j < d_sz; j++) {
            residual[j] *= inv_norm;
        }
    }

    rabitq.rot.ComputeSignBitsOne(residual.data(), code_out);

    std::vector<float> decoded(d_sz);
    rabitq.Decode(code_out, decoded.data());

    float ip = 0.0f;
    for (size_t j = 0; j < d_sz; j++) {
        ip += decoded[j] * residual[j];
    }

    float* meta = reinterpret_cast<float*>(code_out + code_sz);
    meta[0] = ip;
    meta[1] = norm_o;
}

// ===========================================================================
//  EnableReranking — allocate raw storage
// ===========================================================================

void IndexIVFRaBitQ::EnableReranking() {
    if (raw_storage == nullptr) {
        raw_storage = new IndexFlatL2(static_cast<idx_t>(d));
        own_raw_storage = true;
    }
    // If vectors have already been added, we need to rebuild raw_storage.
    // Since we can't replay Add calls, we only support enabling before Add.
    // After construction, the caller should call EnableReranking before adding.
}

// ===========================================================================
//  SearchPreassigned — standard (no re-ranking)
// ===========================================================================

void IndexIVFRaBitQ::SearchPreassigned(
        idx_t n, const float* x, idx_t k,
        const idx_t* list_ids, const float* centroid_dis,
        float* distances, idx_t* labels,
        idx_t nprobe_actual, const IDSelector* sel) const {
    HYPERVEC_THROW_IF_NOT_MSG(is_trained,
        "IndexIVFRaBitQ not trained");
    HYPERVEC_THROW_IF_NOT_MSG(k > 0, "k must be positive");

    const size_t d_sz = static_cast<size_t>(d);
    const size_t code_sz = rabitq.code_size;
    const size_t per_vec = code_sz + sizeof(float) * 2;
    const size_t nbytes_code = (d_sz + 7) / 8;

#pragma omp parallel
    {
        std::vector<float> residual_query(d_sz);
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

            heap_heapify<CMax<float, idx_t>>(
                static_cast<size_t>(k), heap_dis, heap_ids);

            for (idx_t pi = 0; pi < nprobe_actual; pi++) {
                const idx_t list_no =
                    list_ids[static_cast<size_t>(qi) * nprobe_actual + pi];
                if (list_no < 0) continue;
                const size_t list_sz =
                    invlists->list_size(static_cast<size_t>(list_no));
                if (list_sz == 0) continue;

                const float* c = centroids.data() +
                                 static_cast<size_t>(list_no) * d_sz;

                // Query-side precomputation
                for (size_t j = 0; j < d_sz; j++) {
                    residual_query[j] = xq[j] - c[j];
                }
                float norm_q = std::sqrt(
                    fvec_norm_L2sqr(residual_query.data(), d_sz));

                if (norm_q > 1e-10f) {
                    float inv_norm = 1.0f / norm_q;
                    std::vector<float> q_norm(d_sz);
                    for (size_t j = 0; j < d_sz; j++) {
                        q_norm[j] = residual_query[j] * inv_norm;
                    }
                    rabitq.rot.InverseTransform(
                        1, q_norm.data(), q_transformed.data());
                } else {
                    std::memset(q_transformed.data(), 0, d_sz * sizeof(float));
                }

                float v_l, delta;
                rabitq.QuantizeQuery(
                    q_transformed.data(), q_quantized.data(), v_l, delta);
                rabitq.ComputeBitPlanes(
                    q_quantized.data(), bit_planes, static_cast<int>(d_sz));

                int sum_q = 0;
                for (size_t j = 0; j < d_sz; j++) {
                    sum_q += q_quantized[j];
                }

                InvertedLists::ScopedCodes scoped_codes(
                    invlists, static_cast<size_t>(list_no));
                InvertedLists::ScopedIds scoped_ids(
                    invlists, static_cast<size_t>(list_no));
                const uint8_t* codes_p = scoped_codes.get();
                const idx_t* ids_p = scoped_ids.get();

                float threshold = heap_dis[0];
                for (size_t j = 0; j < list_sz; j++) {
                    if (sel && !sel->IsMember(ids_p[j])) continue;

                    const uint8_t* entry = codes_p + j * per_vec;
                    const uint8_t* code = entry;
                    float inner_product;
                    float norm_o;
                    std::memcpy(&inner_product, entry + code_sz, sizeof(float));
                    std::memcpy(&norm_o, entry + code_sz + sizeof(float),
                                sizeof(float));

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
                        ip_est, norm_o, norm_q, 0.0f);

                    if (CMax<float, idx_t>::cmp(threshold, dist_est)) {
                        heap_replace_top<CMax<float, idx_t>>(
                            static_cast<size_t>(k),
                            heap_dis, heap_ids,
                            dist_est, ids_p[j]);
                        threshold = heap_dis[0];
                    }
                }
            }

            heap_reorder<CMax<float, idx_t>>(
                static_cast<size_t>(k), heap_dis, heap_ids);
        }
    }
}

// ===========================================================================
//  SearchWithRerank — error-bound-based re-ranking
// ===========================================================================

void IndexIVFRaBitQ::SearchWithRerank(
        idx_t n, const float* x, idx_t k, int pool_mult,
        float* distances, idx_t* labels,
        idx_t nprobe) const {
    HYPERVEC_THROW_IF_NOT_MSG(is_trained,
        "IndexIVFRaBitQ not trained");

    // If no raw storage is available, fall back to normal search
    if (raw_storage == nullptr) {
        IVFSearchParameters params;
        params.nprobe = nprobe;
        Search(n, x, k, distances, labels, &params);
        return;
    }

    const idx_t pool_k = k * pool_mult;
    const size_t d_sz = static_cast<size_t>(d);
    const size_t code_sz = rabitq.code_size;
    const size_t per_vec = code_sz + sizeof(float) * 2;
    const size_t nbytes_code = (d_sz + 7) / 8;

    // ---- Phase 1: standard IVF search (collect pool_k candidates) ----
    IVFSearchParameters ivf_params;
    ivf_params.nprobe = nprobe;

    std::vector<float> pool_dist(static_cast<size_t>(n) * pool_k);
    std::vector<idx_t> pool_ids(static_cast<size_t>(n) * pool_k);

    // Find nearest centroids for each query
    std::vector<float> centroid_dis(static_cast<size_t>(n) * nprobe);
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n) * nprobe);
    FindNearestCentroids(n, x, nprobe, centroid_dis.data(),
                         centroid_ids.data());

    // Collect pool via SearchPreassigned, using pool_k instead of k
    // We reuse the outer SearchPreassigned pattern with a larger k.
    // To avoid duplicating the entire function, we run the standard
    // Search with nprobe, then fall through to re-ranking below.

    // Standard search to get pool
    {
        std::vector<float> centroid_dis2(static_cast<size_t>(n) * nprobe);
        std::vector<idx_t> centroid_ids2(static_cast<size_t>(n) * nprobe);
        FindNearestCentroids(n, x, nprobe, centroid_dis2.data(),
                             centroid_ids2.data());

        // We call SearchPreassigned with pool_k to collect a larger candidate set
        SearchPreassigned(n, x, pool_k, centroid_ids2.data(),
                          centroid_dis2.data(), pool_dist.data(),
                          pool_ids.data(), nprobe, nullptr);
    }

    // ---- Phase 2: re-rank with exact distances, pruning by error bound ----
    #pragma omp parallel for
    for (idx_t qi = 0; qi < n; qi++) {
        // Final heap (k elements, best=smallest distance)
        float* final_dis = distances + qi * k;
        idx_t* final_ids = labels + qi * k;
        heap_heapify<CMax<float, idx_t>>(static_cast<size_t>(k),
                                         final_dis, final_ids);

        const float* xq = x + qi * d;

        for (idx_t pi = 0; pi < pool_k; pi++) {
            idx_t id = pool_ids[qi * pool_k + pi];
            if (id < 0) continue;

            float est_dist = pool_dist[qi * pool_k + pi];
            float current_kth = final_dis[0];

            // Re-rank candidate if its estimated distance beats the
            // current k-th exact distance; otherwise skip.
            if (est_dist < current_kth) {
                // Compute exact L2 distance from raw storage
                float true_dist = fvec_L2sqr(
                    xq, raw_storage->GetXb() + static_cast<size_t>(id) * d_sz,
                    d_sz);

                if (CMax<float, idx_t>::cmp(final_dis[0], true_dist)) {
                    heap_replace_top<CMax<float, idx_t>>(
                        static_cast<size_t>(k),
                        final_dis, final_ids, true_dist, id);
                }
            }
        }

        heap_reorder<CMax<float, idx_t>>(static_cast<size_t>(k),
                                         final_dis, final_ids);
    }
}

// ===========================================================================
//  Progressive pruning (Extended RaBitQ B>1 — stub for future)
// ===========================================================================

void IndexIVFRaBitQ::SearchWithProgressivePruning(
        idx_t n, const float* x, idx_t k,
        float* distances, idx_t* labels,
        idx_t nprobe, const SearchParameters* params) const {
    IVFSearchParameters ivf_params;
    ivf_params.nprobe = nprobe;
    Search(n, x, k, distances, labels, &ivf_params);
    (void)params;
}

// ===========================================================================
//  Utilities
// ===========================================================================

size_t IndexIVFRaBitQ::compute_per_vector_code_size() const {
    return rabitq.code_size + sizeof(float) * 2;
}

IVFSearchParameters* IndexIVFRaBitQ::create_default_search_params() const {
    auto* params = new IVFSearchParameters();
    params->nprobe = 8;
    return params;
}

}  // namespace hypervec