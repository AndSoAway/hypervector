/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/hnsw/index_hnsw.h>
#include <quantization/rabitq/index_rabitq.h>

#include <cstddef>
#include <cstdint>

namespace hypervec {

/** HNSW graph backed by RaBitQ-compressed storage, with a raw-vector scaffold
 *  retained during graph construction (dual-storage mode).
 *
 *  Build: raw vectors are stored in `raw_storage` (an IndexFlatL2) and used
 *  to compute graph-construction distances; the same vectors are also encoded
 *  into the inherited `storage` (an IndexRaBitQ). After bulk Add the caller
 *  may call `Freeze()` to release `raw_storage`; afterwards the index is
 *  read-only but uses ~B*D/8 bytes per vector instead of D*4.
 *
 *  Search: bitwise distance estimation via RaBitQDistanceComputer obtained
 *  from `storage`. Graph traversal uses approximate distances; final ranking
 *  is the RaBitQ estimate. No re-ranking in this iteration — recall trades
 *  off against RaBitQ parameter B.
 *
 *  T1 scope: kMetricL2, B=1 (original RaBitQ).
 */
struct IndexHNSWRaBitQ : IndexHNSW {
    /// Raw-vector scaffold used during graph construction. Owned by this
    /// index; deleted by Freeze() or the destructor.
    Index* raw_storage = nullptr;

    /// Default ctor for deserialization.
    IndexHNSWRaBitQ();

    /** @param d        vector dimension
     *  @param B        bits per dimension for RaBitQ (1 = original)
     *  @param M_hnsw   HNSW out-degree at levels >= 1 (level 0 is 2*M_hnsw)
     *  @param metric   distance metric; T1 requires kMetricL2 */
    IndexHNSWRaBitQ(int d, int B, int M_hnsw,
                    MetricType metric = kMetricL2);

    ~IndexHNSWRaBitQ() override;

    /// Train the embedded RaBitQ quantizer (sample orthogonal matrix).
    void Train(idx_t n, const float* x) override;

    /// Add vectors to BOTH stores, then build the HNSW graph using raw-vector
    /// distances. Throws if `raw_storage` is null (frozen/deserialized).
    void Add(idx_t n, const float* x) override;

    /// Clear the graph and the data of whichever stores currently exist.
    void Reset() override;

    /// Drop the raw-vector scaffold; the index becomes read-only.
    void Freeze();

    /// Forwarders to storage(=IndexRaBitQ) for the standalone codec interface.
    size_t SaCodeSize() const override;
    void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;
    void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// Not supported; throws with a clear message.
    void Search1(const float* x, ResultHandler& handler,
                 SearchParameters* params = nullptr) const override;
    void RangeSearch(idx_t n, const float* x, float radius,
                     RangeSearchResult* result,
                     const SearchParameters* params = nullptr) const override;
};

}  // namespace hypervec