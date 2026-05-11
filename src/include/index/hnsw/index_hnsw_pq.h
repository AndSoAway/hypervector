/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/hnsw/index_hnsw.h>
#include <quantization/pq/index_pq.h>

#include <cstddef>
#include <cstdint>

namespace hypervec {

/** HNSW graph backed by PQ-compressed storage, with a raw-vector scaffold
 *  retained during graph construction (dual-storage mode).
 *
 *  Build: raw vectors are stored in `raw_storage` (an IndexFlatL2) and used
 *  to compute graph-construction distances; the same vectors are also encoded
 *  into the inherited `storage` (an IndexPQ). After bulk Add the caller may
 *  call `Freeze()` to release `raw_storage`; afterwards the index is read-only
 *  but uses ~M*nbits/8 bytes per vector instead of d*4.
 *
 *  Search: ADC via PQDistanceComputer obtained from `storage`. Graph traversal
 *  uses approximate distances; final ranking is the PQ ADC sum. No re-ranking
 *  step is performed in this iteration — recall trades off against PQ
 *  parameters (M, nbits).
 *
 *  Persistence 4cc: "IHNp". Only the inherited `storage` (IndexPQ codes) is
 *  serialized; `raw_storage` is build-time scaffolding and does not survive
 *  a save/load round-trip. A deserialized index is implicitly frozen — calls
 *  to Add throw.
 *
 *  T1 scope: kMetricL2 only. IP/cosine deferred (would require IP-spherical
 *  k-means in PQ training and an IP ADC table).
 */
struct IndexHNSWPQ : IndexHNSW {
  /// Raw-vector scaffold used during graph construction. Owned by this index;
  /// deleted by Freeze() or the destructor. nullptr after deserialization or
  /// after Freeze() — that state makes the index read-only.
  Index* raw_storage = nullptr;

  /// Default ctor for deserialization. Produces an empty, untrained,
  /// dimensionless index with both stores set up by ReadIndex.
  IndexHNSWPQ();

  /** @param d        vector dimension (must be divisible by M_pq)
   *  @param M_pq     number of subquantizers
   *  @param nbits    bits per code (1..HYPERVEC_PQ_MAX_NBITS)
   *  @param M_hnsw   HNSW out-degree at levels >= 1 (level 0 is 2*M_hnsw)
   *  @param metric   distance metric; T1 requires kMetricL2 */
  IndexHNSWPQ(int d, int M_pq, int nbits, int M_hnsw,
              MetricType metric = kMetricL2);

  ~IndexHNSWPQ() override;

  /// Train the embedded ProductQuantizer. The IndexFlatL2 scaffold is
  /// self-trained (no codebook).
  void Train(idx_t n, const float* x) override;

  /// Add vectors to BOTH stores, then build the HNSW graph using raw-vector
  /// distances. Throws if `raw_storage` is null (the index has been frozen
  /// or deserialized).
  void Add(idx_t n, const float* x) override;

  /// Clear the graph and the data of whichever stores currently exist.
  /// Does NOT resurrect `raw_storage` if it was freed — frozen state survives
  /// Reset.
  void Reset() override;

  /// Drop the raw-vector scaffold; the index becomes read-only. Idempotent;
  /// calling on an already-frozen index is a no-op.
  void Freeze();

  /// Forwarders to storage(=IndexPQ) for the standalone codec interface.
  size_t SaCodeSize() const override;
  void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;
  void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;

  /// Not supported on IndexHNSWPQ. Throws with a clear message — IndexPQ
  /// does not implement these and silent forwarding would surprise.
  void Search1(const float* x, ResultHandler& handler,
               SearchParameters* params = nullptr) const override;
  void RangeSearch(idx_t n, const float* x, float radius,
                    RangeSearchResult* result,
                    const SearchParameters* params = nullptr) const override;
};

}  // namespace hypervec
