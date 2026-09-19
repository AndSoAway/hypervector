/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <quantization/pq/pq.h>
#include <utils/structures/maybe_owned_vector.h>

#include <cstdint>

namespace hypervec {

/** Flat (non-IVF) index over Product-Quantized codes.
 *
 *  Storage: a contiguous code array of size n_total * pq.code_size bytes.
 *  Search: per-query ADC table → linear scan of all codes → top-k max-heap.
 *  Memory cost is M * ⌈log2(ksub)⌉ bits per vector instead of d * 32 bits,
 *  so this is the simplest way to fit a database larger than RAM.
 *
 *  T1 scope: kMetricL2 only. Construct with another metric throws.
 *
 *  Persistence 4cc: "IPQ8". The trailing "8" is part of the magic, not a
 *  byte-width tag — the format encodes any nbits ∈ [1, 16]. */
struct IndexPQ : Index {
  /// Embedded product quantizer.
  ProductQuantizer pq;

  /// Encoded dataset, size n_total * pq.code_size.
  MaybeOwnedVector<uint8_t> codes;

  /// Default constructor for deserialization. Produces an empty,
  /// untrained, dimensionless index that read_index_pq populates.
  IndexPQ();

  /** @param d       vector dimension
   *  @param M       number of subquantizers (must divide d)
   *  @param nbits   bits per code (1..HYPERVEC_PQ_MAX_NBITS)
   *  @param metric  distance metric; T1 requires kMetricL2 */
  IndexPQ(idx_t d, idx_t M, int nbits, MetricType metric = kMetricL2);

  /** Train the underlying ProductQuantizer on the given vectors. */
  void Train(idx_t n, const float* x) override;

  /** Encode and append the n vectors to the code array. */
  void Add(idx_t n, const float* x) override;

  /** ADC search via ProductQuantizer::SearchL2. The optional SearchParameters
   *  are not yet honoured — passing a non-null `params->sel` throws. */
  void Search(idx_t n, const float* x, idx_t k, float* distances,
              idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  /// Drop all stored codes; the trained PQ centroids are kept.
  void Reset() override;

  /// Lossy reconstruction by decoding stored code [key].
  void Reconstruct(idx_t key, float* recons) const override;

  /** Random-access ADC distance computer over `codes`. The returned object
   *  borrows `pq` and `codes`; caller owns and must delete. */
  DistanceComputer* GetDistanceComputer() const override;

  /* Standalone codec interface (mirrors ProductQuantizer's API). */
  size_t SaCodeSize() const override;
  void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;
  void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;
};

}  // namespace hypervec
