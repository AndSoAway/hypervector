/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/pq/index_pq.h>

#include <quantization/pq/pq_distance_computer.h>
#include <utils/log/assert.h>

#include <cstring>

namespace hypervec {

IndexPQ::IndexPQ() : Index(0, kMetricL2) {
  is_trained = false;
}

IndexPQ::IndexPQ(idx_t d, idx_t M, int nbits, MetricType metric)
  : Index(d, metric), pq(d, M, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(
    metric == kMetricL2,
    "IndexPQ: T1 supports kMetricL2 only, got metric=%d",
    static_cast<int>(metric));
  is_trained = false;
}

void IndexPQ::Train(idx_t n, const float* x) {
  pq.Train(n, x);
  is_trained = true;
}

void IndexPQ::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }
  const size_t old_bytes = static_cast<size_t>(n_total) * pq.code_size;
  const size_t new_bytes = old_bytes + static_cast<size_t>(n) * pq.code_size;
  codes.resize(new_bytes);
  pq.ComputeCodes(n, x, codes.data() + old_bytes);
  n_total += n;
}

void IndexPQ::Search(idx_t n, const float* x, idx_t k, float* distances,
                     idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  // T1 does not honour an IDSelector. Reject explicitly so callers don't
  // get silently-incorrect results.
  HYPERVEC_THROW_IF_NOT_MSG(
    params == nullptr || params->sel == nullptr,
    "IndexPQ::Search does not support IDSelector yet (T1 limitation)");

  pq.SearchL2(n, x, n_total, codes.data(), k, distances, labels);
}

void IndexPQ::Reset() {
  codes.clear();
  n_total = 0;
}

void IndexPQ::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_FMT(
    key >= 0 && key < n_total,
    "IndexPQ::Reconstruct: key %lld out of range [0, %lld)",
    static_cast<long long>(key), static_cast<long long>(n_total));
  pq.Decode(codes.data() + static_cast<size_t>(key) * pq.code_size, recons);
}

DistanceComputer* IndexPQ::GetDistanceComputer() const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  return new PQDistanceComputer(pq, codes.data(), pq.code_size);
}

size_t IndexPQ::SaCodeSize() const {
  return pq.code_size;
}

void IndexPQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  pq.ComputeCodes(n, x, bytes);
}

void IndexPQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  pq.DecodeBatch(n, bytes, x);
}

}  // namespace hypervec
