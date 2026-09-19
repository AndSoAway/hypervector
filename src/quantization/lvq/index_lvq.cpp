/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/lvq/index_lvq.h>

#include <quantization/lvq/lvq_distance_computer.h>
#include <utils/log/assert.h>

namespace hypervec {

IndexLVQ::IndexLVQ() : Index(0, kMetricL2) {
  is_trained = false;
}

IndexLVQ::IndexLVQ(idx_t d, idx_t nlocal, int nbits, MetricType metric)
  : Index(d, metric), lvq(d, nlocal, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(
    metric == kMetricL2, "IndexLVQ: supports kMetricL2 only, got metric=%d",
    static_cast<int>(metric));
  is_trained = false;
}

void IndexLVQ::Train(idx_t n, const float* x) {
  lvq.Train(n, x);
  is_trained = true;
}

void IndexLVQ::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }
  const size_t old_bytes = static_cast<size_t>(n_total) * lvq.code_size;
  const size_t new_bytes = old_bytes + static_cast<size_t>(n) * lvq.code_size;
  codes.resize(new_bytes);
  lvq.ComputeCodes(n, x, codes.data() + old_bytes);
  n_total += n;
}

void IndexLVQ::Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_MSG(
    params == nullptr || params->sel == nullptr,
    "IndexLVQ::Search does not support IDSelector yet");
  lvq.SearchL2(n, x, n_total, codes.data(), k, distances, labels);
}

void IndexLVQ::Reset() {
  codes.clear();
  n_total = 0;
}

void IndexLVQ::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_FMT(
    key >= 0 && key < n_total,
    "IndexLVQ::Reconstruct: key %lld out of range [0, %lld)",
    static_cast<long long>(key), static_cast<long long>(n_total));
  lvq.Decode(codes.data() + static_cast<size_t>(key) * lvq.code_size, recons);
}

DistanceComputer* IndexLVQ::GetDistanceComputer() const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  return new LVQDistanceComputer(lvq, codes.data(), lvq.code_size);
}

size_t IndexLVQ::SaCodeSize() const {
  return lvq.code_size;
}

void IndexLVQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  lvq.ComputeCodes(n, x, bytes);
}

void IndexLVQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  lvq.DecodeBatch(n, bytes, x);
}

}  // namespace hypervec
