/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq_quantizer_adapter.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cstring>
#include <limits>
#include <vector>

namespace hypervec {

IndexPQ::IndexPQ() : Index(0, kMetricL2) { is_trained = false; }

IndexPQ::IndexPQ(idx_t d, idx_t M, int nbits, MetricType metric)
    : Index(d, metric), pq(d, M, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(
      metric == kMetricL2, "IndexPQ: T1 supports kMetricL2 only, got metric=%d",
      static_cast<int>(metric));
  is_trained = false;
}

IndexCapabilities IndexPQ::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.requires_training = true;
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexPQ::Train(idx_t n, const float* x) {
  ProductQuantizerAdapter quantizer(pq);
  quantizer.Train(n, x);
  is_trained = quantizer.IsTrained();
}

void IndexPQ::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexPQ::Add: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr, "IndexPQ::Add: x must not be null");

  HYPERVEC_THROW_IF_NOT_MSG(
      n <= (std::numeric_limits<idx_t>::max)() - n_total,
      "IndexPQ::Add: vector count exceeds the supported range");
  ProductQuantizerAdapter quantizer(pq);
  const size_t old_bytes =
      mul_no_overflow(static_cast<size_t>(n_total), quantizer.CodeSize(),
                      "IndexPQ::Add existing code size");
  HYPERVEC_THROW_IF_NOT_MSG(
      codes.size() == old_bytes,
      "IndexPQ::Add: code storage and vector count are inconsistent");
  const size_t batch_bytes =
      mul_no_overflow(static_cast<size_t>(n), quantizer.CodeSize(),
                      "IndexPQ::Add batch code size");
  std::vector<uint8_t> encoded(batch_bytes);
  quantizer.Encode(n, x, encoded.data());

  const size_t new_bytes =
      add_no_overflow(old_bytes, batch_bytes, "IndexPQ::Add total code size");
  codes.resize(new_bytes);
  std::memcpy(codes.data() + old_bytes, encoded.data(), batch_bytes);
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
      "IndexPQ::Reconstruct: key %" PRId64 " out of range [0, %" PRId64 ")",
      static_cast<int64_t>(key), static_cast<int64_t>(n_total));
  const ProductQuantizerAdapter quantizer(pq);
  quantizer.Decode(1, codes.data() + static_cast<size_t>(key) * pq.code_size,
                   recons);
}

DistanceComputer* IndexPQ::GetDistanceComputer() const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  const ProductQuantizerAdapter quantizer(pq);
  return quantizer
      .CreateDistanceComputer(
          EncodedVectorView(codes.data(), n_total, pq.code_size))
      .release();
}

size_t IndexPQ::SaCodeSize() const { return pq.code_size; }

void IndexPQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  const ProductQuantizerAdapter quantizer(pq);
  quantizer.Encode(n, x, bytes);
}

void IndexPQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  const ProductQuantizerAdapter quantizer(pq);
  quantizer.Decode(n, bytes, x);
}

}  // namespace hypervec
