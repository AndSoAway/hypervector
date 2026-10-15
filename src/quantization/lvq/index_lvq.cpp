/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/lvq_quantizer_adapter.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cstring>
#include <limits>
#include <vector>

namespace hypervec {
namespace {

void ValidateCodeStorage(idx_t count, size_t code_size, size_t actual_size,
                         const char* operation) {
  HYPERVEC_THROW_IF_NOT_FMT(count >= 0, "%s: vector count is negative",
                            operation);
  const size_t expected_size =
      mul_no_overflow(static_cast<size_t>(count), code_size, operation);
  HYPERVEC_THROW_IF_NOT_FMT(actual_size == expected_size,
                            "%s: code storage size is inconsistent", operation);
}

}  // namespace

IndexLVQ::IndexLVQ() : Index(0, kMetricL2) { is_trained = false; }

IndexLVQ::IndexLVQ(idx_t d, int nbits, MetricType metric)
    : Index(d, metric), lvq(d, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(metric == kMetricL2,
                            "IndexLVQ: supports kMetricL2 only, got metric=%d",
                            static_cast<int>(metric));
  is_trained = false;
}

IndexCapabilities IndexLVQ::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.requires_training = true;
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexLVQ::Train(idx_t n, const float* x) {
  LocalVectorQuantizerAdapter quantizer(lvq);
  quantizer.Train(n, x);
  is_trained = quantizer.IsTrained();
}

void IndexLVQ::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexLVQ::Add: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr, "IndexLVQ::Add: x must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(codes.is_owned,
                            "IndexLVQ::Add: mapped storage is read-only");

  HYPERVEC_THROW_IF_NOT_MSG(
      n <= (std::numeric_limits<idx_t>::max)() - n_total,
      "IndexLVQ::Add: vector count exceeds the supported range");
  LocalVectorQuantizerAdapter quantizer(lvq);
  const size_t old_bytes =
      mul_no_overflow(static_cast<size_t>(n_total), quantizer.CodeSize(),
                      "IndexLVQ::Add existing code size");
  HYPERVEC_THROW_IF_NOT_MSG(
      codes.size() == old_bytes,
      "IndexLVQ::Add: code storage and vector count are inconsistent");
  const size_t batch_bytes =
      mul_no_overflow(static_cast<size_t>(n), quantizer.CodeSize(),
                      "IndexLVQ::Add batch code size");
  std::vector<uint8_t> encoded(batch_bytes);
  quantizer.Encode(n, x, encoded.data());

  const size_t new_bytes =
      add_no_overflow(old_bytes, batch_bytes, "IndexLVQ::Add total code size");
  codes.resize(new_bytes);
  std::memcpy(codes.data() + old_bytes, encoded.data(), batch_bytes);
  n_total += n;
}

void IndexLVQ::Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  ValidateCodeStorage(n_total, lvq.code_size, codes.size(), "IndexLVQ::Search");
  HYPERVEC_THROW_IF_NOT_MSG(params == nullptr || params->sel == nullptr,
                            "IndexLVQ::Search does not support IDSelector yet");
  lvq.SearchL2(n, x, n_total, codes.data(), k, distances, labels);
}

void IndexLVQ::Reset() {
  codes.clear();
  n_total = 0;
}

void IndexLVQ::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  ValidateCodeStorage(n_total, lvq.code_size, codes.size(),
                      "IndexLVQ::Reconstruct");
  HYPERVEC_THROW_IF_NOT_FMT(
      key >= 0 && key < n_total,
      "IndexLVQ::Reconstruct: key %" PRId64 " out of range [0, %" PRId64 ")",
      static_cast<int64_t>(key), static_cast<int64_t>(n_total));
  HYPERVEC_THROW_IF_NOT_MSG(recons != nullptr,
                            "IndexLVQ::Reconstruct: output must not be null");
  const LocalVectorQuantizerAdapter quantizer(lvq);
  quantizer.Decode(1, codes.data() + static_cast<size_t>(key) * lvq.code_size,
                   recons);
}

DistanceComputer* IndexLVQ::GetDistanceComputer() const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  ValidateCodeStorage(n_total, lvq.code_size, codes.size(),
                      "IndexLVQ::GetDistanceComputer");
  const LocalVectorQuantizerAdapter quantizer(lvq);
  return quantizer
      .CreateDistanceComputer(
          EncodedVectorView(codes.data(), n_total, lvq.code_size))
      .release();
}

size_t IndexLVQ::SaCodeSize() const { return lvq.code_size; }

void IndexLVQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  const LocalVectorQuantizerAdapter quantizer(lvq);
  quantizer.Encode(n, x, bytes);
}

void IndexLVQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  const LocalVectorQuantizerAdapter quantizer(lvq);
  quantizer.Decode(n, bytes, x);
}

}  // namespace hypervec
