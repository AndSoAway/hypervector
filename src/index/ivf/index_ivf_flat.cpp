/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/index_ivf_flat.h>
#include <index/ivf/inverted_list_scanner.h>
#include <invlists/inverted_lists.h>
#include <quantization/quantizer.h>
#include <utils/distances/extra_distances.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

template <typename VectorDistanceType>
class FlatInvertedListScanner final : public InvertedListScanner {
 public:
  FlatInvertedListScanner(MetricType metric, size_t code_size,
                          VectorDistanceType distance)
      : InvertedListScanner(metric, code_size),
        distance_(std::move(distance)),
        decoded_(code_size / sizeof(float)) {
    HYPERVEC_THROW_IF_NOT_MSG(
        code_size % sizeof(float) == 0,
        "FlatInvertedListScanner: code size must contain complete floats");
  }

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query != nullptr, "FlatInvertedListScanner: query must not be null");
    query_ = query;
  }

  void SetList(idx_t /*list_no*/, float /*coarse_distance*/) override {}

  float DistanceToCode(const uint8_t* code) const override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ != nullptr,
        "FlatInvertedListScanner: SetQuery must be called before scanning");
    if (reinterpret_cast<uintptr_t>(code) % alignof(float) == 0) {
      return distance_(query_, reinterpret_cast<const float*>(code));
    }
    std::memcpy(decoded_.data(), code, CodeSize());
    return distance_(query_, decoded_.data());
  }

 private:
  VectorDistanceType distance_;
  mutable std::vector<float> decoded_;
  const float* query_ = nullptr;
};

}  // namespace

IndexIVFFlat::IndexIVFFlat(idx_t d, idx_t nlist, MetricType metric)
    : IndexIVF(d, nlist, static_cast<size_t>(d) * sizeof(float), metric) {
  HYPERVEC_THROW_IF_NOT_FMT(
      metric == kMetricL2 || metric == kMetricInnerProduct,
      "IndexIVFFlat: supports kMetricL2 and kMetricInnerProduct only, got "
      "metric=%d",
      static_cast<int>(metric));
}

IndexCapabilities IndexIVFFlat::GetCapabilities() const {
  IndexCapabilities capabilities = IndexIVF::GetCapabilities();
  capabilities.supports_range_search = true;
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexIVFFlat::EncodeVectors(idx_t n, const float* x,
                                 uint8_t* codes) const {
  const FlatQuantizer quantizer(d, metric_type, metric_arg);
  quantizer.Encode(n, x, codes);
}

InvertedListScannerPtr IndexIVFFlat::CreateInvertedListScanner() const {
  return with_VectorDistance(
      static_cast<size_t>(d), metric_type, metric_arg,
      [this](auto distance) -> InvertedListScannerPtr {
        using DistanceType = decltype(distance);
        return std::make_unique<FlatInvertedListScanner<DistanceType>>(
            metric_type, invlists->code_size, std::move(distance));
      });
}

void IndexIVFFlat::Reconstruct(idx_t key, float* recons) const {
  for (size_t list_no = 0; list_no < static_cast<size_t>(nlist); list_no++) {
    size_t list_sz = invlists->list_size(list_no);
    if (list_sz == 0) {
      continue;
    }
    InvertedLists::ScopedIds ids(invlists, list_no);
    const idx_t* id_ptr = ids.get();
    for (size_t j = 0; j < list_sz; j++) {
      if (id_ptr[j] == key) {
        InvertedLists::ScopedCodes codes(invlists, list_no);
        const FlatQuantizer quantizer(d, metric_type, metric_arg);
        quantizer.Decode(1, codes.get() + j * quantizer.CodeSize(), recons);
        return;
      }
    }
  }
  HYPERVEC_THROW_FMT("IndexIVFFlat::Reconstruct: key %" PRId64 " not found",
                     key);
}

}  // namespace hypervec
