/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/inverted_list_scanner.h>
#include <invlists/inverted_lists.h>
#include <quantization/rabitq/index_ivf_rabitq.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

float SquaredNorm(const float* vector, idx_t dimension, const char* operation) {
  double norm = 0.0;
  for (idx_t i = 0; i < dimension; ++i) {
    HYPERVEC_THROW_IF_NOT_FMT(std::isfinite(vector[i]),
                              "%s: non-finite value at offset %zu", operation,
                              static_cast<size_t>(i));
    norm += static_cast<double>(vector[i]) * vector[i];
  }
  HYPERVEC_THROW_IF_NOT_FMT(
      std::isfinite(norm) &&
          norm <= static_cast<double>((std::numeric_limits<float>::max)()),
      "%s: norm is not representable", operation);
  return static_cast<float>(norm);
}

void ValidateVectors(idx_t n, const float* vectors, idx_t dimension,
                     const char* operation) {
  HYPERVEC_THROW_IF_NOT_FMT(n >= 0, "%s: n must be non-negative", operation);
  HYPERVEC_THROW_IF_NOT_FMT(
      n == 0 || vectors != nullptr,
      "%s: vector input must not be null when n is positive", operation);
  mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(dimension),
                  "RaBitQ vector element count");
  for (idx_t row = 0; row < n; ++row) {
    SquaredNorm(vectors + row * dimension, dimension, operation);
  }
}

class RaBitQInvertedListScanner final : public InvertedListScanner {
 public:
  RaBitQInvertedListScanner(const RaBitQQuantizer& quantizer,
                            const std::vector<float>& coarse_centroids,
                            idx_t dimension, idx_t list_count, bool by_residual)
      : InvertedListScanner(kMetricL2, quantizer.CodeSize()),
        quantizer_(quantizer),
        coarse_centroids_(coarse_centroids),
        dimension_(dimension),
        list_count_(list_count),
        by_residual_(by_residual),
        residual_query_(static_cast<size_t>(dimension)),
        rotated_query_(quantizer.RotatedDimension()) {
    HYPERVEC_THROW_IF_NOT_MSG(
        dimension > 0 && quantizer.Dimension() == dimension,
        "RaBitQ scanner dimension does not match the quantizer");
    HYPERVEC_THROW_IF_NOT_MSG(list_count > 0,
                              "RaBitQ scanner list count must be positive");
    const size_t expected_centroid_count = mul_no_overflow(
        static_cast<size_t>(list_count), static_cast<size_t>(dimension),
        "RaBitQ scanner coarse centroid count");
    HYPERVEC_THROW_IF_NOT_MSG(
        coarse_centroids.size() == expected_centroid_count,
        "RaBitQ scanner coarse centroid table has an invalid size");
  }

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                              "RaBitQ scanner query must not be null");
    query_ = query;
    query_ready_ = false;
    if (!by_residual_) {
      PrepareQuery(query_);
    }
  }

  void SetList(idx_t list_no, float /*coarse_distance*/) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ != nullptr,
        "RaBitQ scanner SetQuery must be called before SetList");
    HYPERVEC_THROW_IF_NOT_MSG(
        list_no >= 0 && list_no < list_count_,
        "RaBitQ scanner list id is outside the coarse centroid table");
    if (!by_residual_) {
      return;
    }

    const float* centroid =
        coarse_centroids_.data() + static_cast<size_t>(list_no) * dimension_;
    for (idx_t i = 0; i < dimension_; ++i) {
      residual_query_[static_cast<size_t>(i)] = query_[i] - centroid[i];
    }
    PrepareQuery(residual_query_.data());
  }

  float DistanceToCode(const uint8_t* code) const override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ready_,
        "RaBitQ scanner SetQuery and SetList must be called before scanning");
    HYPERVEC_THROW_IF_NOT_MSG(code != nullptr,
                              "RaBitQ scanner code must not be null");
    return quantizer_.EstimateSquaredDistance(rotated_query_.data(),
                                              query_norm_squared_, code);
  }

 private:
  void PrepareQuery(const float* query) {
    query_norm_squared_ =
        SquaredNorm(query, dimension_, "RaBitQ scanner query");
    quantizer_.Transform(query, rotated_query_.data());
    query_ready_ = true;
  }

  const RaBitQQuantizer& quantizer_;
  const std::vector<float>& coarse_centroids_;
  idx_t dimension_;
  idx_t list_count_;
  bool by_residual_;
  const float* query_ = nullptr;
  std::vector<float> residual_query_;
  std::vector<float> rotated_query_;
  float query_norm_squared_ = 0.0F;
  bool query_ready_ = false;
};

}  // namespace

IndexIVFRaBitQ::IndexIVFRaBitQ()
    : IndexIVF(0, 0, 0, kMetricL2), rabitq(nullptr) {}

IndexIVFRaBitQ::IndexIVFRaBitQ(idx_t d, idx_t nlist, uint64_t seed,
                               int rotation_rounds, MetricType metric)
    : IndexIVF(d, nlist, 0, metric),
      rabitq(std::make_unique<RaBitQQuantizer>(d, seed, rotation_rounds)) {
  HYPERVEC_THROW_IF_NOT_FMT(
      metric == kMetricL2,
      "IndexIVFRaBitQ: supports kMetricL2 only, got metric=%d",
      static_cast<int>(metric));
  auto* replacement =
      new ArrayInvertedLists(static_cast<size_t>(nlist), rabitq->CodeSize());
  delete invlists;
  invlists = replacement;
  own_invlists = true;
}

IndexCapabilities IndexIVFRaBitQ::GetCapabilities() const {
  IndexCapabilities capabilities = IndexIVF::GetCapabilities();
  capabilities.supports_range_search = true;
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexIVFRaBitQ::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0,
      "IndexIVFRaBitQ::Train: reset the index before replacing trained state");
  HYPERVEC_THROW_IF_NOT_MSG(
      rabitq != nullptr, "IndexIVFRaBitQ::Train: quantizer must be configured");
  ValidateVectors(n, x, d, "IndexIVFRaBitQ::Train");
  std::vector<float> trained_centroids = TrainCoarseCentroids(n, x);
  centroids = std::move(trained_centroids);
  is_trained = true;
}

void IndexIVFRaBitQ::EncodeVectors(idx_t n, const float* x,
                                   uint8_t* codes) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      rabitq != nullptr,
      "IndexIVFRaBitQ::EncodeVectors: quantizer must be configured");
  ValidateVectors(n, x, d, "IndexIVFRaBitQ::EncodeVectors");
  if (!by_residual) {
    rabitq->Encode(n, x, codes);
    return;
  }

  std::vector<float> coarse_distances(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_distances.data(), centroid_ids.data());
  const size_t residual_count =
      mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(d),
                      "IndexIVFRaBitQ::EncodeVectors residual count");
  std::vector<float> residuals(residual_count);
  for (idx_t row = 0; row < n; ++row) {
    const float* centroid =
        centroids.data() + centroid_ids[static_cast<size_t>(row)] * d;
    const float* vector = x + row * d;
    float* residual = residuals.data() + row * d;
    for (idx_t column = 0; column < d; ++column) {
      residual[column] = vector[column] - centroid[column];
    }
  }
  rabitq->Encode(n, residuals.data(), codes);
}

void IndexIVFRaBitQ::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexIVFRaBitQ::AddWithIds: index is not trained");
  HYPERVEC_THROW_IF_NOT_MSG(
      n >= 0, "IndexIVFRaBitQ::AddWithIds: n must be non-negative");
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      x != nullptr,
      "IndexIVFRaBitQ::AddWithIds: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      rabitq != nullptr,
      "IndexIVFRaBitQ::AddWithIds: quantizer must be configured");
  ValidateVectors(n, x, d, "IndexIVFRaBitQ::AddWithIds");

  std::vector<float> coarse_distances(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_distances.data(), centroid_ids.data());
  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(n), rabitq->CodeSize(),
                      "IndexIVFRaBitQ::AddWithIds code bytes");
  std::vector<uint8_t> codes(code_bytes);

  if (by_residual) {
    std::vector<float> residual(static_cast<size_t>(d));
    for (idx_t row = 0; row < n; ++row) {
      const float* centroid =
          centroids.data() + centroid_ids[static_cast<size_t>(row)] * d;
      const float* vector = x + row * d;
      for (idx_t column = 0; column < d; ++column) {
        residual[static_cast<size_t>(column)] =
            vector[column] - centroid[column];
      }
      rabitq->Encode(
          1, residual.data(),
          codes.data() + static_cast<size_t>(row) * rabitq->CodeSize());
    }
  } else {
    rabitq->Encode(n, x, codes.data());
  }
  AddEncodedVectors(n, centroid_ids.data(), codes.data(), xids);
}

void IndexIVFRaBitQ::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      rabitq != nullptr,
      "IndexIVFRaBitQ::Reconstruct: quantizer must be configured");
  HYPERVEC_THROW_IF_NOT_MSG(
      recons != nullptr,
      "IndexIVFRaBitQ::Reconstruct: output must not be null");
  for (size_t list_no = 0; list_no < static_cast<size_t>(nlist); ++list_no) {
    const size_t list_size = invlists->list_size(list_no);
    if (list_size == 0) {
      continue;
    }
    InvertedLists::ScopedIds ids(invlists, list_no);
    for (size_t offset = 0; offset < list_size; ++offset) {
      if (ids.get()[offset] != key) {
        continue;
      }
      InvertedLists::ScopedCodes codes(invlists, list_no);
      rabitq->Decode(1, codes.get() + offset * rabitq->CodeSize(), recons);
      if (by_residual) {
        const float* centroid = centroids.data() + list_no * d;
        for (idx_t column = 0; column < d; ++column) {
          recons[column] += centroid[column];
        }
      }
      return;
    }
  }
  HYPERVEC_THROW_FMT("IndexIVFRaBitQ::Reconstruct: key %" PRId64 " not found",
                     key);
}

InvertedListScannerPtr IndexIVFRaBitQ::CreateInvertedListScanner() const {
  HYPERVEC_THROW_IF_NOT_MSG(
      rabitq != nullptr,
      "IndexIVFRaBitQ::CreateInvertedListScanner: quantizer must be "
      "configured");
  return std::make_unique<RaBitQInvertedListScanner>(*rabitq, centroids, d,
                                                     nlist, by_residual);
}

}  // namespace hypervec
