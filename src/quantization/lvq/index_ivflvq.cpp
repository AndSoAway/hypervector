/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/inverted_list_scanner.h>
#include <invlists/inverted_lists.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/lvq_quantizer_adapter.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

class LVQInvertedListScanner final : public InvertedListScanner {
 public:
  LVQInvertedListScanner(const LocalVectorQuantizer& lvq,
                         const std::vector<float>& coarse_centroids,
                         idx_t dimension, idx_t list_count, bool by_residual)
      : InvertedListScanner(kMetricL2, lvq.code_size),
        lvq_(lvq),
        coarse_centroids_(coarse_centroids),
        dimension_(dimension),
        list_count_(list_count),
        by_residual_(by_residual) {
    HYPERVEC_THROW_IF_NOT_MSG(lvq.is_trained,
                              "LVQ scanner requires a trained quantizer");
    HYPERVEC_THROW_IF_NOT_MSG(
        dimension > 0 && lvq.d == dimension,
        "LVQ scanner dimension does not match the quantizer");
    HYPERVEC_THROW_IF_NOT_MSG(list_count > 0 && lvq.nlocal > 0 && lvq.ksub > 0,
                              "LVQ scanner model sizes must be positive");
    const size_t expected_centroid_count = mul_no_overflow(
        static_cast<size_t>(list_count), static_cast<size_t>(dimension),
        "LVQ scanner coarse centroid count");
    HYPERVEC_THROW_IF_NOT_MSG(
        coarse_centroids.size() == expected_centroid_count,
        "LVQ scanner coarse centroid table has an invalid size");
    residual_query_.resize(static_cast<size_t>(dimension));
    distance_table_.resize(mul_no_overflow(static_cast<size_t>(lvq.nlocal),
                                           static_cast<size_t>(lvq.ksub),
                                           "LVQ scanner distance table size"));
  }

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                              "LVQ scanner query must not be null");
    query_ = query;
    table_ready_ = false;
    if (!by_residual_) {
      lvq_.ComputeDistanceTable(query_, distance_table_.data());
      table_ready_ = true;
    }
  }

  void SetList(idx_t list_no, float /*coarse_distance*/) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ != nullptr,
        "LVQ scanner SetQuery must be called before SetList");
    if (!by_residual_) {
      return;
    }

    HYPERVEC_THROW_IF_NOT_MSG(
        list_no >= 0 && list_no < list_count_,
        "LVQ scanner list id is outside the coarse centroid table");
    const size_t centroid_offset =
        static_cast<size_t>(list_no) * static_cast<size_t>(dimension_);
    const float* centroid = coarse_centroids_.data() + centroid_offset;
    for (idx_t i = 0; i < dimension_; ++i) {
      residual_query_[static_cast<size_t>(i)] = query_[i] - centroid[i];
    }
    lvq_.ComputeDistanceTable(residual_query_.data(), distance_table_.data());
    table_ready_ = true;
  }

  float DistanceToCode(const uint8_t* code) const override {
    HYPERVEC_THROW_IF_NOT_MSG(
        table_ready_,
        "LVQ scanner SetQuery and SetList must be called before scanning");
    HYPERVEC_THROW_IF_NOT_MSG(code != nullptr,
                              "LVQ scanner code must not be null");
    return lvq_.ApplyDistanceTable(distance_table_.data(), code);
  }

 private:
  const LocalVectorQuantizer& lvq_;
  const std::vector<float>& coarse_centroids_;
  idx_t dimension_;
  idx_t list_count_;
  bool by_residual_;
  const float* query_ = nullptr;
  std::vector<float> residual_query_;
  std::vector<float> distance_table_;
  bool table_ready_ = false;
};

}  // namespace

IndexIVFLVQ::IndexIVFLVQ() : IndexIVF(0, 0, 0, kMetricL2) {}

IndexIVFLVQ::IndexIVFLVQ(idx_t d, idx_t nlist, idx_t nlocal, int nbits,
                         MetricType metric)
    : IndexIVF(d, nlist, 0, metric), lvq(d, nlocal, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(
      metric == kMetricL2,
      "IndexIVFLVQ: supports kMetricL2 only, got metric=%d",
      static_cast<int>(metric));
  delete invlists;
  invlists = new ArrayInvertedLists(static_cast<size_t>(nlist), lvq.code_size);
  own_invlists = true;
}

IndexCapabilities IndexIVFLVQ::GetCapabilities() const {
  IndexCapabilities capabilities = IndexIVF::GetCapabilities();
  capabilities.supports_range_search = true;
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexIVFLVQ::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0,
      "IndexIVFLVQ::Train: reset the index before replacing trained state");

  std::vector<float> trained_centroids = TrainCoarseCentroids(n, x);
  LocalVectorQuantizer trained_lvq(lvq.d, lvq.nlocal, lvq.nbits);
  LocalVectorQuantizerAdapter trained_quantizer(trained_lvq);
  if (by_residual) {
    std::vector<float> coarse_dis(static_cast<size_t>(n));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
    FindNearestCentroidsIn(trained_centroids, n, x, 1, coarse_dis.data(),
                           centroid_ids.data());

    const size_t residual_count =
        mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(d),
                        "IndexIVFLVQ::Train residual count");
    std::vector<float> residuals(residual_count);
    for (idx_t i = 0; i < n; i++) {
      const float* c = trained_centroids.data() + centroid_ids[i] * d;
      const float* xi = x + i * d;
      float* ri = residuals.data() + i * d;
      for (idx_t j = 0; j < d; j++) {
        ri[j] = xi[j] - c[j];
      }
    }
    trained_quantizer.Train(n, residuals.data());
  } else {
    trained_quantizer.Train(n, x);
  }

  centroids = std::move(trained_centroids);
  lvq = std::move(trained_lvq);
  is_trained = lvq.is_trained;
}

void IndexIVFLVQ::EncodeVectors(idx_t n, const float* x, uint8_t* codes) const {
  const LocalVectorQuantizerAdapter quantizer(lvq);
  if (!by_residual) {
    quantizer.Encode(n, x, codes);
    return;
  }

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  const size_t residual_count =
      mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(d),
                      "IndexIVFLVQ::EncodeVectors residual count");
  std::vector<float> residuals(residual_count);
  for (idx_t i = 0; i < n; i++) {
    const float* c = centroids.data() + centroid_ids[i] * d;
    const float* xi = x + i * d;
    float* ri = residuals.data() + i * d;
    for (idx_t j = 0; j < d; j++) {
      ri[j] = xi[j] - c[j];
    }
  }
  quantizer.Encode(n, residuals.data(), codes);
}

void IndexIVFLVQ::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexIVFLVQ::AddWithIds: index is not trained");
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexIVFLVQ::AddWithIds: n must be non-negative");
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      x != nullptr,
      "IndexIVFLVQ::AddWithIds: x must not be null when n is positive");

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  const LocalVectorQuantizerAdapter quantizer(lvq);
  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(n), quantizer.CodeSize(),
                      "IndexIVFLVQ::AddWithIds code bytes");
  std::vector<uint8_t> codes(code_bytes);
  if (by_residual) {
    std::vector<float> residual(static_cast<size_t>(d));
    for (idx_t i = 0; i < n; i++) {
      const float* c = centroids.data() + centroid_ids[i] * d;
      const float* xi = x + i * d;
      for (idx_t j = 0; j < d; j++) {
        residual[static_cast<size_t>(j)] = xi[j] - c[j];
      }
      quantizer.Encode(
          1, residual.data(),
          codes.data() + static_cast<size_t>(i) * quantizer.CodeSize());
    }
  } else {
    quantizer.Encode(n, x, codes.data());
  }

  AddEncodedVectors(n, centroid_ids.data(), codes.data(), xids);
}

void IndexIVFLVQ::Reconstruct(idx_t key, float* recons) const {
  const LocalVectorQuantizerAdapter quantizer(lvq);
  for (size_t list_no = 0; list_no < static_cast<size_t>(nlist); list_no++) {
    const size_t sz = invlists->list_size(list_no);
    if (sz == 0) {
      continue;
    }
    InvertedLists::ScopedIds ids(invlists, list_no);
    const idx_t* id_ptr = ids.get();
    for (size_t j = 0; j < sz; j++) {
      if (id_ptr[j] == key) {
        InvertedLists::ScopedCodes codes(invlists, list_no);
        quantizer.Decode(1, codes.get() + j * quantizer.CodeSize(), recons);
        if (by_residual) {
          const float* c = centroids.data() + static_cast<idx_t>(list_no) * d;
          for (idx_t l = 0; l < d; l++) {
            recons[l] += c[l];
          }
        }
        return;
      }
    }
  }
  HYPERVEC_THROW_FMT("IndexIVFLVQ::Reconstruct: key %" PRId64 " not found",
                     key);
}

InvertedListScannerPtr IndexIVFLVQ::CreateInvertedListScanner() const {
  return std::make_unique<LVQInvertedListScanner>(lvq, centroids, d, nlist,
                                                  by_residual);
}

}  // namespace hypervec
