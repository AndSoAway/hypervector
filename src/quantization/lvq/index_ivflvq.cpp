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
  explicit LVQInvertedListScanner(const IndexIVFLVQ& index)
      : InvertedListScanner(index.metric_type, index.lvq.code_size),
        index_(index),
        residual_query_(static_cast<size_t>(index.d)),
        distance_table_(static_cast<size_t>(index.lvq.nlocal) *
                        index.lvq.ksub) {
    HYPERVEC_THROW_IF_NOT_MSG(index_.lvq.is_trained,
                              "LVQInvertedListScanner: LVQ is not trained");
  }

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                              "LVQInvertedListScanner: query must not be null");
    query_ = query;
    list_is_set_ = false;
    if (!index_.by_residual) {
      index_.lvq.ComputeDistanceTable(query_, distance_table_.data());
    }
  }

  void SetList(idx_t list_no, float /* coarse_distance */) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ != nullptr,
        "LVQInvertedListScanner: SetQuery must be called before SetList");
    HYPERVEC_THROW_IF_NOT_MSG(
        list_no >= 0 && list_no < index_.nlist,
        "LVQInvertedListScanner: list id is outside [0, nlist)");
    if (index_.by_residual) {
      const float* centroid = index_.centroids.data() + list_no * index_.d;
      for (idx_t i = 0; i < index_.d; ++i) {
        residual_query_[static_cast<size_t>(i)] = query_[i] - centroid[i];
      }
      index_.lvq.ComputeDistanceTable(residual_query_.data(),
                                      distance_table_.data());
    }
    list_is_set_ = true;
  }

  float DistanceToCode(const uint8_t* code) const override {
    HYPERVEC_THROW_IF_NOT_MSG(
        list_is_set_,
        "LVQInvertedListScanner: SetList must be called before scanning");
    HYPERVEC_THROW_IF_NOT_MSG(code != nullptr,
                              "LVQInvertedListScanner: code must not be null");
    return index_.lvq.ApplyDistanceTable(distance_table_.data(), code);
  }

 private:
  const IndexIVFLVQ& index_;
  std::vector<float> residual_query_;
  std::vector<float> distance_table_;
  const float* query_ = nullptr;
  bool list_is_set_ = false;
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

void IndexIVFLVQ::Train(idx_t n, const float* x) {
  is_trained = false;
  lvq.is_trained = false;

  std::vector<float> trained_centroids = TrainCoarseCentroids(n, x);
  LocalVectorQuantizer trained_lvq(lvq.d, lvq.nlocal, lvq.nbits);
  if (by_residual) {
    std::vector<float> coarse_dis(static_cast<size_t>(n));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
    FindNearestCentroidsIn(trained_centroids, n, x, 1, coarse_dis.data(),
                           centroid_ids.data());

    std::vector<float> residuals(static_cast<size_t>(n) * d);
    for (idx_t i = 0; i < n; i++) {
      const float* c = trained_centroids.data() + centroid_ids[i] * d;
      const float* xi = x + i * d;
      float* ri = residuals.data() + i * d;
      for (idx_t j = 0; j < d; j++) {
        ri[j] = xi[j] - c[j];
      }
    }
    LocalVectorQuantizerAdapter quantizer(trained_lvq);
    quantizer.Train({n, d, residuals.data(), nullptr});
  } else {
    LocalVectorQuantizerAdapter quantizer(trained_lvq);
    quantizer.Train({n, d, x, nullptr});
  }

  centroids = std::move(trained_centroids);
  lvq = std::move(trained_lvq);
  is_trained = true;
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

  std::vector<float> residuals(static_cast<size_t>(n) * d);
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

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  std::vector<uint8_t> codes(static_cast<size_t>(n) * lvq.code_size);
  const LocalVectorQuantizerAdapter quantizer(lvq);
  if (by_residual) {
    std::vector<float> residual(static_cast<size_t>(d));
    for (idx_t i = 0; i < n; i++) {
      const float* c = centroids.data() + centroid_ids[i] * d;
      const float* xi = x + i * d;
      for (idx_t j = 0; j < d; j++) {
        residual[static_cast<size_t>(j)] = xi[j] - c[j];
      }
      quantizer.Encode(1, residual.data(),
                       codes.data() + static_cast<size_t>(i) * lvq.code_size);
    }
  } else {
    quantizer.Encode(n, x, codes.data());
  }

  AddEncodedVectors(n, centroid_ids.data(), codes.data(), xids);
}

std::unique_ptr<InvertedListScanner> IndexIVFLVQ::CreateInvertedListScanner()
    const {
  return std::make_unique<LVQInvertedListScanner>(*this);
}

void IndexIVFLVQ::Reconstruct(idx_t key, float* recons) const {
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
        lvq.Decode(codes.get() + j * lvq.code_size, recons);
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

}  // namespace hypervec
