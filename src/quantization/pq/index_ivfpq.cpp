/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/inverted_list_scanner.h>
#include <invlists/inverted_lists.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/pq_quantizer_adapter.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

std::vector<float> BuildPrecomputedTable(
    idx_t d, idx_t nlist, const std::vector<float>& coarse_centroids,
    const ProductQuantizer& pq) {
  const idx_t pq_M = pq.M;
  const idx_t pq_ksub = pq.ksub;
  const size_t table_per_cell = static_cast<size_t>(pq_M) * pq_ksub;
  std::vector<float> table(static_cast<size_t>(nlist) * table_per_cell, 0.0f);

  std::vector<float> residual_norms(table_per_cell);
  for (idx_t m = 0; m < pq_M; ++m) {
    fvec_norms_L2sqr(residual_norms.data() + m * pq_ksub, pq.GetCentroids(m, 0),
                     static_cast<size_t>(pq.dsub),
                     static_cast<size_t>(pq_ksub));
  }

#pragma omp parallel for if (nlist > 1)
  for (idx_t i = 0; i < nlist; ++i) {
    const float* centroid = coarse_centroids.data() + i * d;
    float* cell_table = table.data() + i * table_per_cell;
    for (idx_t m = 0; m < pq_M; ++m) {
      fvec_inner_products_ny(cell_table + m * pq_ksub, centroid + m * pq.dsub,
                             pq.GetCentroids(m, 0),
                             static_cast<size_t>(pq.dsub),
                             static_cast<size_t>(pq_ksub));
    }
    for (size_t entry = 0; entry < table_per_cell; ++entry) {
      cell_table[entry] = residual_norms[entry] + 2.0f * cell_table[entry];
    }
  }
  return table;
}

class PQInvertedListScanner final : public InvertedListScanner {
 public:
  explicit PQInvertedListScanner(const IndexIVFPQ& index)
    : InvertedListScanner(index.metric_type, index.pq.code_size)
    , index_(index)
    , residual_query_(static_cast<size_t>(index.d))
    , distance_table_(static_cast<size_t>(index.pq.M) * index.pq.ksub)
    , query_inner_products_(index.use_precomputed_table != 0
                              ? distance_table_.size()
                              : 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
      index_.pq.is_trained, "PQInvertedListScanner: PQ is not trained");
    if (UsesPrecomputedTable()) {
      HYPERVEC_THROW_IF_NOT_MSG(
        index_.by_residual,
        "PQInvertedListScanner: precomputed tables require residual codes");
      const size_t expected_size =
        static_cast<size_t>(index_.nlist) * distance_table_.size();
      HYPERVEC_THROW_IF_NOT_MSG(
        index_.precomputed_table.size() == expected_size,
        "PQInvertedListScanner: precomputed table has an invalid size");
    }
  }

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(
      query != nullptr, "PQInvertedListScanner: query must not be null");
    query_ = query;
    list_is_set_ = false;

    if (UsesPrecomputedTable()) {
      for (idx_t subquantizer = 0; subquantizer < index_.pq.M;
           ++subquantizer) {
        fvec_inner_products_ny(
          query_inner_products_.data() + subquantizer * index_.pq.ksub,
          query + subquantizer * index_.pq.dsub,
          index_.pq.GetCentroids(subquantizer, 0),
          static_cast<size_t>(index_.pq.dsub),
          static_cast<size_t>(index_.pq.ksub));
      }
    } else if (!index_.by_residual) {
      index_.pq.ComputeDistanceTable(query_, distance_table_.data());
    }
  }

  void SetList(idx_t list_no, float coarse_distance) override {
    HYPERVEC_THROW_IF_NOT_MSG(
      query_ != nullptr,
      "PQInvertedListScanner: SetQuery must be called before SetList");
    HYPERVEC_THROW_IF_NOT_MSG(
      list_no >= 0 && list_no < index_.nlist,
      "PQInvertedListScanner: list id is outside [0, nlist)");

    list_offset_ = 0.0f;
    if (UsesPrecomputedTable()) {
      const float* precomputed =
        index_.precomputed_table.data() +
        static_cast<size_t>(list_no) * distance_table_.size();
      for (size_t i = 0; i < distance_table_.size(); ++i) {
        distance_table_[i] =
          precomputed[i] - 2.0f * query_inner_products_[i];
      }
      list_offset_ = coarse_distance;
    } else if (index_.by_residual) {
      const float* centroid = index_.centroids.data() + list_no * index_.d;
      for (idx_t i = 0; i < index_.d; ++i) {
        residual_query_[static_cast<size_t>(i)] = query_[i] - centroid[i];
      }
      index_.pq.ComputeDistanceTable(residual_query_.data(),
                                     distance_table_.data());
    }
    list_is_set_ = true;
  }

  float DistanceToCode(const uint8_t* code) const override {
    HYPERVEC_THROW_IF_NOT_MSG(
      list_is_set_,
      "PQInvertedListScanner: SetList must be called before scanning");
    HYPERVEC_THROW_IF_NOT_MSG(
      code != nullptr, "PQInvertedListScanner: code must not be null");
    return list_offset_ +
           index_.pq.ApplyDistanceTable(distance_table_.data(), code);
  }

 private:
  bool UsesPrecomputedTable() const noexcept {
    return index_.use_precomputed_table != 0;
  }

  const IndexIVFPQ& index_;
  std::vector<float> residual_query_;
  std::vector<float> distance_table_;
  std::vector<float> query_inner_products_;
  const float* query_ = nullptr;
  float list_offset_ = 0.0f;
  bool list_is_set_ = false;
};

}  // namespace

// ===========================================================================
// Construction
// ===========================================================================

IndexIVFPQ::IndexIVFPQ() : IndexIVF(0, 0, 0, kMetricL2) {}

IndexIVFPQ::IndexIVFPQ(idx_t d, idx_t nlist, idx_t M, int nbits,
                       MetricType metric)
    : IndexIVF(d, nlist,
               /*code_size=*/
               (static_cast<size_t>(M) * static_cast<size_t>(nbits) + 7) / 8,
               metric),
      pq(d, M, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(
      metric == kMetricL2,
      "IndexIVFPQ: T1+T2 supports kMetricL2 only, got metric=%d",
      static_cast<int>(metric));
}

// ===========================================================================
// Training
// ===========================================================================

void IndexIVFPQ::Train(idx_t n, const float* x) {
  is_trained = false;
  pq.is_trained = false;
  precomputed_table.clear();

  std::vector<float> trained_centroids = TrainCoarseCentroids(n, x);
  ProductQuantizer trained_pq(pq.d, pq.M, pq.nbits);

  if (by_residual) {
    std::vector<float> coarse_dis(static_cast<size_t>(n));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
    FindNearestCentroidsIn(trained_centroids, n, x, 1, coarse_dis.data(),
                           centroid_ids.data());

    std::vector<float> residuals(static_cast<size_t>(n) * d);
    for (idx_t i = 0; i < n; i++) {
      const float* c =
          trained_centroids.data() + centroid_ids[static_cast<size_t>(i)] * d;
      const float* xi = x + i * d;
      float* ri = residuals.data() + i * d;
      for (idx_t j = 0; j < d; j++) {
        ri[j] = xi[j] - c[j];
      }
    }

    ProductQuantizerAdapter quantizer(trained_pq);
    quantizer.Train({n, d, residuals.data(), nullptr});
  } else {
    ProductQuantizerAdapter quantizer(trained_pq);
    quantizer.Train({n, d, x, nullptr});
  }

  std::vector<float> trained_precomputed_table;
  if (use_precomputed_table != 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        by_residual, "IndexIVFPQ precomputed tables require residual encoding");
    trained_precomputed_table =
        BuildPrecomputedTable(d, nlist, trained_centroids, trained_pq);
  }

  centroids = std::move(trained_centroids);
  pq = std::move(trained_pq);
  precomputed_table = std::move(trained_precomputed_table);
  is_trained = true;
}

// ===========================================================================
// EncodeVectors / AddWithIds
// ===========================================================================

void IndexIVFPQ::EncodeVectors(idx_t n, const float* x, uint8_t* codes) const {
  const ProductQuantizerAdapter quantizer(pq);
  if (!by_residual) {
    quantizer.Encode(n, x, codes);
    return;
  }

  // by_residual=true with no list ids in scope: recompute assignments. Hot
  // adders should use AddWithIds (which we override to avoid this).
  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  std::vector<float> residuals(static_cast<size_t>(n) * d);
  for (idx_t i = 0; i < n; i++) {
    const float* c =
        centroids.data() + centroid_ids[static_cast<size_t>(i)] * d;
    const float* xi = x + i * d;
    float* ri = residuals.data() + i * d;
    for (idx_t j = 0; j < d; j++) {
      ri[j] = xi[j] - c[j];
    }
  }

  quantizer.Encode(n, residuals.data(), codes);
}

void IndexIVFPQ::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexIVFPQ::AddWithIds: index is not trained");
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexIVFPQ::AddWithIds: n must be non-negative");
  if (n == 0) {
    return;
  }

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  std::vector<uint8_t> codes(static_cast<size_t>(n) * pq.code_size);
  const ProductQuantizerAdapter quantizer(pq);

  if (by_residual) {
    // Encode residuals; reuse a single per-vector buffer to keep allocation
    // out of the hot path. We could batch via pq.ComputeCodes after building
    // the full residual matrix, but the matrix would be n*d floats which can
    // dwarf the codes themselves; per-vector keeps memory bounded.
    std::vector<float> residual(static_cast<size_t>(d));
    for (idx_t i = 0; i < n; i++) {
      const float* c =
          centroids.data() + centroid_ids[static_cast<size_t>(i)] * d;
      const float* xi = x + i * d;
      for (idx_t j = 0; j < d; j++) {
        residual[static_cast<size_t>(j)] = xi[j] - c[j];
      }
      quantizer.Encode(1, residual.data(),
                       codes.data() + static_cast<size_t>(i) * pq.code_size);
    }
  } else {
    quantizer.Encode(n, x, codes.data());
  }

  AddEncodedVectors(n, centroid_ids.data(), codes.data(), xids);
}

std::unique_ptr<InvertedListScanner>
IndexIVFPQ::CreateInvertedListScanner() const {
  return std::make_unique<PQInvertedListScanner>(*this);
}

// ===========================================================================
// Reconstruct
// ===========================================================================

void IndexIVFPQ::Reconstruct(idx_t key, float* recons) const {
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
        pq.Decode(codes.get() + j * pq.code_size, recons);
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
  HYPERVEC_THROW_FMT("IndexIVFPQ::Reconstruct: key %" PRId64 " not found", key);
}

// ===========================================================================
// PrecomputeTable (T2)
// ===========================================================================

void IndexIVFPQ::PrecomputeTable() {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT(pq.is_trained);
  HYPERVEC_THROW_IF_NOT_MSG(
      by_residual,
      "IndexIVFPQ::PrecomputeTable assumes by_residual=true; "
      "the L2 expansion only telescopes when PQ encodes residuals");

  precomputed_table = BuildPrecomputedTable(d, nlist, centroids, pq);
}

}  // namespace hypervec
