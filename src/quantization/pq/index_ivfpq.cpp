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
#include <utils/algo/kmeans/kmeans.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

std::vector<float> BuildPrecomputedTable(
    idx_t d, idx_t nlist, const std::vector<float>& coarse_centroids,
    const ProductQuantizer& pq) {
  const idx_t pq_m = pq.M;
  const idx_t pq_ksub = pq.ksub;
  const size_t table_per_cell = static_cast<size_t>(pq_m) * pq_ksub;
  std::vector<float> table(static_cast<size_t>(nlist) * table_per_cell, 0.0f);

  std::vector<float> residual_norms(table_per_cell);
  for (idx_t m = 0; m < pq_m; ++m) {
    fvec_norms_L2sqr(residual_norms.data() + m * pq_ksub, pq.GetCentroids(m, 0),
                     static_cast<size_t>(pq.dsub),
                     static_cast<size_t>(pq_ksub));
  }

#pragma omp parallel for if (nlist > 1)
  for (idx_t i = 0; i < nlist; ++i) {
    const float* centroid = coarse_centroids.data() + i * d;
    float* cell_table = table.data() + i * table_per_cell;
    for (idx_t m = 0; m < pq_m; ++m) {
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
  PQInvertedListScanner(const ProductQuantizer& pq,
                        const std::vector<float>& coarse_centroids,
                        const std::vector<float>& precomputed_table,
                        idx_t dimension, idx_t list_count, bool by_residual,
                        bool use_precomputed_table)
      : InvertedListScanner(kMetricL2, pq.code_size),
        pq_(pq),
        coarse_centroids_(coarse_centroids),
        precomputed_table_(precomputed_table),
        dimension_(dimension),
        list_count_(list_count),
        by_residual_(by_residual),
        use_precomputed_table_(use_precomputed_table) {
    HYPERVEC_THROW_IF_NOT_MSG(pq.is_trained,
                              "PQ scanner requires a trained quantizer");
    HYPERVEC_THROW_IF_NOT_MSG(
        dimension > 0 && pq.d == dimension,
        "PQ scanner dimension does not match the quantizer");
    HYPERVEC_THROW_IF_NOT_MSG(list_count > 0 && pq.M > 0 && pq.ksub > 0,
                              "PQ scanner model sizes must be positive");
    HYPERVEC_THROW_IF_NOT_MSG(
        !use_precomputed_table_ || by_residual_,
        "PQ scanner precomputed tables require residual encoding");

    const size_t expected_centroid_count = mul_no_overflow(
        static_cast<size_t>(list_count), static_cast<size_t>(dimension),
        "PQ scanner coarse centroid count");
    HYPERVEC_THROW_IF_NOT_MSG(
        coarse_centroids.size() == expected_centroid_count,
        "PQ scanner coarse centroid table has an invalid size");

    table_size_ =
        mul_no_overflow(static_cast<size_t>(pq.M), static_cast<size_t>(pq.ksub),
                        "PQ scanner distance table size");
    if (use_precomputed_table_) {
      const size_t expected_precomputed_count =
          mul_no_overflow(static_cast<size_t>(list_count), table_size_,
                          "PQ scanner precomputed table size");
      HYPERVEC_THROW_IF_NOT_MSG(
          precomputed_table.size() == expected_precomputed_count,
          "PQ scanner precomputed table has an invalid size");
      query_inner_products_.resize(table_size_);
    }
    residual_query_.resize(static_cast<size_t>(dimension));
    distance_table_.resize(table_size_);
  }

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                              "PQ scanner query must not be null");
    query_ = query;
    table_ready_ = false;
    list_offset_ = 0.0F;

    if (use_precomputed_table_) {
      for (idx_t m = 0; m < pq_.M; ++m) {
        fvec_inner_products_ny(query_inner_products_.data() + m * pq_.ksub,
                               query + m * pq_.dsub, pq_.GetCentroids(m, 0),
                               static_cast<size_t>(pq_.dsub),
                               static_cast<size_t>(pq_.ksub));
      }
    } else if (!by_residual_) {
      pq_.ComputeDistanceTable(query, distance_table_.data());
      table_ready_ = true;
    }
  }

  void SetList(idx_t list_no, float coarse_distance) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ != nullptr, "PQ scanner SetQuery must be called before SetList");
    HYPERVEC_THROW_IF_NOT_MSG(
        list_no >= 0 && list_no < list_count_,
        "PQ scanner list id is outside the coarse centroid table");

    if (use_precomputed_table_) {
      const float* cell_table = precomputed_table_.data() +
                                static_cast<size_t>(list_no) * table_size_;
      for (size_t i = 0; i < table_size_; ++i) {
        distance_table_[i] = cell_table[i] - 2.0F * query_inner_products_[i];
      }
      list_offset_ = coarse_distance;
      table_ready_ = true;
      return;
    }
    if (!by_residual_) {
      return;
    }

    const size_t centroid_offset =
        static_cast<size_t>(list_no) * static_cast<size_t>(dimension_);
    const float* centroid = coarse_centroids_.data() + centroid_offset;
    for (idx_t i = 0; i < dimension_; ++i) {
      residual_query_[static_cast<size_t>(i)] = query_[i] - centroid[i];
    }
    pq_.ComputeDistanceTable(residual_query_.data(), distance_table_.data());
    list_offset_ = 0.0F;
    table_ready_ = true;
  }

  float DistanceToCode(const uint8_t* code) const override {
    HYPERVEC_THROW_IF_NOT_MSG(
        table_ready_,
        "PQ scanner SetQuery and SetList must be called before scanning");
    HYPERVEC_THROW_IF_NOT_MSG(code != nullptr,
                              "PQ scanner code must not be null");
    return list_offset_ + pq_.ApplyDistanceTable(distance_table_.data(), code);
  }

 private:
  const ProductQuantizer& pq_;
  const std::vector<float>& coarse_centroids_;
  const std::vector<float>& precomputed_table_;
  idx_t dimension_;
  idx_t list_count_;
  bool by_residual_;
  bool use_precomputed_table_;
  size_t table_size_ = 0;
  const float* query_ = nullptr;
  std::vector<float> residual_query_;
  std::vector<float> distance_table_;
  std::vector<float> query_inner_products_;
  float list_offset_ = 0.0F;
  bool table_ready_ = false;
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

IndexCapabilities IndexIVFPQ::GetCapabilities() const {
  IndexCapabilities capabilities = IndexIVF::GetCapabilities();
  capabilities.supports_range_search = true;
  capabilities.supports_reconstruct = true;
  return capabilities;
}

// ===========================================================================
// Training
// ===========================================================================

void IndexIVFPQ::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0,
      "IndexIVFPQ::Train: reset the index before replacing trained state");
  HYPERVEC_THROW_IF_NOT_MSG(
      use_precomputed_table == 0 || by_residual,
      "IndexIVFPQ::Train: precomputed tables require residual encoding");

  std::vector<float> trained_centroids = TrainCoarseCentroids(n, x);
  ProductQuantizer trained_pq(pq.d, pq.M, pq.nbits);
  PQParameters pq_params;
  ProductQuantizerAdapter trained_quantizer(trained_pq, pq_params);

  if (by_residual) {
    // PQ trains on at most ksub * max_points_per_centroid rows. Select them
    // before computing residuals, avoiding an all-row coarse assignment and
    // an n*d temporary that the quantizer would immediately discard.
    idx_t training_count = n;
    if (pq_params.max_points_per_centroid > 0) {
      const size_t limit = mul_no_overflow(
          static_cast<size_t>(trained_pq.ksub),
          static_cast<size_t>(pq_params.max_points_per_centroid),
          "IndexIVFPQ training sample limit");
      HYPERVEC_THROW_IF_NOT_MSG(
          limit <= static_cast<size_t>((std::numeric_limits<idx_t>::max)()),
          "IndexIVFPQ training sample limit exceeds idx_t");
      training_count = std::min(n, static_cast<idx_t>(limit));
    }

    std::vector<float> sampled_vectors;
    const float* training_vectors = x;
    if (training_count < n) {
      const std::vector<idx_t> rows =
          SampleKMeansTrainingRows(n, training_count, pq_params.seed);
      sampled_vectors.resize(mul_no_overflow(
          static_cast<size_t>(training_count), static_cast<size_t>(d),
          "IndexIVFPQ training vectors"));
      for (idx_t i = 0; i < training_count; ++i) {
        std::memcpy(sampled_vectors.data() + i * d,
                    x + rows[static_cast<size_t>(i)] * d,
                    static_cast<size_t>(d) * sizeof(float));
      }
      training_vectors = sampled_vectors.data();
    }

    std::vector<float> coarse_dis(static_cast<size_t>(training_count));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(training_count));
    FindNearestCentroidsIn(trained_centroids, training_count, training_vectors,
                           1, coarse_dis.data(), centroid_ids.data());

    std::vector<float> residuals(mul_no_overflow(
        static_cast<size_t>(training_count), static_cast<size_t>(d),
        "IndexIVFPQ training residuals"));
    for (idx_t i = 0; i < training_count; ++i) {
      const float* c =
          trained_centroids.data() + centroid_ids[static_cast<size_t>(i)] * d;
      const float* xi = training_vectors + i * d;
      float* ri = residuals.data() + i * d;
      for (idx_t j = 0; j < d; j++) {
        ri[j] = xi[j] - c[j];
      }
    }

    pq_params.max_points_per_centroid = 0;  // already sampled above
    ProductQuantizerAdapter residual_quantizer(trained_pq, pq_params);
    residual_quantizer.Train(training_count, residuals.data());
  } else {
    trained_quantizer.Train(n, x);
  }

  std::vector<float> trained_precomputed_table;
  if (use_precomputed_table != 0) {
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
  HYPERVEC_THROW_IF_NOT_MSG(
      x != nullptr,
      "IndexIVFPQ::AddWithIds: x must not be null when n is positive");

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  const ProductQuantizerAdapter quantizer(pq);
  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(n), quantizer.CodeSize(),
                      "IndexIVFPQ::AddWithIds code bytes");
  std::vector<uint8_t> codes(code_bytes);

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
      quantizer.Encode(
          1, residual.data(),
          codes.data() + static_cast<size_t>(i) * quantizer.CodeSize());
    }
  } else {
    quantizer.Encode(n, x, codes.data());
  }

  AddEncodedVectors(n, centroid_ids.data(), codes.data(), xids);
}

// ===========================================================================
// Reconstruct
// ===========================================================================

void IndexIVFPQ::Reconstruct(idx_t key, float* recons) const {
  const ProductQuantizerAdapter quantizer(pq);
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

InvertedListScannerPtr IndexIVFPQ::CreateInvertedListScanner() const {
  return std::make_unique<PQInvertedListScanner>(
      pq, centroids, precomputed_table, d, nlist, by_residual,
      use_precomputed_table != 0);
}

}  // namespace hypervec
