/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/refine/index_refine.h>

#include <utils/distances/distances.h>
#include <utils/log/assert.h>
#include <utils/structures/heap.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <omp.h>

#include <algorithm>
#include <vector>

namespace hypervec {

namespace {

// Build an owned flat refine store matching base's dimension and metric.
IndexFlat* MakeRefineStore(const Index* base) {
  if (IsSimilarityMetric(base->metric_type)) {
    return new IndexFlatIP(base->d);
  }
  return new IndexFlatL2(base->d);
}

}  // namespace

IndexRefineFlat::IndexRefineFlat() : Index() {}

IndexRefineFlat::IndexRefineFlat(Index* base)
  : Index(base->d, base->metric_type)
  , base(base)
  , refine(MakeRefineStore(base))
  , own_fields(true) {
  metric_arg = base->metric_arg;
  is_trained = base->is_trained;
  n_total = base->n_total;
}

IndexRefineFlat::IndexRefineFlat(Index* base, IndexFlat* refine)
  : Index(base->d, base->metric_type)
  , base(base)
  , refine(refine)
  , own_fields(false) {
  HYPERVEC_THROW_IF_NOT_MSG(base->d == refine->d,
                            "IndexRefineFlat: base and refine dim mismatch");
  metric_arg = base->metric_arg;
  is_trained = base->is_trained;
  n_total = base->n_total;
}

IndexRefineFlat::~IndexRefineFlat() {
  if (own_fields) {
    delete base;
    delete refine;
  }
}

void IndexRefineFlat::Train(idx_t n, const float* x) {
  base->Train(n, x);
  is_trained = base->is_trained;
}

void IndexRefineFlat::Add(idx_t n, const float* x) {
  base->Add(n, x);
  refine->Add(n, x);
  n_total = base->n_total;
}

void IndexRefineFlat::Search(idx_t n, const float* x, idx_t k, float* distances,
                             idx_t* labels,
                             const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(k > 0);

  float kf = k_factor;
  if (auto* rp = dynamic_cast<const RefineSearchParameters*>(params)) {
    if (rp->k_factor > 0) kf = rp->k_factor;
  }
  idx_t k_base = std::max<idx_t>(k, (idx_t)(k * kf));
  if (k_base > n_total) k_base = n_total;

  // 1. Over-fetch candidates from the coarse base index.
  std::vector<float> base_dist((size_t)n * k_base);
  std::vector<idx_t> base_lab((size_t)n * k_base);
  base->Search(n, x, k_base, base_dist.data(), base_lab.data(), params);

  const bool is_sim = IsSimilarityMetric(metric_type);

  // 2. Re-rank each query's candidates with exact distances from the refine
  //    store, then select the true top-k.
#pragma omp parallel
  {
    std::vector<float> exact((size_t)k_base);
    std::vector<idx_t> valid_ids((size_t)k_base);

#pragma omp for
    for (idx_t i = 0; i < n; i++) {
      const idx_t* cand = base_lab.data() + (size_t)i * k_base;

      // Base may return fewer than k_base hits, padded with -1.
      idx_t m = 0;
      for (idx_t j = 0; j < k_base; j++) {
        if (cand[j] >= 0) valid_ids[m++] = cand[j];
      }

      float* out_d = distances + (size_t)i * k;
      idx_t* out_l = labels + (size_t)i * k;

      if (m == 0) {
        for (idx_t j = 0; j < k; j++) { out_d[j] = 0; out_l[j] = -1; }
        continue;
      }

      // Exact distances from query i to its m candidates.
      refine->ComputeDistanceSubset(1, x + (size_t)i * d, m, exact.data(),
                                    valid_ids.data());

      // Select top-k by metric: L2 keeps smallest (max-heap), IP keeps
      // largest (min-heap).
      idx_t kk = std::min<idx_t>(k, m);
      if (is_sim) {
        minheap_heapify(kk, out_d, out_l);
        for (idx_t j = 0; j < m; j++) {
          if (exact[j] > out_d[0]) {
            minheap_replace_top(kk, out_d, out_l, exact[j], valid_ids[j]);
          }
        }
        minheap_reorder(kk, out_d, out_l);
      } else {
        maxheap_heapify(kk, out_d, out_l);
        for (idx_t j = 0; j < m; j++) {
          if (exact[j] < out_d[0]) {
            maxheap_replace_top(kk, out_d, out_l, exact[j], valid_ids[j]);
          }
        }
        maxheap_reorder(kk, out_d, out_l);
      }
      for (idx_t j = kk; j < k; j++) { out_d[j] = 0; out_l[j] = -1; }
    }
  }
}

void IndexRefineFlat::RangeSearch(idx_t n, const float* x, float radius,
                                  RangeSearchResult* result,
                                  const SearchParameters* params) const {
  base->RangeSearch(n, x, radius, result, params);
}

void IndexRefineFlat::Reconstruct(idx_t key, float* recons) const {
  refine->Reconstruct(key, recons);
}

void IndexRefineFlat::Reset() {
  base->Reset();
  refine->Reset();
  n_total = 0;
}

uint32_t IndexRefineFlat::fourcc() const {
  return hypervec::fourcc("IRFf");
}

void IndexRefineFlat::write_body(IOWriter* f) const {
  WriteIndex(base, f);
  WriteIndex(refine, f);
  WRITE1(k_factor);
}

}  // namespace hypervec
