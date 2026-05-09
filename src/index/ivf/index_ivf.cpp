/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/index_ivf.h>

#include <utils/algo/kmeans/kmeans.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <vector>

namespace hypervec {

// ---------------------------------------------------------------------------
// IndexIVF
// ---------------------------------------------------------------------------

IndexIVF::IndexIVF(idx_t d, idx_t nlist, size_t code_size, MetricType metric)
  : Index(d, metric)
  , nlist(nlist)
  , nprobe(1)
  , invlists(new ArrayInvertedLists(nlist, code_size))
  , own_invlists(true) {
  is_trained = false;
  centroids.resize(static_cast<size_t>(nlist) * d);
}

IndexIVF::~IndexIVF() {
  if (own_invlists) {
    delete invlists;
  }
}

void IndexIVF::Train(idx_t n, const float* x) {
  // Default KMeansParameters reproduces historical IVF behaviour
  // (niter=25, seed=HYPERVEC_KMEANS_DEFAULT_SEED, nredo=1).
  RunKMeans(n, x, d, nlist, centroids.data(), KMeansParameters{});
  is_trained = true;
}

void IndexIVF::Add(idx_t n, const float* x) {
  AddWithIds(n, x, nullptr);
}

void IndexIVF::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }

  // Find nearest centroid for each vector
  std::vector<float> centroid_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, centroid_dis.data(), centroid_ids.data());

  // Encode all vectors into the list storage format
  const size_t code_sz = invlists->code_size;
  std::vector<uint8_t> codes(static_cast<size_t>(n) * code_sz);
  EncodeVectors(n, x, codes.data());

  // Insert into inverted lists
  for (idx_t i = 0; i < n; i++) {
    const idx_t id = (xids != nullptr) ? xids[i] : n_total + i;
    const idx_t list_no = centroid_ids[static_cast<size_t>(i)];
    invlists->add_entry(static_cast<size_t>(list_no), id,
                        codes.data() + static_cast<size_t>(i) * code_sz);
  }

  n_total += n;
}

void IndexIVF::Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels,
                      const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT(k > 0);

  const IVFSearchParameters* ivf_params =
    dynamic_cast<const IVFSearchParameters*>(params);
  const IDSelector* sel = params ? params->sel : nullptr;
  idx_t nprobe_actual =
    ivf_params ? ivf_params->nprobe : nprobe;
  nprobe_actual = std::min(nprobe_actual, nlist);

  std::vector<float> centroid_dis(static_cast<size_t>(n) * nprobe_actual);
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n) * nprobe_actual);
  FindNearestCentroids(n, x, nprobe_actual, centroid_dis.data(),
                       centroid_ids.data());

  SearchPreassigned(n, x, k, centroid_ids.data(), centroid_dis.data(),
                    distances, labels, nprobe_actual, sel);
}

void IndexIVF::RangeSearch(idx_t n, const float* x, float radius,
                           RangeSearchResult* result,
                           const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(is_trained);

  const IVFSearchParameters* ivf_params =
    dynamic_cast<const IVFSearchParameters*>(params);
  const IDSelector* sel = params ? params->sel : nullptr;
  idx_t nprobe_actual =
    ivf_params ? ivf_params->nprobe : nprobe;
  nprobe_actual = std::min(nprobe_actual, nlist);

  std::vector<float> centroid_dis(static_cast<size_t>(n) * nprobe_actual);
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n) * nprobe_actual);
  FindNearestCentroids(n, x, nprobe_actual, centroid_dis.data(),
                       centroid_ids.data());

  const bool sim = IsSimilarityMetric(metric_type);

  // Collect results per query, then fill the RangeSearchResult in two passes.
  std::vector<std::vector<std::pair<float, idx_t>>> per_query(
    static_cast<size_t>(n));

  for (idx_t qi = 0; qi < n; qi++) {
    const float* xq = x + qi * d;
    for (idx_t pi = 0; pi < nprobe_actual; pi++) {
      const idx_t list_no = centroid_ids[qi * nprobe_actual + pi];
      if (list_no < 0) {
        continue;
      }
      const size_t list_sz = invlists->list_size(static_cast<size_t>(list_no));
      if (list_sz == 0) {
        continue;
      }

      InvertedLists::ScopedCodes codes(invlists, static_cast<size_t>(list_no));
      InvertedLists::ScopedIds ids(invlists, static_cast<size_t>(list_no));
      const float* vecs = reinterpret_cast<const float*>(codes.get());
      const idx_t* id_ptr = ids.get();

      for (size_t j = 0; j < list_sz; j++) {
        if (sel && !sel->IsMember(id_ptr[j])) {
          continue;
        }
        float dist;
        if (sim) {
          dist = fvec_inner_product(xq, vecs + j * static_cast<size_t>(d),
                                    static_cast<size_t>(d));
          if (dist >= radius) {
            per_query[static_cast<size_t>(qi)].push_back({dist, id_ptr[j]});
          }
        } else {
          dist = fvec_L2sqr(xq, vecs + j * static_cast<size_t>(d),
                            static_cast<size_t>(d));
          if (dist <= radius) {
            per_query[static_cast<size_t>(qi)].push_back({dist, id_ptr[j]});
          }
        }
      }
    }
  }

  // Pass 1: set per-query counts in lims[0..nq-1]
  for (idx_t qi = 0; qi < n; qi++) {
    result->lims[qi] = per_query[static_cast<size_t>(qi)].size();
  }
  // DoAllocation converts lims to cumulative offsets and allocates arrays
  result->DoAllocation();

  // Pass 2: copy results into allocated arrays
  for (idx_t qi = 0; qi < n; qi++) {
    const size_t off = result->lims[static_cast<size_t>(qi)];
    const auto& qr = per_query[static_cast<size_t>(qi)];
    for (size_t j = 0; j < qr.size(); j++) {
      result->distances[off + j] = qr[j].first;
      result->labels[off + j] = qr[j].second;
    }
  }
}

void IndexIVF::Reset() {
  invlists->Reset();
  n_total = 0;
}

void IndexIVF::FindNearestCentroids(idx_t nq, const float* xq, idx_t k,
                                    float* distances, idx_t* labels) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (IsSimilarityMetric(metric_type)) {
    float_minheap_array_t res = {static_cast<size_t>(nq),
                                 static_cast<size_t>(k), labels, distances};
    knn_inner_product(xq, centroids.data(), static_cast<size_t>(d),
                      static_cast<size_t>(nq), static_cast<size_t>(nlist),
                      &res);
  } else {
    float_maxheap_array_t res = {static_cast<size_t>(nq),
                                 static_cast<size_t>(k), labels, distances};
    knn_L2sqr(xq, centroids.data(), static_cast<size_t>(d),
              static_cast<size_t>(nq), static_cast<size_t>(nlist), &res);
  }
}

}  // namespace hypervec
