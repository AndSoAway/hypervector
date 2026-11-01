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
#include <exception>
#include <utility>
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

IndexCapabilities IndexIVF::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.requires_training = true;
  capabilities.supports_add_with_ids = true;
  return capabilities;
}

void IndexIVF::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0,
      "IndexIVF::Train: reset the index before replacing coarse centroids");
  std::vector<float> trained_centroids = TrainCoarseCentroids(n, x);
  centroids = std::move(trained_centroids);
  is_trained = true;
}

void IndexIVF::Add(idx_t n, const float* x) {
  AddWithIds(n, x, nullptr);
}

void IndexIVF::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexIVF::AddWithIds: index is not trained");
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexIVF::AddWithIds: n must be non-negative");
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

  AddEncodedVectors(n, centroid_ids.data(), codes.data(), xids);
}

void IndexIVF::Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels,
                      const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0, "IndexIVF::Search: n must be non-negative");
  HYPERVEC_THROW_IF_NOT(k > 0);
  if (n == 0) {
    return;
  }

  const IVFSearchParameters* ivf_params =
    dynamic_cast<const IVFSearchParameters*>(params);
  const IDSelector* sel = params ? params->sel : nullptr;
  idx_t nprobe_actual =
    ivf_params ? ivf_params->nprobe : nprobe;
  HYPERVEC_THROW_IF_NOT_MSG(nprobe_actual > 0,
                            "IndexIVF::Search: nprobe must be positive");
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
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexIVF::RangeSearch: n must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(result != nullptr,
                            "IndexIVF::RangeSearch: result must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      result->nq == static_cast<size_t>(n),
      "IndexIVF::RangeSearch: result query count does not match n");
  HYPERVEC_THROW_IF_NOT_MSG(
      invlists->code_size == static_cast<size_t>(d) * sizeof(float),
      "IndexIVF::RangeSearch only supports raw float vector codes; compressed "
      "IVF indexes require a codec-aware implementation");

  const IVFSearchParameters* ivf_params =
    dynamic_cast<const IVFSearchParameters*>(params);
  const IDSelector* sel = params ? params->sel : nullptr;
  idx_t nprobe_actual =
    ivf_params ? ivf_params->nprobe : nprobe;
  HYPERVEC_THROW_IF_NOT_MSG(nprobe_actual > 0,
                            "IndexIVF::RangeSearch: nprobe must be positive");
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
  FindNearestCentroidsIn(centroids, nq, xq, k, distances, labels);
}

std::vector<float> IndexIVF::TrainCoarseCentroids(idx_t n,
                                                  const float* x) const {
  std::vector<float> trained_centroids(static_cast<size_t>(nlist) * d);
  KMeansParameters params;
  params.metric = metric_type;
  params.metric_arg = metric_arg;
  params.spherical = metric_type == kMetricInnerProduct;
  RunKMeans(n, x, d, nlist, trained_centroids.data(), params);
  return trained_centroids;
}

void IndexIVF::FindNearestCentroidsIn(
    const std::vector<float>& coarse_centroids, idx_t nq, const float* xq,
    idx_t k, float* distances, idx_t* labels) const {
  HYPERVEC_THROW_IF_NOT_MSG(nq >= 0,
                            "centroid query count must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(k > 0 && k <= nlist,
                            "centroid neighbor count must be in [1, nlist]");
  HYPERVEC_THROW_IF_NOT_MSG(
      coarse_centroids.size() == static_cast<size_t>(nlist) * d,
      "coarse centroid table has an invalid size");
  if (nq == 0) {
    return;
  }
  if (IsSimilarityMetric(metric_type)) {
    float_minheap_array_t res = {static_cast<size_t>(nq),
                                 static_cast<size_t>(k), labels, distances};
    knn_inner_product(xq, coarse_centroids.data(), static_cast<size_t>(d),
                      static_cast<size_t>(nq), static_cast<size_t>(nlist),
                      &res);
  } else {
    float_maxheap_array_t res = {static_cast<size_t>(nq),
                                 static_cast<size_t>(k), labels, distances};
    knn_L2sqr(xq, coarse_centroids.data(), static_cast<size_t>(d),
              static_cast<size_t>(nq), static_cast<size_t>(nlist), &res);
  }
}

void IndexIVF::AddEncodedVectors(idx_t n, const idx_t* list_ids,
                                 const uint8_t* codes, const idx_t* xids) {
  const size_t code_size = invlists->code_size;
  std::vector<std::vector<idx_t>> ids_by_list(static_cast<size_t>(nlist));
  std::vector<std::vector<uint8_t>> codes_by_list(static_cast<size_t>(nlist));
  std::vector<size_t> old_sizes(static_cast<size_t>(nlist));
  std::vector<idx_t> touched_lists;

  for (idx_t i = 0; i < n; ++i) {
    const idx_t list_no = list_ids[static_cast<size_t>(i)];
    HYPERVEC_THROW_IF_NOT_MSG(list_no >= 0 && list_no < nlist,
                              "coarse assignment is outside [0, nlist)");
    auto& ids = ids_by_list[static_cast<size_t>(list_no)];
    auto& list_codes = codes_by_list[static_cast<size_t>(list_no)];
    if (ids.empty()) {
      old_sizes[static_cast<size_t>(list_no)] =
          invlists->list_size(static_cast<size_t>(list_no));
      touched_lists.push_back(list_no);
    }
    ids.push_back(xids != nullptr ? xids[i] : n_total + i);
    const uint8_t* code = codes + static_cast<size_t>(i) * code_size;
    list_codes.insert(list_codes.end(), code, code + code_size);
  }

  try {
    for (const idx_t list_no : touched_lists) {
      const auto& ids = ids_by_list[static_cast<size_t>(list_no)];
      const auto& list_codes = codes_by_list[static_cast<size_t>(list_no)];
      invlists->add_entries(static_cast<size_t>(list_no), ids.size(),
                            ids.data(), list_codes.data());
    }
  } catch (...) {
    const std::exception_ptr insertion_error = std::current_exception();
    for (const idx_t list_no : touched_lists) {
      try {
        invlists->resize(static_cast<size_t>(list_no),
                         old_sizes[static_cast<size_t>(list_no)]);
      } catch (...) {
        // Preserve the insertion error if a custom backend cannot roll back.
      }
    }
    std::rethrow_exception(insertion_error);
  }

  n_total += n;
}

}  // namespace hypervec
