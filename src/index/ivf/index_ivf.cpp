/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/index_ivf.h>
#include <index/ivf/inverted_list_scanner.h>
#include <utils/algo/kmeans/kmeans.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>
#include <utils/structures/heap.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {

// ---------------------------------------------------------------------------
// IndexIVF
// ---------------------------------------------------------------------------

IndexIVF::IndexIVF(idx_t d, idx_t nlist, size_t code_size, MetricType metric)
    : Index(d, metric),
      nlist(nlist),
      nprobe(1),
      invlists(new ArrayInvertedLists(nlist, code_size)),
      own_invlists(true) {
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

void IndexIVF::Add(idx_t n, const float* x) { AddWithIds(n, x, nullptr); }

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
                      idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0, "IndexIVF::Search: n must be non-negative");
  HYPERVEC_THROW_IF_NOT(k > 0);
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || (x != nullptr && distances != nullptr && labels != nullptr),
      "IndexIVF::Search: input and output pointers must not be null");
  if (n == 0) {
    return;
  }

  const IVFSearchParameters* ivf_params =
      dynamic_cast<const IVFSearchParameters*>(params);
  const IDSelector* sel = params ? params->sel : nullptr;
  idx_t nprobe_actual = ivf_params ? ivf_params->nprobe : nprobe;
  HYPERVEC_THROW_IF_NOT_MSG(nprobe_actual > 0,
                            "IndexIVF::Search: nprobe must be positive");
  nprobe_actual = std::min(nprobe_actual, nlist);

  const size_t assignment_count = mul_no_overflow(
      static_cast<size_t>(n), static_cast<size_t>(nprobe_actual),
      "IndexIVF::Search assignment count");
  std::vector<float> centroid_dis(assignment_count);
  std::vector<idx_t> centroid_ids(assignment_count);
  FindNearestCentroids(n, x, nprobe_actual, centroid_dis.data(),
                       centroid_ids.data());

  SearchPreassigned(n, x, k, centroid_ids.data(), centroid_dis.data(),
                    distances, labels, nprobe_actual, sel);
}

void IndexIVF::SearchPreassigned(idx_t n, const float* x, idx_t k,
                                 const idx_t* list_ids,
                                 const float* centroid_dis, float* distances,
                                 idx_t* labels, idx_t nprobe_actual,
                                 const IDSelector* sel) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      is_trained, "IndexIVF::SearchPreassigned: index is not trained");
  HYPERVEC_THROW_IF_NOT_MSG(
      n >= 0, "IndexIVF::SearchPreassigned: n must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(k > 0,
                            "IndexIVF::SearchPreassigned: k must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      nprobe_actual > 0 && nprobe_actual <= nlist,
      "IndexIVF::SearchPreassigned: nprobe must be in [1, nlist]");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || (x != nullptr && list_ids != nullptr && distances != nullptr &&
                 labels != nullptr),
      "IndexIVF::SearchPreassigned: input and output pointers must not be "
      "null");
  if (n == 0) {
    return;
  }

  const size_t assignment_count = mul_no_overflow(
      static_cast<size_t>(n), static_cast<size_t>(nprobe_actual),
      "IndexIVF::SearchPreassigned assignment count");
  for (size_t i = 0; i < assignment_count; ++i) {
    HYPERVEC_THROW_IF_NOT_MSG(
        list_ids[i] < 0 || list_ids[i] < nlist,
        "IndexIVF::SearchPreassigned: list id is outside [0, nlist)");
  }

  size_t scanner_count = 1;
#ifdef _OPENMP
  scanner_count = static_cast<size_t>(omp_get_max_threads());
#endif
  scanner_count = std::min(scanner_count, static_cast<size_t>(n));
  std::vector<std::unique_ptr<InvertedListScanner>> scanners;
  scanners.reserve(scanner_count);
  for (size_t i = 0; i < scanner_count; ++i) {
    std::unique_ptr<InvertedListScanner> scanner = CreateInvertedListScanner();
    HYPERVEC_THROW_IF_NOT_MSG(
        scanner != nullptr,
        "IndexIVF::SearchPreassigned: scanner factory returned null");
    HYPERVEC_THROW_IF_NOT_MSG(
        scanner->Metric() == metric_type &&
            scanner->CodeSize() == invlists->code_size,
        "IndexIVF::SearchPreassigned: scanner does not match index storage");
    scanners.push_back(std::move(scanner));
  }

  const bool similarity = IsSimilarityMetric(metric_type);
#pragma omp parallel num_threads(scanner_count) if (n > 1)
  {
    size_t scanner_no = 0;
#ifdef _OPENMP
    scanner_no = static_cast<size_t>(omp_get_thread_num());
#endif
    InvertedListScanner* scanner = scanners[scanner_no].get();
#pragma omp for schedule(dynamic, 1)
    for (idx_t query_no = 0; query_no < n; ++query_no) {
      scanner->SetQuery(x + query_no * d);
      float* heap_distances = distances + query_no * k;
      idx_t* heap_ids = labels + query_no * k;
      if (similarity) {
        heap_heapify<CMin<float, idx_t>>(k, heap_distances, heap_ids);
      } else {
        heap_heapify<CMax<float, idx_t>>(k, heap_distances, heap_ids);
      }

      for (idx_t probe = 0; probe < nprobe_actual; ++probe) {
        const size_t probe_offset =
            static_cast<size_t>(query_no) * nprobe_actual + probe;
        const idx_t list_no = list_ids[probe_offset];
        if (list_no < 0) {
          continue;
        }
        const size_t list_size =
            invlists->list_size(static_cast<size_t>(list_no));
        if (list_size == 0) {
          continue;
        }

        const float coarse_distance =
            centroid_dis == nullptr ? 0.0F : centroid_dis[probe_offset];
        scanner->SetList(list_no, coarse_distance);
        InvertedLists::ScopedCodes codes(invlists,
                                         static_cast<size_t>(list_no));
        InvertedLists::ScopedIds ids(invlists, static_cast<size_t>(list_no));
        scanner->ScanKnn(codes.get(), ids.get(), list_size, sel, k,
                         heap_distances, heap_ids);
      }

      if (similarity) {
        heap_reorder<CMin<float, idx_t>>(k, heap_distances, heap_ids);
      } else {
        heap_reorder<CMax<float, idx_t>>(k, heap_distances, heap_ids);
      }
    }
  }
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
  HYPERVEC_THROW_IF_NOT_MSG(n == 0 || x != nullptr,
                            "IndexIVF::RangeSearch: query pointer must not be "
                            "null when n is positive");
  if (n == 0) {
    result->DoAllocation();
    return;
  }

  const IVFSearchParameters* ivf_params =
      dynamic_cast<const IVFSearchParameters*>(params);
  const IDSelector* sel = params ? params->sel : nullptr;
  idx_t nprobe_actual = ivf_params ? ivf_params->nprobe : nprobe;
  HYPERVEC_THROW_IF_NOT_MSG(nprobe_actual > 0,
                            "IndexIVF::RangeSearch: nprobe must be positive");
  nprobe_actual = std::min(nprobe_actual, nlist);

  const size_t assignment_count = mul_no_overflow(
      static_cast<size_t>(n), static_cast<size_t>(nprobe_actual),
      "IndexIVF::RangeSearch assignment count");
  std::vector<float> centroid_dis(assignment_count);
  std::vector<idx_t> centroid_ids(assignment_count);
  FindNearestCentroids(n, x, nprobe_actual, centroid_dis.data(),
                       centroid_ids.data());

  // Collect results per query, then fill the RangeSearchResult in two passes.
  std::vector<std::vector<std::pair<float, idx_t>>> per_query(
      static_cast<size_t>(n));

  std::unique_ptr<InvertedListScanner> scanner = CreateInvertedListScanner();
  HYPERVEC_THROW_IF_NOT_MSG(
      scanner != nullptr,
      "IndexIVF::RangeSearch: scanner factory returned null");
  HYPERVEC_THROW_IF_NOT_MSG(
      scanner->Metric() == metric_type &&
          scanner->CodeSize() == invlists->code_size,
      "IndexIVF::RangeSearch: scanner does not match index storage");

  for (idx_t qi = 0; qi < n; qi++) {
    scanner->SetQuery(x + qi * d);
    for (idx_t pi = 0; pi < nprobe_actual; pi++) {
      const size_t probe_offset = static_cast<size_t>(qi) * nprobe_actual + pi;
      const idx_t list_no = centroid_ids[probe_offset];
      if (list_no < 0) {
        continue;
      }
      const size_t list_sz = invlists->list_size(static_cast<size_t>(list_no));
      if (list_sz == 0) {
        continue;
      }

      InvertedLists::ScopedCodes codes(invlists, static_cast<size_t>(list_no));
      InvertedLists::ScopedIds ids(invlists, static_cast<size_t>(list_no));
      scanner->SetList(list_no, centroid_dis[probe_offset]);
      scanner->ScanRange(codes.get(), ids.get(), list_sz, sel, radius,
                         &per_query[static_cast<size_t>(qi)]);
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

InvertedListScannerPtr IndexIVF::CreateInvertedListScanner() const {
  HYPERVEC_THROW_MSG(
      "IndexIVF: this index does not provide an inverted-list scanner");
}

}  // namespace hypervec
