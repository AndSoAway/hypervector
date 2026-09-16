/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// HNSW-only index implementation

#include <index/flat/index_flat.h>
#include <index/hnsw/hnsw_build_utils.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/visited_table.h>
#include <utils/common/range_search_result.h>
#include <utils/common/result_handler.h>
#include <utils/log/assert.h>
#include <utils/structures/random.h>
#include <utils/structures/sorting.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>

namespace hypervec {

using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;

/**************************************************************
 * Add / Search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
  if (IsSimilarityMetric(storage->metric_type)) {
    return new NegativeDistanceComputer(storage->GetDistanceComputer());
  } else {
    return storage->GetDistanceComputer();
  }
}

}  // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M, MetricType metric)
    : Index(d, metric), hnsw(M), storage(nullptr) {}

IndexHNSW::IndexHNSW(Index* storage, int M)
    : Index(storage->d, storage->metric_type), hnsw(M), storage(storage) {
  metric_arg = storage->metric_arg;
  is_trained = storage->is_trained;
}

IndexHNSW::~IndexHNSW() {
  if (storage && own_fields) {
    delete storage;
  }
}

IndexCapabilities IndexHNSW::Capabilities() const noexcept {
  if (storage == nullptr) {
    return 0;
  }

  const IndexCapabilities storage_capabilities = storage->Capabilities();
  IndexCapabilities capabilities = IndexCapability::KnnSearch |
                                   IndexCapability::Filter |
                                   IndexCapability::ConcurrentSearch;
  if (HasCapability(storage_capabilities, IndexCapability::NeedsTrain)) {
    capabilities = capabilities | IndexCapability::NeedsTrain;
  }
  if (HasCapability(storage_capabilities, IndexCapability::IncrementalAdd)) {
    capabilities = capabilities | IndexCapability::BatchBuild |
                   IndexCapability::IncrementalAdd;
  }
  if (HasCapability(storage_capabilities, IndexCapability::RangeSearch)) {
    capabilities = capabilities | IndexCapability::RangeSearch;
  }
  if (HasCapability(storage_capabilities, IndexCapability::Serialize)) {
    capabilities = capabilities | IndexCapability::Serialize;
  }
  // Clone is intentionally not forwarded: clone_IndexHNSW currently does not
  // preserve graph construction parameters such as M.
  return capabilities;
}

void IndexHNSW::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(storage != nullptr,
                            "IndexHNSW::Train: storage is null");
  try {
    storage->Train(n, x);
  } catch (...) {
    // The storage index owns its Train failure contract. Mirror its final
    // state so the graph wrapper never disagrees with its storage.
    is_trained = storage->is_trained;
    throw;
  }
  is_trained = storage->is_trained;
}

void IndexHNSW::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(storage != nullptr,
                            "IndexHNSW::Add: storage is null");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexHNSW::Add: call Train before Add");
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0, "IndexHNSW::Add: n must be non-negative");
  if (n == 0) {
    return;
  }

  const idx_t n0 = n_total;
  HYPERVEC_THROW_IF_NOT_MSG(
      storage->n_total == n0,
      "IndexHNSW::Add: storage and graph counts are inconsistent");

  // Add vectors to storage
  storage->Add(n, x);
  HYPERVEC_THROW_IF_NOT_MSG(
      storage->n_total == n0 + n,
      "IndexHNSW::Add: storage did not add the requested number of vectors");
  n_total = storage->n_total;

  // Build HNSW graph structure
  // Initialize HNSW parameters if first Add
  if (hnsw.ef_construction == 0) {
    hnsw.ef_construction = 40;
  }

  // Prepare level assignments only for this batch. PrepareLevelTab appends n
  // entries to the graph metadata; passing n_total corrupts repeated Add.
  hnsw.PrepareLevelTab(static_cast<size_t>(n), false);

  // Create distance computer for building.
  // Must go through storage_distance_computer() so similarity metrics (IP,
  // Jaccard) are negated — HNSW graph traversal assumes "smaller is better".
  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

  // For single-threaded building, Add vectors one by one
  OmpLockArray lock_array(static_cast<size_t>(n_total) + 1);

  VisitedTable vt(n_total);

  // Add each new vector to the HNSW graph
  for (idx_t i = n0; i < n_total; i++) {
    int pt_level = hnsw.levels[i] - 1;  // levels store level+1 (1-based)
    dis->SetQuery(x + (i - n0) * d);
    hnsw.AddWithLocks(*dis, pt_level, i, lock_array.Get(), vt, false);
  }
}

void IndexHNSW::Reset() {
  hnsw.Reset();
  storage->Reset();
  n_total = 0;
}

void IndexHNSW::Search(idx_t n, const float* x, idx_t k, float* distances,
                       idx_t* labels, const SearchParameters* params) const {
  // Use HNSW graph-based Search
  // Get distance computer from storage.
  // Must go through storage_distance_computer() so similarity metrics are
  // negated — HNSW graph traversal assumes "smaller is better".
  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

  // Do not mutate the process-global OpenMP thread count here. Concurrent
  // Python callers may run Search() after the SWIG layer releases the GIL, so
  // thread count should be controlled by the caller via the OpenMP runtime
  // environment instead of per query.

  // Create result handler
  using RH = HeapBlockResultHandler<HNSW::C>;
  RH bres(n, distances, labels, k);
  typename RH::SingleResultHandler res(bres);

  // Create visited table
  VisitedTable vt(n_total);

  // Search each query
  for (idx_t i = 0; i < n; i++) {
    dis->SetQuery(x + i * d);
    res.begin(i);
    hnsw.Search(*dis, this, res, vt, params);
    res.end();
  }
}

void IndexHNSW::RangeSearch(idx_t n, const float* x, float radius,
                            RangeSearchResult* result,
                            const SearchParameters* params) const {
  storage->RangeSearch(n, x, radius, result, params);
}

void IndexHNSW::Search1(const float* /*x*/, ResultHandler& /*handler*/,
                        SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG(
      "IndexHNSW::Search1 is unsupported until it uses graph search");
}

void IndexHNSW::PermuteEntries(const idx_t* /*perm*/) {
  HYPERVEC_THROW_MSG("IndexHNSW::PermuteEntries is not implemented");
}

void IndexHNSW::Reconstruct(idx_t key, float* recons) const {
  storage->Reconstruct(key, recons);
}

DistanceComputer* IndexHNSW::GetDistanceComputer() const {
  return storage_distance_computer(storage);
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() { is_trained = true; }

static Index* make_hnsw_flat_storage(int d, MetricType metric) {
  if (metric == kMetricL2) {
    return new IndexFlatL2(d);
  }
  if (metric == kMetricInnerProduct) {
    return new IndexFlatIP(d);
  }
  return new IndexFlat(d, metric);
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
    : IndexHNSW(make_hnsw_flat_storage(d, metric), M) {
  own_fields = true;
  is_trained = true;
}

}  // namespace hypervec
