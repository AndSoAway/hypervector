/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// HNSW-only index implementation

#include <core/hypervec_assert.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/visited_table.h>
#include <omp.h>
#include <search/aux_index_structures.h>
#include <search/result_handler.h>
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

#include "core/index.h"

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
}

IndexHNSW::~IndexHNSW() {
  if (storage && own_fields) {
    delete storage;
  }
}

void IndexHNSW::Train(idx_t n, const float* x) {
  storage->Train(n, x);
}

void IndexHNSW::Add(idx_t n, const float* x) {
  // Add vectors to storage
  storage->Add(n, x);
  idx_t n0 = n_total;
  n_total = storage->n_total;

  // Build HNSW graph structure
  // Initialize HNSW parameters if first Add
  if (hnsw.ef_construction == 0) {
    hnsw.ef_construction = 40;
  }

  // Prepare level assignment for all vectors (including existing ones)
  hnsw.PrepareLevelTab(n_total, false);

  // Create distance computer for building
  auto dis = storage->GetDistanceComputer();

  // For single-threaded building, Add vectors one by one
  std::vector<omp_lock_t> locks(n_total + 1);
  for (int i = 0; i <= n_total; i++) {
    omp_init_lock(&locks[i]);
  }

  VisitedTable vt(n_total);

  // Add each new vector to the HNSW graph
  for (idx_t i = n0; i < n_total; i++) {
    int pt_level = hnsw.levels[i] - 1;  // levels store level+1 (1-based)
    dis->SetQuery(x + (i - n0) * d);
    hnsw.AddWithLocks(*dis, pt_level, i, locks, vt, false);
  }

  // Cleanup locks
  for (int i = 0; i <= n_total; i++) {
    omp_destroy_lock(&locks[i]);
  }

  delete dis;
}

void IndexHNSW::Reset() {
  hnsw.Reset();
  storage->Reset();
  n_total = 0;
}

void IndexHNSW::Search(idx_t n, const float* x, idx_t k, float* distances,
                       idx_t* labels, const SearchParameters* params) const {
  // Use HNSW graph-based Search
  // Get distance computer from storage
  auto dis = storage->GetDistanceComputer();

  // Set number of threads to 1 for reproducibility in demo
  omp_set_num_threads(1);

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

  // Cleanup
  delete dis;
}

void IndexHNSW::RangeSearch(idx_t n, const float* x, float radius,
                             RangeSearchResult* result,
                             const SearchParameters* params) const {
  storage->RangeSearch(n, x, radius, result, params);
}

void IndexHNSW::Search1(const float* x, ResultHandler& handler,
                        SearchParameters* params) const {
  storage->Search1(x, handler, params);
}

void IndexHNSW::PermuteEntries(const idx_t* perm) {
  // Not implemented in minimal HNSW build
}

void IndexHNSW::Reconstruct(idx_t key, float* recons) const {
  storage->Reconstruct(key, recons);
}

DistanceComputer* IndexHNSW::GetDistanceComputer() const {
  return storage->GetDistanceComputer();
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
  is_trained = true;
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
  : IndexHNSW(
      (metric == kMetricL2) ? new IndexFlatL2(d) : new IndexFlat(d, metric),
      M) {
  own_fields = true;
  is_trained = true;
}

}  // namespace hypervec