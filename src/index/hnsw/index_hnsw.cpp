/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// HNSW-only index implementation

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/visited_table.h>
#include <omp.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_pq.h>
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
#include <optional>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "index/hnsw/hnsw_build_utils.h"

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

std::vector<idx_t> ValidatePermutation(const idx_t* perm, idx_t count) {
  HYPERVEC_THROW_IF_NOT_MSG(count >= 0,
                            "IndexHNSW::PermuteEntries: negative vector count");
  std::vector<idx_t> inverse(static_cast<size_t>(count));
  if (count == 0) {
    return inverse;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      perm != nullptr,
      "IndexHNSW::PermuteEntries: permutation must not be null");
  std::vector<uint8_t> seen(static_cast<size_t>(count), 0);
  for (idx_t new_id = 0; new_id < count; ++new_id) {
    const idx_t old_id = perm[new_id];
    HYPERVEC_THROW_IF_NOT_FMT(old_id >= 0 && old_id < count,
                              "IndexHNSW::PermuteEntries: entry %" PRId64
                              " is outside [0, %" PRId64 ")",
                              static_cast<int64_t>(old_id),
                              static_cast<int64_t>(count));
    HYPERVEC_THROW_IF_NOT_FMT(
        seen[static_cast<size_t>(old_id)] == 0,
        "IndexHNSW::PermuteEntries: duplicate old id %" PRId64,
        static_cast<int64_t>(old_id));
    seen[static_cast<size_t>(old_id)] = 1;
    inverse[static_cast<size_t>(old_id)] = new_id;
  }
  return inverse;
}

struct PreparedStoragePermutation {
  MaybeOwnedVector<uint8_t>* storage_codes = nullptr;
  MaybeOwnedVector<uint8_t> codes;
  IndexFlatL2* flat_l2 = nullptr;
  IndexFlat1D* flat_1d = nullptr;
  bool replace_flat_1d_permutation = false;
  std::vector<idx_t> flat_1d_permutation;
};

PreparedStoragePermutation PrepareStoragePermutation(
    Index* storage, const idx_t* perm, idx_t count,
    const std::vector<idx_t>& inverse) {
  HYPERVEC_THROW_IF_NOT_MSG(
      storage != nullptr,
      "IndexHNSW::PermuteEntries: storage must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      storage->n_total == count,
      "IndexHNSW::PermuteEntries: storage and graph counts differ");
  size_t code_size = 0;
  MaybeOwnedVector<uint8_t>* storage_codes = nullptr;
  if (auto* flat_codes = dynamic_cast<IndexFlatCodes*>(storage)) {
    code_size = flat_codes->code_size;
    storage_codes = &flat_codes->codes;
  } else if (auto* pq = dynamic_cast<IndexPQ*>(storage)) {
    code_size = pq->pq.code_size;
    storage_codes = &pq->codes;
  } else if (auto* lvq = dynamic_cast<IndexLVQ*>(storage)) {
    code_size = lvq->lvq.code_size;
    storage_codes = &lvq->codes;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      storage_codes != nullptr,
      "IndexHNSW::PermuteEntries: storage does not support code reordering");
  HYPERVEC_THROW_IF_NOT_MSG(
      code_size > 0 || count == 0,
      "IndexHNSW::PermuteEntries: storage code size is invalid");
  const size_t expected_bytes = mul_no_overflow(
      static_cast<size_t>(count), code_size, "IndexHNSW permutation storage");
  HYPERVEC_THROW_IF_NOT_MSG(
      storage_codes->size() == expected_bytes,
      "IndexHNSW::PermuteEntries: encoded storage size is inconsistent");

  PreparedStoragePermutation prepared;
  prepared.storage_codes = storage_codes;
  prepared.codes.resize(expected_bytes);
  for (idx_t new_id = 0; new_id < count; ++new_id) {
    std::memcpy(
        prepared.codes.data() + static_cast<size_t>(new_id) * code_size,
        storage_codes->data() + static_cast<size_t>(perm[new_id]) * code_size,
        code_size);
  }

  prepared.flat_l2 = dynamic_cast<IndexFlatL2*>(storage);
  prepared.flat_1d = dynamic_cast<IndexFlat1D*>(storage);
  if (prepared.flat_1d != nullptr && !prepared.flat_1d->perm.empty()) {
    HYPERVEC_THROW_IF_NOT_MSG(
        prepared.flat_1d->perm.size() == static_cast<size_t>(count),
        "IndexHNSW::PermuteEntries: IndexFlat1D permutation is inconsistent");
    prepared.replace_flat_1d_permutation = true;
    prepared.flat_1d_permutation.resize(static_cast<size_t>(count));
    for (idx_t rank = 0; rank < count; ++rank) {
      const idx_t old_id = prepared.flat_1d->perm[static_cast<size_t>(rank)];
      HYPERVEC_THROW_IF_NOT_MSG(
          old_id >= 0 && old_id < count,
          "IndexHNSW::PermuteEntries: IndexFlat1D contains an invalid id");
      prepared.flat_1d_permutation[static_cast<size_t>(rank)] =
          inverse[static_cast<size_t>(old_id)];
    }
  }
  return prepared;
}

void CommitStoragePermutation(PreparedStoragePermutation* prepared) {
  using std::swap;
  swap(*prepared->storage_codes, prepared->codes);
  if (prepared->flat_l2 != nullptr) {
    prepared->flat_l2->cached_l2norms.clear();
  }
  if (prepared->replace_flat_1d_permutation) {
    prepared->flat_1d->perm.swap(prepared->flat_1d_permutation);
  }
}

void ValidateSearchState(const IndexHNSW& index) {
  HYPERVEC_THROW_IF_NOT_MSG(index.storage != nullptr,
                            "IndexHNSW::Search: storage must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.n_total >= 0,
      "IndexHNSW::Search: vector count must not be negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.storage->n_total == index.n_total,
      "IndexHNSW::Search: storage and index counts differ");

  const size_t count = static_cast<size_t>(index.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      index.hnsw.levels.size() == count &&
          index.hnsw.offsets.size() == count + 1 &&
          index.hnsw.offsets.front() == 0 &&
          index.hnsw.offsets.back() == index.hnsw.neighbors.size(),
      "IndexHNSW::Search: graph storage is inconsistent");
  if (count == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        index.hnsw.entry_point == -1 && index.hnsw.max_level == -1,
        "IndexHNSW::Search: empty graph has an entry point or level");
    return;
  }

  HYPERVEC_THROW_IF_NOT_MSG(
      index.hnsw.entry_point >= 0 &&
          static_cast<size_t>(index.hnsw.entry_point) < count,
      "IndexHNSW::Search: entry point is outside the index");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.hnsw.max_level >= 0 &&
          index.hnsw.levels[static_cast<size_t>(index.hnsw.entry_point)] >
              index.hnsw.max_level,
      "IndexHNSW::Search: entry-point level is inconsistent");
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

IndexCapabilities IndexHNSW::GetCapabilities() const {
  IndexCapabilities capabilities;
  if (storage != nullptr) {
    const IndexCapabilities storage_capabilities = storage->GetCapabilities();
    capabilities.requires_training = storage_capabilities.requires_training;
    capabilities.supports_range_search =
        storage_capabilities.supports_range_search;
    capabilities.supports_reconstruct =
        storage_capabilities.supports_reconstruct;
  }
  return capabilities;
}

void IndexHNSW::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(storage != nullptr,
                            "IndexHNSW::Train: storage is null");
  storage->Train(n, x);
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

  // PrepareLevelTab appends graph metadata, so only pass this batch's size.
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
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexHNSW::Search: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexHNSW::Search: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      k <= (std::numeric_limits<int>::max)(),
      "IndexHNSW::Search: k exceeds the supported graph-search range");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexHNSW::Search: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexHNSW::Search: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexHNSW::Search: labels must not be null when n is positive");
  ValidateSearchState(*this);
  if (n == 0) {
    return;
  }
  (void)mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                        "IndexHNSW::Search output size");
  if (n_total > 0) {
    int ef_search = hnsw.ef_search;
    if (const auto* hnsw_params =
            dynamic_cast<const SearchParametersHNSW*>(params)) {
      ef_search = hnsw_params->ef_search;
    }
    HYPERVEC_THROW_IF_NOT_MSG(ef_search > 0,
                              "IndexHNSW::Search: ef_search must be positive");
  }

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

void IndexHNSW::Search1(const float* x, ResultHandler& handler,
                        SearchParameters* params) const {
  storage->Search1(x, handler, params);
}

void IndexHNSW::PermuteEntries(const idx_t* perm) {
  PermuteEntriesImpl(perm, nullptr);
}

void IndexHNSW::PermuteEntriesImpl(const idx_t* perm,
                                   Index* secondary_storage) {
  const std::vector<idx_t> inverse = ValidatePermutation(perm, n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.levels.size() == static_cast<size_t>(n_total),
      "IndexHNSW::PermuteEntries: graph and index counts differ");

  PreparedStoragePermutation primary =
      PrepareStoragePermutation(storage, perm, n_total, inverse);
  std::optional<PreparedStoragePermutation> secondary;
  if (secondary_storage != nullptr) {
    HYPERVEC_THROW_IF_NOT_MSG(
        secondary_storage != storage,
        "IndexHNSW::PermuteEntries: secondary storage aliases storage");
    secondary.emplace(
        PrepareStoragePermutation(secondary_storage, perm, n_total, inverse));
  }

  // HNSW validates and stages its replacement arrays before swapping them.
  // Once it succeeds, the prepared code-buffer swaps below cannot allocate.
  hnsw.PermuteEntries(perm);
  CommitStoragePermutation(&primary);
  if (secondary.has_value()) {
    CommitStoragePermutation(&secondary.value());
  }
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

IndexHNSWFlat::IndexHNSWFlat() {
  is_trained = true;
}

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
