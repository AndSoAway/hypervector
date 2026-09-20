/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/hnsw/visited_table.h>
#include <quantization/pq/index_pq.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>

#include <memory>

#include "index/hnsw/hnsw_build_utils.h"

namespace hypervec {

namespace {

/// Mirrors the file-private helper at index_hnsw.cpp:45-51. Wraps similarity
/// metrics in NegativeDistanceComputer so HNSW always sees a "smaller is
/// better" computer. T1 IndexHNSWPQ is L2-only, so the wrap is unreachable
/// today; kept symmetric with IndexHNSW for future extension.
DistanceComputer* StorageDistanceComputer(const Index* storage) {
  if (IsSimilarityMetric(storage->metric_type)) {
    return new NegativeDistanceComputer(storage->GetDistanceComputer());
  }
  return storage->GetDistanceComputer();
}

}  // namespace

IndexHNSWPQ::IndexHNSWPQ() {
  // Deserialization-only ctor. ReadIndex populates d, n_total, storage, etc.
  is_trained = false;
}

IndexHNSWPQ::IndexHNSWPQ(int d, int M_pq, int nbits, int M_hnsw,
                         MetricType metric)
  : IndexHNSW(d, M_hnsw, metric) {
  HYPERVEC_THROW_IF_NOT_FMT(metric == kMetricL2,
                            "IndexHNSWPQ: T1 supports kMetricL2 only, got "
                            "metric=%d",
                            static_cast<int>(metric));
  storage = new IndexPQ(d, M_pq, nbits, kMetricL2);
  raw_storage = new IndexFlatL2(d);
  own_fields = true;
  is_trained = false;
}

IndexHNSWPQ::~IndexHNSWPQ() {
  // Base IndexHNSW dtor deletes `storage` when own_fields is set. Take care
  // of raw_storage here.
  if (raw_storage) {
    delete raw_storage;
    raw_storage = nullptr;
  }
}

void IndexHNSWPQ::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->Train(n, x);
  is_trained = storage->is_trained;
}

void IndexHNSWPQ::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0, "IndexHNSWPQ::Add: n must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
    raw_storage != nullptr,
    "IndexHNSWPQ::Add: index is frozen (raw scaffold has been released or "
    "the index was deserialized) — Add not allowed");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexHNSWPQ::Add: call Train before Add");
  if (n == 0) {
    return;
  }

  const idx_t n0 = n_total;
  HYPERVEC_THROW_IF_NOT_MSG(
      raw_storage->n_total == n0 && storage->n_total == n0,
      "IndexHNSWPQ::Add: storage counts are inconsistent");

  // Add to both stores; assert their counts agree.
  raw_storage->Add(n, x);
  storage->Add(n, x);
  HYPERVEC_THROW_IF_NOT_MSG(
      raw_storage->n_total == n0 + n && storage->n_total == n0 + n,
      "IndexHNSWPQ::Add: storage did not add the requested vector count");
  n_total = storage->n_total;

  if (hnsw.ef_construction == 0) {
    hnsw.ef_construction = 40;
  }

  hnsw.PrepareLevelTab(static_cast<size_t>(n), false);

  // Graph-construction distances come from the raw scaffold, NOT from
  // PQ-decoded storage. This is the whole point of dual storage.
  std::unique_ptr<DistanceComputer> dis(StorageDistanceComputer(raw_storage));

  OmpLockArray lock_array(static_cast<size_t>(n_total) + 1);

  VisitedTable vt(static_cast<size_t>(n_total));

  for (idx_t i = n0; i < n_total; ++i) {
    const int pt_level = hnsw.levels[i] - 1;  // levels store level+1
    dis->SetQuery(x + (i - n0) * d);
    hnsw.AddWithLocks(*dis, pt_level, static_cast<int>(i), lock_array.Get(), vt,
                      false);
  }
}

void IndexHNSWPQ::Reset() {
  hnsw.Reset();
  if (storage) {
    storage->Reset();
  }
  if (raw_storage) {
    raw_storage->Reset();
  }
  n_total = 0;
}

void IndexHNSWPQ::Freeze() {
  if (raw_storage) {
    delete raw_storage;
    raw_storage = nullptr;
  }
}

void IndexHNSWPQ::PermuteEntries(const idx_t* perm) {
  PermuteEntriesImpl(perm, raw_storage);
}

size_t IndexHNSWPQ::SaCodeSize() const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  return storage->SaCodeSize();
}

void IndexHNSWPQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->SaEncode(n, x, bytes);
}

void IndexHNSWPQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->SaDecode(n, bytes, x);
}

void IndexHNSWPQ::Search1(const float* /*x*/, ResultHandler& /*handler*/,
                          SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG("IndexHNSWPQ::Search1 not supported");
}

void IndexHNSWPQ::RangeSearch(idx_t /*n*/, const float* /*x*/,
                              float /*radius*/,
                              RangeSearchResult* /*result*/,
                              const SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG("IndexHNSWPQ::RangeSearch not supported");
}

}  // namespace hypervec
