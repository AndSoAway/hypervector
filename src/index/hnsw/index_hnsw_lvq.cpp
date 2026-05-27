/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/hnsw/index_hnsw_lvq.h>

#include <index/flat/index_flat.h>
#include <index/hnsw/visited_table.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>

#include <omp.h>

#include <vector>

namespace hypervec {

namespace {

DistanceComputer* StorageDistanceComputer(const Index* storage) {
  if (IsSimilarityMetric(storage->metric_type)) {
    return new NegativeDistanceComputer(storage->GetDistanceComputer());
  }
  return storage->GetDistanceComputer();
}

}  // namespace

IndexHNSWLVQ::IndexHNSWLVQ() {
  is_trained = false;
}

IndexHNSWLVQ::IndexHNSWLVQ(int d, int nlocal, int nbits, int M_hnsw,
                           MetricType metric)
  : IndexHNSW(d, M_hnsw, metric) {
  HYPERVEC_THROW_IF_NOT_FMT(
    metric == kMetricL2, "IndexHNSWLVQ: supports kMetricL2 only, got metric=%d",
    static_cast<int>(metric));
  storage = new IndexLVQ(d, nlocal, nbits, kMetricL2);
  raw_storage = new IndexFlatL2(d);
  own_fields = true;
  is_trained = false;
}

IndexHNSWLVQ::~IndexHNSWLVQ() {
  if (raw_storage) {
    delete raw_storage;
    raw_storage = nullptr;
  }
}

void IndexHNSWLVQ::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->Train(n, x);
  is_trained = storage->is_trained;
}

void IndexHNSWLVQ::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
    raw_storage != nullptr,
    "IndexHNSWLVQ::Add: index is frozen or deserialized");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexHNSWLVQ::Add: call Train before Add");
  if (n == 0) {
    return;
  }

  raw_storage->Add(n, x);
  storage->Add(n, x);
  HYPERVEC_THROW_IF_NOT(raw_storage->n_total == storage->n_total);

  const idx_t n0 = n_total;
  n_total = storage->n_total;

  if (hnsw.ef_construction == 0) {
    hnsw.ef_construction = 40;
  }
  hnsw.PrepareLevelTab(n_total, false);

  DistanceComputer* dis = StorageDistanceComputer(raw_storage);
  std::vector<omp_lock_t> locks(static_cast<size_t>(n_total) + 1);
  for (idx_t i = 0; i <= n_total; ++i) {
    omp_init_lock(&locks[i]);
  }

  VisitedTable vt(static_cast<size_t>(n_total));
  for (idx_t i = n0; i < n_total; ++i) {
    const int pt_level = hnsw.levels[i] - 1;
    dis->SetQuery(x + (i - n0) * d);
    hnsw.AddWithLocks(*dis, pt_level, static_cast<int>(i), locks, vt, false);
  }

  for (idx_t i = 0; i <= n_total; ++i) {
    omp_destroy_lock(&locks[i]);
  }
  delete dis;
}

void IndexHNSWLVQ::Reset() {
  hnsw.Reset();
  if (storage) {
    storage->Reset();
  }
  if (raw_storage) {
    raw_storage->Reset();
  }
  n_total = 0;
}

void IndexHNSWLVQ::Freeze() {
  if (raw_storage) {
    delete raw_storage;
    raw_storage = nullptr;
  }
}

size_t IndexHNSWLVQ::SaCodeSize() const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  return storage->SaCodeSize();
}

void IndexHNSWLVQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->SaEncode(n, x, bytes);
}

void IndexHNSWLVQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->SaDecode(n, bytes, x);
}

void IndexHNSWLVQ::Search1(const float* /*x*/, ResultHandler& /*handler*/,
                           SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG("IndexHNSWLVQ::Search1 not supported");
}

void IndexHNSWLVQ::RangeSearch(idx_t /*n*/, const float* /*x*/,
                               float /*radius*/,
                               RangeSearchResult* /*result*/,
                               const SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG("IndexHNSWLVQ::RangeSearch not supported");
}

}  // namespace hypervec
