/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/hnsw/index_hnsw_rabitq.h>

#include <index/flat/index_flat.h>
#include <index/hnsw/visited_table.h>
#include <quantization/rabitq/index_rabitq.h>
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

IndexHNSWRaBitQ::IndexHNSWRaBitQ() {
  // Deserialization-only ctor.
  is_trained = false;
}

IndexHNSWRaBitQ::IndexHNSWRaBitQ(int d, int B, int M_hnsw, MetricType metric)
  : IndexHNSW(d, M_hnsw, metric) {
  HYPERVEC_THROW_IF_NOT_FMT(
    metric == kMetricL2, "IndexHNSWRaBitQ: supports kMetricL2 only, got metric=%d",
    static_cast<int>(metric));
  storage = new IndexRaBitQ(d, B, kMetricL2);
  raw_storage = new IndexFlatL2(d);
  own_fields = true;
  is_trained = false;
}

IndexHNSWRaBitQ::~IndexHNSWRaBitQ() {
  if (raw_storage) {
    delete raw_storage;
    raw_storage = nullptr;
  }
}

void IndexHNSWRaBitQ::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->Train(n, x);
  is_trained = storage->is_trained;
}

void IndexHNSWRaBitQ::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
    raw_storage != nullptr,
    "IndexHNSWRaBitQ::Add: index is frozen or deserialized");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexHNSWRaBitQ::Add: call Train before Add");
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

void IndexHNSWRaBitQ::Reset() {
  hnsw.Reset();
  if (storage) {
    storage->Reset();
  }
  if (raw_storage) {
    raw_storage->Reset();
  }
  n_total = 0;
}

void IndexHNSWRaBitQ::Freeze() {
  if (raw_storage) {
    delete raw_storage;
    raw_storage = nullptr;
  }
}

size_t IndexHNSWRaBitQ::SaCodeSize() const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  return storage->SaCodeSize();
}

void IndexHNSWRaBitQ::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->SaEncode(n, x, bytes);
}

void IndexHNSWRaBitQ::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  HYPERVEC_THROW_IF_NOT(storage != nullptr);
  storage->SaDecode(n, bytes, x);
}

void IndexHNSWRaBitQ::Search1(const float* /*x*/, ResultHandler& /*handler*/,
                              SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG("IndexHNSWRaBitQ::Search1 not supported");
}

void IndexHNSWRaBitQ::RangeSearch(idx_t /*n*/, const float* /*x*/,
                                  float /*radius*/,
                                  RangeSearchResult* /*result*/,
                                  const SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG("IndexHNSWRaBitQ::RangeSearch not supported");
}

}  // namespace hypervec