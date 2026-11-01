/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <quantization/pq/index_pq.h>
#include <utils/log/assert.h>

namespace hypervec {

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
  HYPERVEC_THROW_IF_NOT_MSG(storage != nullptr,
                            "IndexHNSWPQ::Add: storage is null");
  HYPERVEC_THROW_IF_NOT_MSG(
    raw_storage != nullptr,
    "IndexHNSWPQ::Add: index is frozen (raw scaffold has been released or "
    "the index was deserialized) — Add not allowed");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexHNSWPQ::Add: call Train before Add");
  AddImpl(n, x, raw_storage, raw_storage);
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
