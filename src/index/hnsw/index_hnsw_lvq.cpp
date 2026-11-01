/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <utils/log/assert.h>

namespace hypervec {

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
  HYPERVEC_THROW_IF_NOT_MSG(storage != nullptr,
                            "IndexHNSWLVQ::Add: storage is null");
  HYPERVEC_THROW_IF_NOT_MSG(
    raw_storage != nullptr,
    "IndexHNSWLVQ::Add: index is frozen or deserialized");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexHNSWLVQ::Add: call Train before Add");
  AddImpl(n, x, raw_storage, raw_storage);
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

void IndexHNSWLVQ::PermuteEntries(const idx_t* perm) {
  PermuteEntriesImpl(perm, raw_storage);
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
