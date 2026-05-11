/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 (the "License") found in the
 * LICENSE file in the root directory of this source tree.

 * HNSW-only index clone implementation
 */

#include <utils/log/assert.h>
#include <persistence/index_clone.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <quantization/pq/index_pq.h>

namespace hypervec {

// Clone IndexHNSW and its subclasses
IndexHNSW* clone_IndexHNSW(const IndexHNSW* ihnsw) {
  // IndexHNSWPQ first — must precede IndexHNSWFlat / IndexHNSW because
  // dynamic_cast matches the most-derived type and the base-class branches
  // would otherwise swallow it.
  if (auto* hnswpq = dynamic_cast<const IndexHNSWPQ*>(ihnsw)) {
    auto* pq_storage = dynamic_cast<const IndexPQ*>(hnswpq->storage);
    HYPERVEC_THROW_IF_NOT_MSG(
      pq_storage != nullptr,
      "clone_IndexHNSW(IndexHNSWPQ): inner storage is not an IndexPQ");
    // Untrained shell — symmetric with IndexHNSWFlat's clone behavior, except
    // PQ requires a Train() call before Add(). M_hnsw = 32 to match the flat
    // clone default; HNSW M is not recoverable from the original via
    // public accessors.
    return new IndexHNSWPQ(hnswpq->d, static_cast<int>(pq_storage->pq.M),
                           pq_storage->pq.nbits, 32, hnswpq->metric_type);
  }
  if (dynamic_cast<const IndexHNSWFlat*>(ihnsw)) {
    return new IndexHNSWFlat(ihnsw->d, 32, ihnsw->metric_type);
  }
  if (dynamic_cast<const IndexHNSW*>(ihnsw)) {
    Index* storage = clone_index(ihnsw->storage);
    auto* res = new IndexHNSW(storage, 32);
    res->is_trained = ihnsw->is_trained;
    return res;
  }
  HYPERVEC_THROW_MSG("clone not supported for this type of IndexHNSW");
}

Index* clone_index(const Index* index) {
  const IndexHNSW* ihnsw = dynamic_cast<const IndexHNSW*>(index);
  if (ihnsw) {
    return clone_IndexHNSW(ihnsw);
  }

  // Try other index types that are still available
  const IndexFlat* iflat = dynamic_cast<const IndexFlat*>(index);
  if (iflat) {
    if (dynamic_cast<const IndexFlatL2*>(index)) {
      return new IndexFlatL2(iflat->d);
    }
    if (dynamic_cast<const IndexFlatIP*>(index)) {
      return new IndexFlatIP(iflat->d);
    }
    return new IndexFlat(iflat->d, iflat->metric_type);
  }

  const IndexIDMap* idxmap = dynamic_cast<const IndexIDMap*>(index);
  if (idxmap) {
    Index* underlying = clone_index(idxmap->index);
    auto* res = new IndexIDMap(underlying);
    return res;
  }

  HYPERVEC_THROW_MSG("clone not supported for this type of Index");
}

}  // namespace hypervec