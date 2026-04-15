/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <core/hypervec_assert.h>
#include <index/idmap/index_id_map.h>
#include <search/aux_index_structures.h>
#include <utils/structures/heap.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace hypervec {

/*****************************************************
 * IndexIDMap implementation
 *******************************************************/

IndexIDMap::IndexIDMap(Index* index) : Index(index->d, index->metric_type) {
  this->index = index;
  this->n_total = index->n_total;
  this->is_trained = index->is_trained;
  this->verbose = index->verbose;
}

idx_t IndexIDMap::to_internal(idx_t id) const {
  auto it = id_map.find(id);
  if (it == id_map.end()) {
    return -1;
  }
  return it->second;
}

idx_t IndexIDMap::from_internal(idx_t id) const {
  if (id < 0 || id >= (idx_t)rev_map.size()) {
    return -1;
  }
  return rev_map[id];
}

void IndexIDMap::Add(idx_t n, const float* x) {
  // Store original IDs
  std::vector<idx_t> ids(n);
  for (idx_t i = 0; i < n; i++) {
    ids[i] = index->n_total + i;
    id_map[ids[i]] = index->n_total + i;
    if (maintain_rev_map) {
      rev_map.push_back(ids[i]);
    }
  }
  index->Add(n, x);
  this->n_total = index->n_total;
}

void IndexIDMap::Search(idx_t n, const float* x, idx_t k, float* distances,
                        idx_t* labels, const SearchParameters* params) const {
  index->Search(n, x, k, distances, labels, params);
  // Translate labels
  for (idx_t i = 0; i < n * k; i++) {
    if (labels[i] >= 0) {
      labels[i] = from_internal(labels[i]);
    }
  }
}

void IndexIDMap::RangeSearch(idx_t n, const float* x, float radius,
                              RangeSearchResult* result,
                              const SearchParameters* params) const {
  index->RangeSearch(n, x, radius, result, params);
}

void IndexIDMap::Reset() {
  id_map.clear();
  rev_map.clear();
  index->Reset();
  this->n_total = 0;
}

void IndexIDMap::Reconstruct(idx_t key, float* recons) const {
  idx_t internal_key = to_internal(key);
  if (internal_key < 0) {
    HYPERVEC_THROW_MSG("key not found");
  }
  index->Reconstruct(internal_key, recons);
}

void IndexIDMap::check_consistency() const {
  if (id_map.size() != rev_map.size()) {
    HYPERVEC_THROW_MSG("inconsistency between id_map and rev_map");
  }
  for (auto& p : id_map) {
    if (rev_map[p.second] != p.first) {
      HYPERVEC_THROW_MSG("inconsistency in id_map / rev_map");
    }
  }
}

void IndexIDMap::MergeFrom(Index& otherIndex, idx_t add_id) {
  HYPERVEC_THROW_MSG("not implemented");
}

void IndexIDMap::construct_rev_map() {
  rev_map.resize(id_map.size());
  for (auto& p : id_map) {
    rev_map[p.second] = p.first;
  }
}

size_t IndexIDMap::RemoveIds(const IDSelector& sel) {
  HYPERVEC_THROW_MSG("not implemented");
  return 0;
}

}  // namespace hypervec
