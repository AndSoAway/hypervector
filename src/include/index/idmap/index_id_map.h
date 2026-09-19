/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <utils/selector/id_selector.h>

#include <unordered_map>
#include <vector>

namespace hypervec {

/** Index that translates Search results to ids */
struct IndexIDMap : Index {
  /// Whether this wrapper owns the underlying index.
  bool own_fields = false;

  /// translates the IDs to internal ids
  std::unordered_map<idx_t, idx_t> id_map;

  /// Inverse map, from internal ids to ids
  std::vector<idx_t> rev_map;

  /// Legacy compatibility flag. Correct translation requires both maps, so
  /// current writes always maintain rev_map.
  bool maintain_rev_map = true;

  IndexIDMap() = default;

  explicit IndexIDMap(Index* index);

  ~IndexIDMap() override;

  IndexCapabilities GetCapabilities() const override;

  idx_t to_internal(idx_t id) const;

  idx_t from_internal(idx_t id) const;

  void Train(idx_t n, const float* x) override;

  void Train(idx_t n, const float* x, idx_t n_train_q,
             const float* xq_train) override;

  void Add(idx_t n, const float* x) override;

  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;

  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void RangeSearch(idx_t n, const float* x, float radius,
                   RangeSearchResult* result,
                   const SearchParameters* params = nullptr) const override;

  void Reset() override;

  void Reconstruct(idx_t key, float* recons) const override;

  void check_consistency() const;

  void CheckCompatibleForMerge(const Index& otherIndex) const override;

  void MergeFrom(Index& otherIndex, idx_t add_id) override;

  void construct_rev_map();

  size_t RemoveIds(const IDSelector& sel) override;

  template <class T>
  T* get_index() {
    return dynamic_cast<T*>(index);
  }

  /// Pointer to the underlying index
  Index* index = nullptr;
};

}  // namespace hypervec
