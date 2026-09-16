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

#include <memory>
#include <unordered_map>
#include <vector>

namespace hypervec {

/** Index that translates Search results to ids */
struct IndexIDMap : Index {
  /// translates the IDs to internal ids
  std::unordered_map<idx_t, idx_t> id_map;

  /// Inverse map, from internal ids to ids
  std::vector<idx_t> rev_map;

  /// Whether to maintain internal-to-external translation. Search result
  /// translation requires this to be true. If disabled for memory-sensitive
  /// write-only use, call construct_rev_map() before Search or RangeSearch.
  bool maintain_rev_map = true;

  /// Default constructor for deserialization only.
  IndexIDMap();

  explicit IndexIDMap(Index* index);

  explicit IndexIDMap(std::unique_ptr<Index> index);

  ~IndexIDMap() override;

  std::string_view TypeName() const noexcept override { return "idmap"; }

  IndexCapabilities Capabilities() const noexcept override;

  idx_t to_internal(idx_t id) const;

  /// Translate an internal ID. Fails explicitly when maintain_rev_map=false.
  idx_t from_internal(idx_t id) const;

  /// Trains the wrapped index and mirrors its final is_trained state,
  /// including when the wrapped training call throws.
  void Train(idx_t n, const float* x) override;

  /// Build the wrapped index, then publish the external ID mapping. Mapping
  /// publication is exception-safe but is not atomic across process failure.
  void Build(const DenseDatasetView& data) override;

  void Add(idx_t n, const float* x) override;

  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;

  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void RangeSearch(idx_t n, const float* x, float radius,
                   RangeSearchResult* result,
                   const SearchParameters* params = nullptr) const override;

  void Reset() override;

  void Reconstruct(idx_t key, float* recons) const override;

  /// Validate ID-map invariants. The default reverse-map mode uses no
  /// additional memory proportional to n_total.
  void check_consistency() const;

  void MergeFrom(Index& otherIndex, idx_t add_id);

  void construct_rev_map();

  size_t RemoveIds(const IDSelector& sel);

  bool OwnsIndex() const noexcept;

  template <class T>
  T* get_index() {
    return dynamic_cast<T*>(index);
  }

  /// Pointer to the underlying index
  Index* index = nullptr;

 private:
  std::unique_ptr<Index> owned_index_;
};

}  // namespace hypervec
