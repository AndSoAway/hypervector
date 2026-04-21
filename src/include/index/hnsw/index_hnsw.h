/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#pragma once

#include <index/index.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/hnsw.h>
#include <utils/utils.h>

#include <optional>
#include <vector>

namespace hypervec {

struct IndexHNSW;
struct IndexPQ;
struct IndexScalarQuantizer;

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSW : Index {
  typedef HNSW::storage_idx_t storage_idx_t;

  // the link structure
  HNSW hnsw;

  // the sequential storage
  bool own_fields = false;
  Index* storage = nullptr;

  // When set to false, level 0 in the knn graph is not initialized.
  // This option is used by IndexHNSWCagra during copy operations
  // as level 0 knn graph is copied over from the source index.
  bool init_level0 = true;

  // When set to true, all neighbors in level 0 are filled up
  // to the maximum size allowed (2 * M). This option is used by
  // IndexHNSWCagra to create a full base layer graph that is
  // used during copyFrom operations.
  bool keep_max_size_level0 = false;

  // See impl/VisitedTable.h.
  std::optional<bool> use_visited_hashset;

  explicit IndexHNSW(int d = 0, int M = 32, MetricType metric = kMetricL2);
  explicit IndexHNSW(Index* storage, int M = 32);

  ~IndexHNSW() override;

  void Add(idx_t n, const float* x) override;

  /// Trains the storage if needed
  void Train(idx_t n, const float* x) override;

  /// entry point for Search
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void RangeSearch(idx_t n, const float* x, float radius,
                    RangeSearchResult* result,
                    const SearchParameters* params = nullptr) const override;

  /** Search one vector with a custom result handler */
  void Search1(const float* x, ResultHandler& handler,
               SearchParameters* params = nullptr) const override;

  void Reconstruct(idx_t key, float* recons) const override;

  void Reset() override;

  void ShrinkLevel0Neighbors(int size);

  /** Perform Search only on level 0, given the starting points for
   * each vertex.
   *
   * @param search_type 1:perform one Search per nprobe, 2: enqueue
   *                    all entry points
   */
  void SearchLevel0(idx_t n, const float* x, idx_t k,
                      const storage_idx_t* nearest, const float* nearest_d,
                      float* distances, idx_t* labels, int nprobe = 1,
                      int search_type = 1,
                      const SearchParameters* params = nullptr) const;

  /// alternative graph building
  void InitLevel0FromKnngraph(int k, const float* D, const idx_t* I);

  /// alternative graph building
  void InitLevel0FromEntryPoints(int npt, const storage_idx_t* points,
                                      const storage_idx_t* nearests);

  // reorder links from nearest to farthest
  void ReorderLinks();

  void LinkSingletons();

  virtual void PermuteEntries(const idx_t* perm);

  DistanceComputer* GetDistanceComputer() const override;
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlat : IndexHNSW {
  IndexHNSWFlat();
  IndexHNSWFlat(int d, int M, MetricType metric = kMetricL2);
};

}  // namespace hypervec
