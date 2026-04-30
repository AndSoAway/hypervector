/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <invlists/inverted_lists.h>
#include <utils/common/range_search_result.h>
#include <utils/selector/id_selector.h>

#include <vector>

namespace hypervec {

/** Search parameters for IVF-family indexes. */
struct IVFSearchParameters : SearchParameters {
  idx_t nprobe = 1;  ///< number of inverted lists to probe
  ~IVFSearchParameters() override {}
};

/** Abstract base class for Inverted File (IVF) indexes.
 *
 * The index clusters the dataset into nlist cells using k-means during
 * Train().  Each vector is assigned to one cell and stored in the
 * corresponding inverted list.  Search probes nprobe cells and merges
 * results.
 *
 * Subclasses provide EncodeVectors() (how to store a vector as bytes in the
 * list) and SearchPreassigned() (how to compute distances within a list). */
struct IndexIVF : Index {
  idx_t nlist;                   ///< number of inverted lists (cluster cells)
  idx_t nprobe;                  ///< default number of lists to probe per query
  std::vector<float> centroids;  ///< cluster centroids, size nlist * d, row-major

  InvertedLists* invlists;  ///< per-cell vector storage
  bool own_invlists;        ///< whether to delete invlists on destruction

  /** @param d          vector dimension
   *  @param nlist      number of clusters / inverted lists
   *  @param code_size  bytes per stored vector (passed to ArrayInvertedLists)
   *  @param metric     distance metric */
  IndexIVF(idx_t d, idx_t nlist, size_t code_size,
           MetricType metric = kMetricL2);

  ~IndexIVF() override;

  /** Train k-means on n vectors.  Sets is_trained = true. */
  void Train(idx_t n, const float* x) override;

  /** Assign vectors to cells and add them; IDs are sequential from n_total. */
  void Add(idx_t n, const float* x) override;

  /** Assign vectors to cells and add them with explicit IDs. */
  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;

  void Search(idx_t n, const float* x, idx_t k, float* distances,
              idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void RangeSearch(idx_t n, const float* x, float radius,
                   RangeSearchResult* result,
                   const SearchParameters* params = nullptr) const override;

  void Reset() override;

  // -----------------------------------------------------------------------
  // Subclass interface

  /** Encode n raw float vectors into bytes for inverted-list storage.
   *  @param n      number of vectors
   *  @param x      input vectors, size n * d
   *  @param codes  output buffer, size n * invlists->code_size */
  virtual void EncodeVectors(idx_t n, const float* x,
                             uint8_t* codes) const = 0;

  /** Search within pre-selected inverted lists and update heap-formatted
   *  output arrays.  Heaps must be initialised by the caller before the
   *  first call (distances / labels filled with neutral values and -1).
   *
   *  @param n             number of queries
   *  @param x             query vectors, size n * d
   *  @param k             results per query
   *  @param list_ids      list indices to probe, size n * nprobe_actual
   *                       (row i holds nprobe_actual list numbers for query i)
   *  @param centroid_dis  distances to selected centroids, same layout
   *  @param distances     output heap values, size n * k (modified in-place)
   *  @param labels        output heap IDs, size n * k (modified in-place)
   *  @param nprobe_actual number of lists per query in list_ids
   *  @param sel           optional ID selector (nullptr = accept all) */
  virtual void SearchPreassigned(idx_t n, const float* x, idx_t k,
                                 const idx_t* list_ids,
                                 const float* centroid_dis, float* distances,
                                 idx_t* labels, idx_t nprobe_actual,
                                 const IDSelector* sel) const = 0;

  // -----------------------------------------------------------------------
  // Utility (non-virtual, available to subclasses and users)

  /** Find the k nearest centroids for nq queries.
   *  @param nq        number of queries
   *  @param xq        query vectors, size nq * d
   *  @param k         number of centroids per query
   *  @param distances output distances, size nq * k
   *  @param labels    output centroid indices, size nq * k */
  void FindNearestCentroids(idx_t nq, const float* xq, idx_t k,
                            float* distances, idx_t* labels) const;
};

}  // namespace hypervec
