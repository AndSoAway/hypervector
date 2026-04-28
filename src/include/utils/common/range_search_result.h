/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once


#include <utils/distances/metric_type.h>

#include <memory>
#include <vector>

namespace hypervec {

/** The objective is to have a simple result structure while
 *  minimizing the number of mem copies in the result. The method
 *  DoAllocation can be overloaded to allocate the result tables in
 *  the matrix type of a scripting language like Lua or Python. */
struct RangeSearchResult {
  size_t nq;     ///< nb of queries
  size_t* lims;  ///< size (nq + 1)

  idx_t* labels;     ///< result for query i is labels[lims[i]:lims[i+1]]
  float* distances;  ///< corresponding distances (not sorted)

  size_t buffer_size;  ///< size of the result buffers used

  /// lims must be allocated on input to RangeSearch.
  explicit RangeSearchResult(size_t nq, bool alloc_lims = true);

  /// called when lims contains the nb of elements result entries
  /// for each query
  virtual void DoAllocation();

  virtual ~RangeSearchResult();
};

struct RangeSearchPartialResult;

/// result structure for a single query
struct RangeQueryResult {
  idx_t qno;    //< id of the query
  size_t nres;  //< nb of results for this query
  RangeSearchPartialResult* pres;

  /// called by Search function to report a new result
  void Add(float dis, idx_t id);
};

/// the entries in the buffers are split per query
struct RangeSearchPartialResult : BufferList {
  RangeSearchResult* res;

  /// eventually the result will be stored in res_in
  explicit RangeSearchPartialResult(RangeSearchResult* res_in);

  /// query ids + nb of results per query.
  std::vector<RangeQueryResult> queries;

  /// begin a new result
  RangeQueryResult& NewResult(idx_t qno);

  /*****************************************
   * functions used at the end of the Search to merge the result
   * lists */
  void Finalize();

  /// called by RangeSearch before DoAllocation
  void SetLims();

  /// called by RangeSearch after DoAllocation
  void CopyResult(bool incremental = false);

  /// merge a set of PartialResult's into one RangeSearchResult
  /// on output the partialresults are empty!
  static void merge(std::vector<RangeSearchPartialResult*>& partial_results,
                    bool do_delete = true);
};

}  // namespace hypervec