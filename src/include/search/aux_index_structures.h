/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// Auxiliary index structures, that are used in indexes but that can
// be forward-declared

#ifndef HYPERVEC_AUX_INDEX_STRUCTURES_H
#define HYPERVEC_AUX_INDEX_STRUCTURES_H

#include <core/metric_type.h>
#include <core/platform_macros.h>
#include <stdint.h>

#include <cstring>
#include <memory>
#include <mutex>
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

/****************************************************************
 * Result structures for range Search.
 *
 * The main constraint here is that we want to support parallel
 * queries from different threads in various ways: 1 thread per query,
 * several threads per query. We store the actual results in blocks of
 * fixed size rather than exponentially increasing memory. At the end,
 * we copy the block content to a linear result array.
 *****************************************************************/

/** List of temporary buffers used to store results before they are
 *  copied to the RangeSearchResult object. */
struct BufferList {
  // buffer sizes in # entries
  size_t buffer_size;

  struct Buffer {
    idx_t* ids;
    float* dis;
  };

  std::vector<Buffer> buffers;
  size_t wp;  ///< write pointer in the last buffer.

  explicit BufferList(size_t buffer_size);

  ~BufferList();

  /// create a new buffer
  void AppendBuffer();

  /// Add one result, possibly appending a new buffer if needed
  void Add(idx_t id, float dis);

  /// copy elements ofs:ofs+n-1 seen as linear data in the buffers to
  /// tables dest_ids, dest_dis
  void CopyRange(size_t ofs, size_t n, idx_t* dest_ids, float* dest_dis);
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

/***********************************************************
 * Interrupt callback
 ***********************************************************/

struct HYPERVEC_API InterruptCallback {
  virtual bool WantInterrupt() = 0;
  virtual ~InterruptCallback() {}

  // lock that protects concurrent calls to IsInterrupted
  static std::mutex lock;

  static std::unique_ptr<InterruptCallback> instance;

  static void ClearInstance();

  /** check if:
   * - an interrupt callback is set
   * - the callback returns true
   * if this is the case, then throw an exception. Should not be called
   * from multiple threads.
   */
  static void check();

  /// same as check() but return true if is interrupted instead of
  /// throwing. Can be called from multiple threads.
  static bool IsInterrupted();

  /** assuming each iteration takes a certain number of flops, what
   * is a reasonable interval to check for interrupts?
   */
  static size_t GetPeriodHint(size_t flops);
};

struct TimeoutCallback : InterruptCallback {
  std::chrono::time_point<std::chrono::steady_clock> start;
  double timeout;
  bool WantInterrupt() override;
  void SetTimeout(double timeout_in_seconds);
  static void Reset(double timeout_in_seconds);
};

}  // namespace hypervec

#endif
