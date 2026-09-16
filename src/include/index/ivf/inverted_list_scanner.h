/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <utils/distances/metric_type.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hypervec {

/** Per-thread state for scanning one encoded IVF list at a time.
 *
 * Implementations own query-dependent scratch tables and understand one code
 * layout. The common ScanKnn and ScanRange loops apply filters and preserve
 * the metric's heap/radius semantics. Instances are intentionally not
 * thread-safe; create one scanner per worker thread.
 */
class InvertedListScanner {
 public:
  InvertedListScanner(MetricType metric, size_t code_size);
  virtual ~InvertedListScanner() = default;

  MetricType Metric() const noexcept { return metric_; }
  size_t CodeSize() const noexcept { return code_size_; }

  virtual void SetQuery(const float* query) = 0;
  virtual void SetList(idx_t list_no, float coarse_distance) = 0;
  virtual float DistanceToCode(const uint8_t* code) const = 0;

  /** Scan one list into an already-initialized result heap.
   *
   * Returns the number of codes whose distance was evaluated after filtering.
   */
  size_t ScanKnn(const uint8_t* codes, const idx_t* ids, size_t count,
                 const IDSelector* selector, idx_t k, float* heap_distances,
                 idx_t* heap_ids) const;

  /** Append matching entries from one list to a query's range results.
   *
   * Returns the number of codes whose distance was evaluated after filtering.
   */
  size_t ScanRange(const uint8_t* codes, const idx_t* ids, size_t count,
                   const IDSelector* selector, float radius,
                   std::vector<std::pair<float, idx_t>>* results) const;

 private:
  MetricType metric_;
  size_t code_size_;
};

}  // namespace hypervec
