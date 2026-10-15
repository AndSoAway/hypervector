/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/distances/metric_type.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hypervec {

struct IDSelector;

/** Per-thread state for scanning one encoded IVF list at a time.
 *
 * Implementations own query-dependent scratch state and interpret one fixed
 * width code layout. The common loops apply ID filtering and preserve the
 * metric's k-NN and range-search ordering. Scanner instances are not thread
 * safe; callers must create independent instances for concurrent workers.
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

  /** Scan one list into an already initialized result heap.
   *
   * Returns the number of codes evaluated after ID filtering.
   */
  size_t ScanKnn(const uint8_t* codes, const idx_t* ids, size_t count,
                 const IDSelector* selector, idx_t k, float* heap_distances,
                 idx_t* heap_ids) const;

  /** Append entries satisfying the metric-aware radius comparison.
   *
   * Distance metrics accept values <= radius; similarity metrics accept
   * values >= radius. Returns the number of codes evaluated after filtering.
   */
  size_t ScanRange(const uint8_t* codes, const idx_t* ids, size_t count,
                   const IDSelector* selector, float radius,
                   std::vector<std::pair<float, idx_t>>* results) const;

 private:
  MetricType metric_;
  size_t code_size_;
};

}  // namespace hypervec
