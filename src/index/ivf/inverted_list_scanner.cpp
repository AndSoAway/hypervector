/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/inverted_list_scanner.h>
#include <utils/log/assert.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <vector>

namespace hypervec {
namespace {

void ValidateList(const uint8_t* codes, const idx_t* ids, size_t count,
                  size_t code_size, const char* operation) {
  HYPERVEC_THROW_IF_NOT_FMT(
      count == 0 || codes != nullptr,
      "InvertedListScanner::%s: codes must not be null when count is positive",
      operation);
  HYPERVEC_THROW_IF_NOT_FMT(
      count == 0 || ids != nullptr,
      "InvertedListScanner::%s: ids must not be null when count is positive",
      operation);
  mul_no_overflow(count, code_size, "InvertedListScanner list byte size");
}

}  // namespace

InvertedListScanner::InvertedListScanner(MetricType metric, size_t code_size)
    : metric_(MetricTypeFromInt(static_cast<int>(metric))),
      code_size_(code_size) {
  HYPERVEC_THROW_IF_NOT_MSG(code_size > 0,
                            "InvertedListScanner: code_size must be positive");
}

size_t InvertedListScanner::ScanKnn(const uint8_t* codes, const idx_t* ids,
                                    size_t count, const IDSelector* selector,
                                    idx_t k, float* heap_distances,
                                    idx_t* heap_ids) const {
  HYPERVEC_THROW_IF_NOT_MSG(k > 0,
                            "InvertedListScanner::ScanKnn: k must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      heap_distances != nullptr && heap_ids != nullptr,
      "InvertedListScanner::ScanKnn: result heap must not be null");
  ValidateList(codes, ids, count, code_size_, "ScanKnn");

  size_t distance_count = 0;
  float threshold = heap_distances[0];
  if (IsSimilarityMetric(metric_)) {
    for (size_t i = 0; i < count; ++i) {
      if (selector != nullptr && !selector->IsMember(ids[i])) {
        continue;
      }
      const float distance = DistanceToCode(codes + i * code_size_);
      ++distance_count;
      if (CMin<float, idx_t>::cmp(threshold, distance)) {
        heap_replace_top<CMin<float, idx_t>>(k, heap_distances, heap_ids,
                                             distance, ids[i]);
        threshold = heap_distances[0];
      }
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      if (selector != nullptr && !selector->IsMember(ids[i])) {
        continue;
      }
      const float distance = DistanceToCode(codes + i * code_size_);
      ++distance_count;
      if (CMax<float, idx_t>::cmp(threshold, distance)) {
        heap_replace_top<CMax<float, idx_t>>(k, heap_distances, heap_ids,
                                             distance, ids[i]);
        threshold = heap_distances[0];
      }
    }
  }
  return distance_count;
}

size_t InvertedListScanner::ScanRange(
    const uint8_t* codes, const idx_t* ids, size_t count,
    const IDSelector* selector, float radius,
    std::vector<std::pair<float, idx_t>>* results) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      results != nullptr,
      "InvertedListScanner::ScanRange: results must not be null");
  ValidateList(codes, ids, count, code_size_, "ScanRange");

  size_t distance_count = 0;
  const bool similarity = IsSimilarityMetric(metric_);
  for (size_t i = 0; i < count; ++i) {
    if (selector != nullptr && !selector->IsMember(ids[i])) {
      continue;
    }
    const float distance = DistanceToCode(codes + i * code_size_);
    ++distance_count;
    if ((similarity && distance >= radius) ||
        (!similarity && distance <= radius)) {
      results->emplace_back(distance, ids[i]);
    }
  }
  return distance_count;
}

}  // namespace hypervec
