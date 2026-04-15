/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#ifndef HYPERVEC_METRIC_TYPE_H
#define HYPERVEC_METRIC_TYPE_H

#include <core/hypervec_assert.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace hypervec {

/// The metric space for vector comparison for HyperVec indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
///
/// NOTE: when adding or removing values, update MetricTypeFromInt()
///       and MetricTypeCount() below.
enum MetricType {
  kMetricInnerProduct,  ///< maximum inner product Search
  kMetricL2,             ///< squared L2 Search
  kMetricL1,             ///< L1 (aka cityblock)
  kMetricLinf,           ///< infinity distance
  kMetricLp,             ///< L_p distance, p is given by a hypervec::Index
                         /// metric_arg

  /// some additional metrics defined in scipy.spatial.distance
  kMetricCanberra = 20,
  kMetricBrayCurtis,
  kMetricJensenShannon,

  /// sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i)) where a_i, b_i > 0
  kMetricJaccard,
  /// Squared Euclidean distance, ignoring NaNs
  kMetricNaNEuclidean,
  /// Gower's distance - numeric dimensions are in [0,1] and categorical
  /// dimensions are negative integers
  kMetricGower,
};

/// all vector indices are this type
using idx_t = int64_t;

/// this function is used to distinguish between min and max indexes since
/// we need to support similarity and dis-similarity metrics in a flexible way
constexpr bool IsSimilarityMetric(MetricType metric_type) {
  return ((metric_type == kMetricInnerProduct) ||
          (metric_type == kMetricJaccard));
}

/// Convert an integer to MetricType with range validation.
/// Throws HypervecException if the value is not a valid MetricType.
inline MetricType MetricTypeFromInt(int x) {
  HYPERVEC_THROW_IF_NOT_FMT((x >= kMetricInnerProduct && x <= kMetricLp) ||
                              (x >= kMetricCanberra && x <= kMetricGower),
                            "invalid metric type %d", x);
  return static_cast<MetricType>(x);
}

/// Count of entries in the MetricType enum.
constexpr size_t MetricTypeCount() {
  return (kMetricLp - kMetricInnerProduct) + 1 +
         (kMetricGower - kMetricCanberra) + 1;
}

}  // namespace hypervec

#endif
