/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <eval/vector_dataset.h>
#include <utils/distances/metric_type.h>

#include <string_view>

namespace hypervec {

/** Similarity semantics for an evaluation workload.
 *
 * GroundTruthMetric remains the concrete type for source and binary
 * compatibility with the original exact-ground-truth API.
 */
enum class GroundTruthMetric {
  kL2,
  kInnerProduct,
  kCosine,
};
using SemanticMetric = GroundTruthMetric;

/** Stable command/report name for a semantic metric. */
std::string_view SemanticMetricName(SemanticMetric metric);

/** Translate workload semantics to the metric implemented by an index. */
MetricType IndexMetricForSemanticMetric(SemanticMetric metric);

/** Validate finite inputs and cosine unit-norm requirements. */
void ValidateSemanticMetricDataset(const FloatVectorDataset& dataset,
                                   SemanticMetric metric);

/** Reject an index whose stored metric cannot implement the workload. */
void ValidateSemanticMetricIndex(MetricType index_metric,
                                 SemanticMetric semantic_metric);

}  // namespace hypervec
