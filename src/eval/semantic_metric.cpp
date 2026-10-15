/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/semantic_metric.h>
#include <utils/log/assert.h>

#include <cmath>
#include <cstddef>

namespace hypervec {

namespace {

constexpr double kUnitSquaredNormTolerance = 1e-4;

void ValidateUnitRows(const FloatVectorDataset& dataset) {
  const size_t dimension = static_cast<size_t>(dataset.dimension);
  for (idx_t row = 0; row < dataset.vector_count; ++row) {
    const size_t offset = static_cast<size_t>(row) * dimension;
    double squared_norm = 0.0;
    for (size_t column = 0; column < dimension; ++column) {
      const double value = dataset.values[offset + column];
      squared_norm += value * value;
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        std::abs(squared_norm - 1.0) <= kUnitSquaredNormTolerance,
        "cosine evaluation requires L2-normalized vectors");
  }
}

}  // namespace

std::string_view SemanticMetricName(SemanticMetric metric) {
  switch (metric) {
    case SemanticMetric::kL2:
      return "l2";
    case SemanticMetric::kInnerProduct:
      return "inner_product";
    case SemanticMetric::kCosine:
      return "cosine";
  }
  HYPERVEC_THROW_MSG("unknown semantic metric");
}

MetricType IndexMetricForSemanticMetric(SemanticMetric metric) {
  switch (metric) {
    case SemanticMetric::kL2:
      return kMetricL2;
    case SemanticMetric::kInnerProduct:
    case SemanticMetric::kCosine:
      return kMetricInnerProduct;
  }
  HYPERVEC_THROW_MSG("unknown semantic metric");
}

void ValidateSemanticMetricDataset(const FloatVectorDataset& dataset,
                                   SemanticMetric metric) {
  ValidateFloatVectorDataset(dataset);
  (void)IndexMetricForSemanticMetric(metric);
  if (metric == SemanticMetric::kCosine) {
    ValidateUnitRows(dataset);
  }
}

void ValidateSemanticMetricIndex(MetricType index_metric,
                                 SemanticMetric semantic_metric) {
  HYPERVEC_THROW_IF_NOT_MSG(
      index_metric == IndexMetricForSemanticMetric(semantic_metric),
      "index metric does not match the evaluation semantic metric");
}

}  // namespace hypervec
