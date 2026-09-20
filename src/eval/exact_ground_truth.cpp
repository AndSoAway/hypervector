/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/exact_ground_truth.h>
#include <index/flat/index_flat.h>
#include <utils/log/assert.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

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
        "cosine ground truth requires L2-normalized vectors");
  }
}

size_t ResultSize(idx_t query_count, idx_t k) {
  const uint64_t rows = static_cast<uint64_t>(query_count);
  const uint64_t neighbors = static_cast<uint64_t>(k);
  HYPERVEC_THROW_IF_NOT_MSG(
      rows <= (std::numeric_limits<size_t>::max)() / neighbors,
      "ground-truth result exceeds addressable memory");
  return static_cast<size_t>(rows * neighbors);
}

}  // namespace

IntegerVectorDataset ComputeExactGroundTruth(const FloatVectorDataset& base,
                                             const FloatVectorDataset& queries,
                                             idx_t k,
                                             GroundTruthMetric metric) {
  ValidateFloatVectorDataset(base);
  ValidateFloatVectorDataset(queries);
  HYPERVEC_THROW_IF_NOT_MSG(base.dimension == queries.dimension,
                            "base and query dimensions must match");
  HYPERVEC_THROW_IF_NOT_MSG(k > 0 && k <= base.vector_count,
                            "ground-truth k must be positive and no larger "
                            "than the base vector count");
  HYPERVEC_THROW_IF_NOT_MSG(
      base.vector_count <= (std::numeric_limits<int32_t>::max)(),
      "base vector count exceeds the ivecs label range");
  HYPERVEC_THROW_IF_NOT_MSG(k <= (std::numeric_limits<int32_t>::max)(),
                            "ground-truth k exceeds the ivecs dimension range");

  MetricType index_metric = kMetricL2;
  switch (metric) {
    case GroundTruthMetric::kL2:
      index_metric = kMetricL2;
      break;
    case GroundTruthMetric::kInnerProduct:
      index_metric = kMetricInnerProduct;
      break;
    case GroundTruthMetric::kCosine:
      ValidateUnitRows(base);
      ValidateUnitRows(queries);
      index_metric = kMetricInnerProduct;
      break;
    default:
      HYPERVEC_THROW_MSG("unsupported ground-truth metric");
  }

  const size_t result_size = ResultSize(queries.vector_count, k);
  std::vector<float> distances(result_size);
  IntegerVectorDataset result;
  result.vector_count = queries.vector_count;
  result.dimension = static_cast<int32_t>(k);
  result.values.resize(result_size);

  IndexFlat index(base.dimension, index_metric);
  index.Add(base.vector_count, base.values.data());
  index.Search(queries.vector_count, queries.values.data(), k, distances.data(),
               result.values.data());
  for (float distance : distances) {
    HYPERVEC_THROW_IF_NOT_MSG(
        std::isfinite(distance),
        "exact ground-truth distance is not finite; rescale the dataset");
  }
  return result;
}

}  // namespace hypervec
