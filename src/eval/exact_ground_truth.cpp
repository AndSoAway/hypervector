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
  ValidateSemanticMetricDataset(base, metric);
  ValidateSemanticMetricDataset(queries, metric);
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

  const MetricType index_metric = IndexMetricForSemanticMetric(metric);

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
