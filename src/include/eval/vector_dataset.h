/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/distances/metric_type.h>

#include <cstdint>
#include <string>
#include <vector>

namespace hypervec {

/** Dense row-major vectors loaded from an evaluation dataset. */
template <typename Value>
struct VectorDataset {
  idx_t vector_count = 0;
  idx_t dimension = 0;
  std::vector<Value> values;
};

using FloatVectorDataset = VectorDataset<float>;
using IntegerVectorDataset = VectorDataset<idx_t>;

/** Source-row IDs for one deterministic, disjoint base/query split. */
struct DatasetRowSplit {
  std::vector<idx_t> base_rows;
  std::vector<idx_t> query_rows;
};

/** Read the standard ANN fvecs format (int32 dimension + float32 row). */
FloatVectorDataset ReadFvecsFile(const std::string& filename);

/** Read the standard ANN ivecs format (int32 dimension + int32 row). */
IntegerVectorDataset ReadIvecsFile(const std::string& filename);

/** Validate shape and finite values before preparing an evaluation dataset. */
void ValidateFloatVectorDataset(const FloatVectorDataset& dataset);

/** Normalize every row to unit L2 norm in place.
 *
 * Rejects zero-norm and non-finite rows instead of silently emitting an
 * undefined cosine-search workload.
 */
void NormalizeL2(FloatVectorDataset* dataset);

/** Select query_count rows with a stable seeded algorithm.
 *
 * Both returned lists preserve source order. Every source row occurs exactly
 * once, so query vectors cannot remain in the base dataset.
 */
DatasetRowSplit MakeDatasetRowSplit(idx_t vector_count, idx_t query_count,
                                    uint64_t seed);

/** Write all rows in standard ANN fvecs format. */
void WriteFvecsFile(const std::string& filename,
                    const FloatVectorDataset& dataset);

/** Write selected source rows in standard ANN fvecs format. */
void WriteFvecsRowsFile(const std::string& filename,
                        const FloatVectorDataset& dataset,
                        const std::vector<idx_t>& rows);

}  // namespace hypervec
