/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/distances/metric_type.h>

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

/** Read the standard ANN fvecs format (int32 dimension + float32 row). */
FloatVectorDataset ReadFvecsFile(const std::string& filename);

/** Read the standard ANN ivecs format (int32 dimension + int32 row). */
IntegerVectorDataset ReadIvecsFile(const std::string& filename);

}  // namespace hypervec
