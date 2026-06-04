/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/distances/fused/distances_fused.h>

namespace hypervec {

bool exhaustive_L2sqr_fused_cmax(
  const float* /*x*/, const float* /*y*/, size_t /*d*/, size_t /*nx*/,
  size_t /*ny*/, Top1BlockResultHandler<CMax<float, int64_t>>& /*res*/,
  const float* /*y_norms*/) {
  return false;
}

}  // namespace hypervec
