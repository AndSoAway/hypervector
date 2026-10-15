/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>

namespace hypervec {

struct LocalVectorQuantizer;

// Internal ISA entry point. Only call via GetDistanceKernel().
float LVQ8DistanceAVX2(const LocalVectorQuantizer& lvq, const float* query,
                       const uint8_t* code);

}  // namespace hypervec
