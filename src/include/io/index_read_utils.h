/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// HNSW-only utils for index_read

#pragma once

#include <io/io.h>

namespace hypervec {

// Forward declarations
struct ProductQuantizer;
struct ScalarQuantizer;

// Placeholder functions (not implemented for HNSW-only)
void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f);
void read_ScalarQuantizer(ScalarQuantizer* ivsc, IOReader* f);

}  // namespace hypervec