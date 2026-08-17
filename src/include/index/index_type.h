/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/common/platform_macros.h>

#include <string>

namespace hypervec {

enum class IndexType {
  kFlat,
  kIvfFlat,
  kIvfLvq,
  kIvfPq,
  kHnswFlat,
  kHnswLvq,
  kHnswPq,
};

HYPERVEC_API bool ParseIndexType(const std::string& value,
                                 IndexType* index_type);
HYPERVEC_API const char* IndexTypeName(IndexType index_type);

}  // namespace hypervec
