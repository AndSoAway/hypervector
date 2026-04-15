/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <hypervec/impl/ScalarQuantizer.h>
#include <map>

namespace hypervec::perf_tests {

std::map<std::string, hypervec::ScalarQuantizer::QuantizerType> sq_types();

} // namespace hypervec::perf_tests
