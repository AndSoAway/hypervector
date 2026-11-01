/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace hypervec::eval_cli {

/** Process-wide peak resident set size, in bytes, when supported. */
std::optional<uint64_t> PeakResidentSetBytes();

/** Operating-system source used by PeakResidentSetBytes. */
std::string_view PeakResidentSetSource();

}  // namespace hypervec::eval_cli
