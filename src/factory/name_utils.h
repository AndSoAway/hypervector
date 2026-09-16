/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <string>
#include <string_view>

namespace hypervec::factory_detail {

/** Lower-case ASCII letters and replace '-' with '_'. */
std::string NormalizeToken(std::string_view value);

/** Normalize a user-facing identifier and reject non-ASCII name bytes. */
std::string NormalizeName(std::string_view value, std::string_view context,
                          bool allow_empty = false);

}  // namespace hypervec::factory_detail
