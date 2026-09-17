/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/monitor/monitor.h"

#include <string>
#include <utility>

namespace hypervec::eval {

Monitor::Monitor(std::string name) : name_(std::move(name)) {}

}  // namespace hypervec::eval
