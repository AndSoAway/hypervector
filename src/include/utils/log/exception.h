/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <stdexcept>
#include <string>

namespace hypervec {

struct HypervecException : std::exception {
  std::string msg;

  explicit HypervecException(const std::string& m) : msg(m) {}

  HypervecException(const std::string& m, const char* funcname,
                    const char* file, int line) {
    msg = std::string(funcname) + ":" + file + ":" + std::to_string(line) +
          ": " + m;
  }

  const char* what() const noexcept override { return msg.c_str(); }
};

}  // namespace hypervec
