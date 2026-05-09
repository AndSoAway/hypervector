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
#include <utility>
#include <vector>

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

/// Aggregate exceptions captured by worker threads (e.g. from an OpenMP
/// parallel region) into a single HypervecException. If only one exception
/// was captured, it is rethrown directly. The int in each pair identifies
/// the worker / loop index that produced the exception.
void handleExceptions(
  std::vector<std::pair<int, std::exception_ptr>>& exceptions);

}  // namespace hypervec
