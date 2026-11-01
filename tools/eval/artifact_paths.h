/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace hypervec::eval_cli {

/** Reject lexical, symlink and existing hard-link aliases before any writes. */
inline void ValidateArtifactPaths(std::initializer_list<std::string> inputs,
                                  std::initializer_list<std::string> outputs) {
  std::vector<std::filesystem::path> occupied;
  for (const std::string& input : inputs) {
    if (!input.empty()) {
      occupied.push_back(std::filesystem::weakly_canonical(input));
    }
  }
  for (const std::string& output : outputs) {
    if (output.empty()) {
      continue;
    }
    const auto target = std::filesystem::weakly_canonical(output);
    for (const auto& path : occupied) {
      std::error_code error;
      if (target == path || std::filesystem::equivalent(target, path, error)) {
        throw std::runtime_error(
            "artifact output must differ from all inputs "
            "and other outputs: " +
            output);
      }
    }
    occupied.push_back(target);
  }
}

}  // namespace hypervec::eval_cli
