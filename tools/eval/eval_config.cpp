/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/eval_config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hypervec::eval {

void ValidateEvalConfig(const EvalConfig& config) {
  if (config.base_path.empty()) {
    throw std::invalid_argument("base_path is required");
  }
  if (config.query_path.empty()) {
    throw std::invalid_argument("query_path is required");
  }
  if (config.groundtruth_path.empty()) {
    throw std::invalid_argument("groundtruth_path is required");
  }
  if (config.index_config_path.empty()) {
    throw std::invalid_argument("index_config is required");
  }
  if (config.action != "build" && config.action != "search" &&
      config.action != "build_search") {
    throw std::invalid_argument(
        "action must be build, search, or build_search");
  }
  if (config.action == "search" && config.index_path.empty()) {
    throw std::invalid_argument(
        "index_path is required for a search-only case");
  }
  if (config.top_k <= 0) {
    throw std::invalid_argument("top_k must be positive");
  }
  if (config.num_threads_building <= 0 || config.num_threads_searching <= 0) {
    throw std::invalid_argument("thread counts must be positive");
  }
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open configuration file: " + path);
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    throw std::runtime_error("failed to read configuration file: " + path);
  }
  return contents.str();
}

}  // namespace hypervec::eval
