/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hypervec::eval {

struct EvalConfig {
  std::string name = "hypervec_eval";
  std::string base_path;
  std::string query_path;
  std::string groundtruth_path;
  std::string action = "build_search";
  std::string index_config_path;
  std::string search_config_path;
  std::string index_path;
  int top_k = 10;
  std::uint64_t search_query_count = 0;
  int num_threads_building = 1;
  int num_threads_searching = 1;
  bool enable_recall = true;
  bool enable_percent_recall = true;
  bool enable_qps = true;
  bool enable_tps = true;
  bool enable_memory = true;
  bool enable_latency = true;
  bool enable_percent_latency = true;

  // Populated by RunEvalCase after the public Engine normalizes the configs.
  std::string canonical_index_config;
  std::string canonical_search_config;
};

struct EvalSuiteConfig {
  std::vector<EvalConfig> cases;
  std::string output_format = "table";
  std::string output_path;
  bool append_output = false;
};

void ValidateEvalConfig(const EvalConfig& config);
std::string ReadTextFile(const std::string& path);

}  // namespace hypervec::eval
