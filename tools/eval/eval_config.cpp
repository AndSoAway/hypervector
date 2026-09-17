/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/eval_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hypervec::eval {
namespace {

template <typename T>
void SetIfDefined(const YAML::Node& node, const char* key, T& value) {
  if (node && node[key]) {
    value = node[key].as<T>();
  }
}

EvalConfig LoadCase(const YAML::Node& node, const EvalConfig& defaults) {
  EvalConfig result = defaults;
  SetIfDefined(node, "name", result.name);
  SetIfDefined(node, "dataset", result.dataset_path);
  SetIfDefined(node, "dataset_path", result.dataset_path);
  SetIfDefined(node, "action", result.action);
  SetIfDefined(node, "type", result.action);
  SetIfDefined(node, "index_config", result.index_config_path);
  SetIfDefined(node, "index_config_path", result.index_config_path);
  SetIfDefined(node, "search_config", result.search_config_path);
  SetIfDefined(node, "search_config_path", result.search_config_path);
  SetIfDefined(node, "index_path", result.index_path);
  SetIfDefined(node, "top_k", result.top_k);
  SetIfDefined(node, "topk", result.top_k);
  SetIfDefined(node, "search_query_count", result.search_query_count);
  SetIfDefined(node, "num_threads", result.num_threads_building);
  SetIfDefined(node, "num_threads_building", result.num_threads_building);
  SetIfDefined(node, "num_threads_searching", result.num_threads_searching);
  SetIfDefined(node, "enable_recall", result.enable_recall);
  SetIfDefined(node, "enable_percent_recall", result.enable_percent_recall);
  SetIfDefined(node, "enable_qps", result.enable_qps);
  SetIfDefined(node, "enable_tps", result.enable_tps);
  SetIfDefined(node, "enable_memory", result.enable_memory);
  SetIfDefined(node, "enable_latency", result.enable_latency);
  SetIfDefined(node, "enable_percent_latency", result.enable_percent_latency);
  SetIfDefined(node, "use_id_based_recall", result.use_id_based_recall);
  if (node["num_threads"] && !node["num_threads_searching"]) {
    result.num_threads_searching = result.num_threads_building;
  }
  return result;
}

}  // namespace

EvalSuiteConfig LoadEvalConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  EvalSuiteConfig suite;
  EvalConfig defaults;
  if (root["global"]) {
    defaults = LoadCase(root["global"], defaults);
  }
  if (root["output"]) {
    SetIfDefined(root["output"], "format", suite.output_format);
    SetIfDefined(root["output"], "path", suite.output_path);
    SetIfDefined(root["output"], "append", suite.append_output);
  }
  if (root["cases"]) {
    if (!root["cases"].IsSequence()) {
      throw std::invalid_argument("cases must be a YAML sequence");
    }
    for (const auto& node : root["cases"]) {
      suite.cases.push_back(LoadCase(node, defaults));
    }
  } else {
    suite.cases.push_back(LoadCase(root, defaults));
  }
  if (suite.cases.empty()) {
    throw std::invalid_argument(
        "the configuration contains no evaluation cases");
  }
  for (const auto& config : suite.cases) {
    ValidateEvalConfig(config);
  }
  return suite;
}

void ValidateEvalConfig(const EvalConfig& config) {
  if (config.dataset_path.empty()) {
    throw std::invalid_argument("dataset_path is required");
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
