/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/index_registry.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

// Third-party dependencies.
#include <nlohmann/json.hpp>

#include "eval/case/eval_case.h"
#include "eval/eval_config.h"
#include "eval/exporter/exporter.h"
#include "eval/exporter/formatter.h"

namespace {

using ArgumentMap = std::unordered_map<std::string, std::string>;

void PrintHelp(const char* program) {
  std::cout
      << "HyperVec performance evaluation tool\n\n"
      << "YAML mode:\n  " << program << " --config eval.yaml\n\n"
      << "CLI mode:\n  " << program
      << " --index-config hnsw.json --search-config search.json"
         " --dataset sift128.hdf5 --topk 10\n\n"
      << "Core options:\n"
      << "  --action build|search|build_search  (default: build_search)\n"
      << "  --index-config PATH               public factory JSON config\n"
      << "  --search-config PATH              runtime search JSON config\n"
      << "  --dataset PATH                    HDF5 train/test/ground truth\n"
      << "  --index_path PATH                 save/load a built index\n"
      << "  --topk N --search_query_count N\n"
      << "  --num_threads N or separate --num_threads_building/searching\n"
      << "  --id_recall                      use ID-intersection recall\n"
      << "Output options:\n"
      << "  --format table|json --output PATH --append\n\n"
      << "Index types: ";
  hypervec::RegisterBuiltinIndexes();
  const auto descriptors = hypervec::IndexRegistry::Global().List();
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    std::cout << (i == 0 ? "" : ", ") << descriptors[i]->canonical_name;
  }
  std::cout << '\n';
}

ArgumentMap ParseArguments(int argc, char** argv) {
  ArgumentMap result;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (!key.starts_with("--")) {
      throw std::invalid_argument("unexpected positional argument: " + key);
    }
    if (key == "--help" || key == "--append" || key == "--id_recall" ||
        key.starts_with("--disable_")) {
      result[key] = "true";
      continue;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + key);
    }
    result[key] = argv[++i];
  }
  return result;
}

template <typename T>
void SetInteger(const ArgumentMap& arguments, const char* key, T& value) {
  const auto iterator = arguments.find(key);
  if (iterator == arguments.end()) {
    return;
  }
  if constexpr (std::is_same_v<T, std::uint64_t>) {
    value = std::stoull(iterator->second);
  } else {
    value = std::stoi(iterator->second);
  }
}

void SetString(const ArgumentMap& arguments, const char* key,
               std::string& value) {
  const auto iterator = arguments.find(key);
  if (iterator != arguments.end()) {
    value = iterator->second;
  }
}

bool Has(const ArgumentMap& arguments, const char* key) {
  return arguments.contains(key);
}

hypervec::eval::EvalSuiteConfig LoadCliConfig(const ArgumentMap& arguments) {
  hypervec::eval::EvalSuiteConfig suite;
  hypervec::eval::EvalConfig config;
  SetString(arguments, "--dataset", config.dataset_path);
  SetString(arguments, "--action", config.action);
  SetString(arguments, "--index-config", config.index_config_path);
  SetString(arguments, "--search-config", config.search_config_path);
  SetString(arguments, "--index_path", config.index_path);
  SetInteger(arguments, "--topk", config.top_k);
  SetInteger(arguments, "--search_query_count", config.search_query_count);
  SetInteger(arguments, "--num_threads", config.num_threads_building);
  if (Has(arguments, "--num_threads")) {
    config.num_threads_searching = config.num_threads_building;
  }
  SetInteger(arguments, "--num_threads_building", config.num_threads_building);
  SetInteger(arguments, "--num_threads_searching",
             config.num_threads_searching);
  config.use_id_based_recall = Has(arguments, "--id_recall");
  config.enable_recall = !Has(arguments, "--disable_recall");
  config.enable_percent_recall = !Has(arguments, "--disable_percent_recall");
  config.enable_qps = !Has(arguments, "--disable_qps");
  config.enable_tps = !Has(arguments, "--disable_tps");
  config.enable_memory = !Has(arguments, "--disable_memory");
  config.enable_latency = !Has(arguments, "--disable_latency");
  config.enable_percent_latency = !Has(arguments, "--disable_percent_latency");
  SetString(arguments, "--format", suite.output_format);
  SetString(arguments, "--output", suite.output_path);
  suite.append_output = Has(arguments, "--append");
  hypervec::eval::ValidateEvalConfig(config);
  suite.cases.push_back(std::move(config));
  return suite;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ArgumentMap arguments = ParseArguments(argc, argv);
    if (argc == 1 || Has(arguments, "--help")) {
      PrintHelp(argv[0]);
      return 0;
    }
    hypervec::eval::EvalSuiteConfig suite;
    const auto config = arguments.find("--config");
    if (config != arguments.end()) {
      suite = hypervec::eval::LoadEvalConfig(config->second);
      SetString(arguments, "--format", suite.output_format);
      SetString(arguments, "--output", suite.output_path);
      if (Has(arguments, "--append")) {
        suite.append_output = true;
      }
    } else {
      suite = LoadCliConfig(arguments);
    }
    nlohmann::json output;
    output["results"] = nlohmann::json::array();
    for (const auto& eval_case : suite.cases) {
      output["results"].push_back(hypervec::eval::RunEvalCase(eval_case));
    }
    const auto formatter = hypervec::eval::MakeFormatter(suite.output_format);
    const auto exporter =
        hypervec::eval::MakeExporter(suite.output_path, suite.append_output);
    exporter->Export(formatter->Format(output));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hypervec_eval: " << error.what() << '\n';
    return 1;
  }
}
