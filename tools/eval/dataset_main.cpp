/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/semantic_metric.h>

#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "eval/json_util.h"

namespace {

using hypervec::SemanticMetric;

struct CommandLine {
  std::string name;
  std::string input_path;
  std::string base_output_path;
  std::string query_output_path;
  std::string metadata_output_path;
  hypervec::idx_t query_count = 0;
  uint64_t seed = 1234;
  SemanticMetric semantic_metric = SemanticMetric::kL2;
  bool has_semantic_metric = false;
  bool show_help = false;
};

void PrintUsage(std::ostream& output) {
  output << "Usage: hypervec_dataset prepare [options]\n"
         << "Options:\n"
         << "  --name NAME                 Stable dataset name\n"
         << "  --input SOURCE.fvecs        Source vectors\n"
         << "  --base-output BASE.fvecs    Output base vectors\n"
         << "  --query-output QUERY.fvecs  Output query vectors\n"
         << "  --metadata-output META.json Output preparation metadata\n"
         << "  --query-count N             Number of held-out query vectors\n"
         << "  --seed N                    Split seed (default: 1234)\n"
         << "  --semantic-metric METRIC    l2, inner_product, or cosine\n"
         << "  --help                      Show this message\n";
}

std::string_view RequireValue(int argc, char** argv, int* position,
                              std::string_view option) {
  if (*position + 1 >= argc) {
    throw std::runtime_error("missing value for " + std::string(option));
  }
  ++*position;
  return argv[*position];
}

uint64_t ParseUnsigned(std::string_view value, std::string_view context) {
  uint64_t parsed = 0;
  const auto conversion =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (conversion.ec != std::errc() ||
      conversion.ptr != value.data() + value.size()) {
    throw std::runtime_error(std::string(context) +
                             " must be a non-negative integer");
  }
  return parsed;
}

SemanticMetric ParseMetric(std::string_view value) {
  if (value == "l2") {
    return SemanticMetric::kL2;
  }
  if (value == "inner_product") {
    return SemanticMetric::kInnerProduct;
  }
  if (value == "cosine") {
    return SemanticMetric::kCosine;
  }
  throw std::runtime_error(
      "--semantic-metric must be l2, inner_product, or cosine");
}

CommandLine ParseCommandLine(int argc, char** argv) {
  CommandLine command;
  if (argc > 1 && std::string_view(argv[1]) != "prepare" &&
      std::string_view(argv[1]) != "--help") {
    throw std::runtime_error("expected the 'prepare' command");
  }
  for (int position = 1; position < argc; ++position) {
    const std::string_view argument = argv[position];
    if (argument == "prepare") {
      continue;
    }
    if (argument == "--help") {
      command.show_help = true;
    } else if (argument == "--name") {
      command.name = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--input") {
      command.input_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--base-output") {
      command.base_output_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--query-output") {
      command.query_output_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--metadata-output") {
      command.metadata_output_path =
          RequireValue(argc, argv, &position, argument);
    } else if (argument == "--query-count") {
      const uint64_t value = ParseUnsigned(
          RequireValue(argc, argv, &position, argument), argument);
      if (value > static_cast<uint64_t>(
                      (std::numeric_limits<hypervec::idx_t>::max)())) {
        throw std::runtime_error("--query-count exceeds the supported range");
      }
      command.query_count = static_cast<hypervec::idx_t>(value);
    } else if (argument == "--seed") {
      command.seed = ParseUnsigned(
          RequireValue(argc, argv, &position, argument), argument);
    } else if (argument == "--semantic-metric") {
      command.semantic_metric =
          ParseMetric(RequireValue(argc, argv, &position, argument));
      command.has_semantic_metric = true;
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  return command;
}

std::filesystem::path NormalizedPath(const std::string& path) {
  return std::filesystem::absolute(path).lexically_normal();
}

void ValidateCommand(const CommandLine& command) {
  if (command.name.empty() || command.input_path.empty() ||
      command.base_output_path.empty() || command.query_output_path.empty() ||
      command.metadata_output_path.empty() || command.query_count <= 0 ||
      !command.has_semantic_metric) {
    throw std::runtime_error(
        "--name, --input, --base-output, --query-output, --metadata-output, "
        "--query-count, and --semantic-metric are required");
  }
  const std::filesystem::path input = NormalizedPath(command.input_path);
  const std::filesystem::path base = NormalizedPath(command.base_output_path);
  const std::filesystem::path query = NormalizedPath(command.query_output_path);
  const std::filesystem::path metadata =
      NormalizedPath(command.metadata_output_path);
  if (input == base || input == query || input == metadata || base == query ||
      base == metadata || query == metadata) {
    throw std::runtime_error("input and output paths must be distinct");
  }
}

void WriteMetadata(const CommandLine& command,
                   const hypervec::FloatVectorDataset& source,
                   const hypervec::DatasetRowSplit& split) {
  std::ofstream output(command.metadata_output_path,
                       std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("cannot open metadata output: " +
                             command.metadata_output_path);
  }
  output << "{\n"
         << "  \"format\": \"hypervec-eval-dataset-v1\",\n"
         << "  \"name\": \"" << hypervec::eval_cli::JsonEscape(command.name)
         << "\",\n"
         << "  \"source_format\": \"fvecs\",\n"
         << "  \"dtype\": \"float32\",\n"
         << "  \"source\": \""
         << hypervec::eval_cli::JsonEscape(command.input_path) << "\",\n"
         << "  \"base\": \""
         << hypervec::eval_cli::JsonEscape(command.base_output_path) << "\",\n"
         << "  \"queries\": \""
         << hypervec::eval_cli::JsonEscape(command.query_output_path) << "\",\n"
         << "  \"dimension\": " << source.dimension << ",\n"
         << "  \"source_count\": " << source.vector_count << ",\n"
         << "  \"base_count\": " << split.base_rows.size() << ",\n"
         << "  \"query_count\": " << split.query_rows.size() << ",\n"
         << "  \"semantic_metric\": \""
         << hypervec::SemanticMetricName(command.semantic_metric) << "\",\n"
         << "  \"l2_normalized\": "
         << (command.semantic_metric == SemanticMetric::kCosine ? "true"
                                                                : "false")
         << ",\n"
         << "  \"split_algorithm\": "
            "\"mt19937_64-partial-fisher-yates-v1\",\n"
         << "  \"split_seed\": " << command.seed << "\n"
         << "}\n";
  output.close();
  if (output.fail()) {
    throw std::runtime_error("cannot write metadata output: " +
                             command.metadata_output_path);
  }
}

int Run(const CommandLine& command) {
  ValidateCommand(command);
  hypervec::FloatVectorDataset source =
      hypervec::ReadFvecsFile(command.input_path);
  hypervec::ValidateFloatVectorDataset(source);
  if (command.semantic_metric == SemanticMetric::kCosine) {
    hypervec::NormalizeL2(&source);
  }
  const hypervec::DatasetRowSplit split = hypervec::MakeDatasetRowSplit(
      source.vector_count, command.query_count, command.seed);
  hypervec::WriteFvecsRowsFile(command.base_output_path, source,
                               split.base_rows);
  hypervec::WriteFvecsRowsFile(command.query_output_path, source,
                               split.query_rows);
  WriteMetadata(command, source, split);

  std::cout << "dataset=" << command.name << '\n';
  std::cout << "dimension=" << source.dimension << '\n';
  std::cout << "base_count=" << split.base_rows.size() << '\n';
  std::cout << "query_count=" << split.query_rows.size() << '\n';
  std::cout << "semantic_metric="
            << hypervec::SemanticMetricName(command.semantic_metric) << '\n';
  std::cout << "l2_normalized="
            << (command.semantic_metric == SemanticMetric::kCosine ? "true"
                                                                   : "false")
            << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CommandLine command = ParseCommandLine(argc, argv);
    if (command.show_help) {
      PrintUsage(std::cout);
      return 0;
    }
    return Run(command);
  } catch (const std::exception& error) {
    std::cerr << "hypervec_dataset: " << error.what() << '\n';
    PrintUsage(std::cerr);
    return 1;
  }
}
