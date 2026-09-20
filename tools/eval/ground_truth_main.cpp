/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/exact_ground_truth.h>
#include <eval/vector_dataset.h>

#include <charconv>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

struct CommandLine {
  std::string base_path;
  std::string query_path;
  std::string output_path;
  hypervec::idx_t k = 0;
  hypervec::GroundTruthMetric metric = hypervec::GroundTruthMetric::kL2;
  bool has_metric = false;
  bool show_help = false;
};

void PrintUsage(std::ostream& output) {
  output << "Usage: hypervec_ground_truth [options]\n"
         << "Options:\n"
         << "  --base BASE.fvecs       Prepared base vectors\n"
         << "  --queries QUERY.fvecs   Prepared query vectors\n"
         << "  --output GT.ivecs       Exact neighbor IDs\n"
         << "  --k N                   Number of neighbors per query\n"
         << "  --metric METRIC         l2, inner_product, or cosine\n"
         << "                           (cosine inputs must be L2-normalized)\n"
         << "  --help                  Show this message\n";
}

std::string_view RequireValue(int argc, char** argv, int* position,
                              std::string_view option) {
  if (*position + 1 >= argc) {
    throw std::runtime_error("missing value for " + std::string(option));
  }
  ++*position;
  return argv[*position];
}

hypervec::idx_t ParsePositive(std::string_view value, std::string_view option) {
  uint64_t parsed = 0;
  const auto conversion =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (conversion.ec != std::errc() ||
      conversion.ptr != value.data() + value.size() || parsed == 0 ||
      parsed > static_cast<uint64_t>(
                   (std::numeric_limits<hypervec::idx_t>::max)())) {
    throw std::runtime_error(std::string(option) +
                             " must be a positive integer");
  }
  return static_cast<hypervec::idx_t>(parsed);
}

hypervec::GroundTruthMetric ParseMetric(std::string_view value) {
  if (value == "l2") {
    return hypervec::GroundTruthMetric::kL2;
  }
  if (value == "inner_product") {
    return hypervec::GroundTruthMetric::kInnerProduct;
  }
  if (value == "cosine") {
    return hypervec::GroundTruthMetric::kCosine;
  }
  throw std::runtime_error("--metric must be l2, inner_product, or cosine");
}

std::string_view MetricName(hypervec::GroundTruthMetric metric) {
  switch (metric) {
    case hypervec::GroundTruthMetric::kL2:
      return "l2";
    case hypervec::GroundTruthMetric::kInnerProduct:
      return "inner_product";
    case hypervec::GroundTruthMetric::kCosine:
      return "cosine";
  }
  throw std::runtime_error("unknown metric");
}

CommandLine ParseCommandLine(int argc, char** argv) {
  CommandLine command;
  for (int position = 1; position < argc; ++position) {
    const std::string_view argument = argv[position];
    if (argument == "--help") {
      command.show_help = true;
    } else if (argument == "--base") {
      command.base_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--queries") {
      command.query_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--output") {
      command.output_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--k") {
      command.k = ParsePositive(RequireValue(argc, argv, &position, argument),
                                argument);
    } else if (argument == "--metric") {
      command.metric =
          ParseMetric(RequireValue(argc, argv, &position, argument));
      command.has_metric = true;
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  return command;
}

void ValidateCommand(const CommandLine& command) {
  if (command.base_path.empty() || command.query_path.empty() ||
      command.output_path.empty() || command.k <= 0 || !command.has_metric) {
    throw std::runtime_error(
        "--base, --queries, --output, --k, and --metric are required");
  }
  const std::filesystem::path output =
      std::filesystem::absolute(command.output_path).lexically_normal();
  const std::filesystem::path base =
      std::filesystem::absolute(command.base_path).lexically_normal();
  const std::filesystem::path queries =
      std::filesystem::absolute(command.query_path).lexically_normal();
  if (output == base || output == queries) {
    throw std::runtime_error("ground-truth output must differ from its inputs");
  }
}

int Run(const CommandLine& command) {
  ValidateCommand(command);
  const hypervec::FloatVectorDataset base =
      hypervec::ReadFvecsFile(command.base_path);
  const hypervec::FloatVectorDataset queries =
      hypervec::ReadFvecsFile(command.query_path);
  const hypervec::IntegerVectorDataset result =
      hypervec::ComputeExactGroundTruth(base, queries, command.k,
                                        command.metric);
  hypervec::WriteIvecsFile(command.output_path, result);

  std::cout << "base_count=" << base.vector_count << '\n';
  std::cout << "query_count=" << queries.vector_count << '\n';
  std::cout << "dimension=" << base.dimension << '\n';
  std::cout << "k=" << command.k << '\n';
  std::cout << "metric=" << MetricName(command.metric) << '\n';
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
    std::cerr << "hypervec_ground_truth: " << error.what() << '\n';
    PrintUsage(std::cerr);
    return 1;
  }
}
