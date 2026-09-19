/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/search_evaluator.h>
#include <eval/vector_dataset.h>
#include <index/search_parameters_factory.h>
#include <persistence/index_io.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

struct CommandLine {
  std::string index_path;
  std::string query_path;
  std::string ground_truth_path;
  hypervec::idx_t k = 10;
  size_t warmup_runs = 1;
  size_t measured_runs = 3;
  hypervec::SearchConfig search_config;
  bool show_help = false;
};

void PrintUsage(std::ostream& output) {
  output
      << "Usage: hypervec_eval --index INDEX --queries QUERY.fvecs "
         "--ground-truth GT.ivecs [options]\n"
      << "Options:\n"
      << "  --k N                    Recall/search depth (default: 10)\n"
      << "  --warmup-runs N          Untimed full-query runs (default: 1)\n"
      << "  --measured-runs N        Timed full-query runs (default: 3)\n"
      << "  --search-param NAME=VALUE  Repeatable integer/bool runtime option\n"
      << "  --help                   Show this message\n";
}

std::string_view RequireValue(int argc, char** argv, int* position,
                              std::string_view option) {
  if (*position + 1 >= argc) {
    throw std::runtime_error("missing value for " + std::string(option));
  }
  ++*position;
  return argv[*position];
}

int64_t ParseInteger(std::string_view value, std::string_view context) {
  int64_t parsed = 0;
  const auto conversion =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (conversion.ec != std::errc() ||
      conversion.ptr != value.data() + value.size()) {
    throw std::runtime_error(std::string(context) + " must be an integer");
  }
  return parsed;
}

size_t ParseRunCount(std::string_view value, std::string_view context,
                     bool allow_zero) {
  const int64_t parsed = ParseInteger(value, context);
  if (parsed < 0 || (!allow_zero && parsed == 0) ||
      static_cast<uint64_t>(parsed) > (std::numeric_limits<size_t>::max)()) {
    throw std::runtime_error(
        std::string(context) +
        (allow_zero ? " must be non-negative" : " must be positive"));
  }
  return static_cast<size_t>(parsed);
}

void ParseSearchParameter(std::string_view assignment,
                          hypervec::SearchConfig* config) {
  const size_t separator = assignment.find('=');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 == assignment.size()) {
    throw std::runtime_error("--search-param must use the form NAME=VALUE");
  }
  const std::string name(assignment.substr(0, separator));
  const std::string_view value = assignment.substr(separator + 1);
  if (value == "true") {
    config->SetBoolean(name, true);
  } else if (value == "false") {
    config->SetBoolean(name, false);
  } else {
    config->SetInteger(name, ParseInteger(value, "search parameter"));
  }
}

CommandLine ParseCommandLine(int argc, char** argv) {
  CommandLine command;
  for (int position = 1; position < argc; ++position) {
    const std::string_view argument = argv[position];
    if (argument == "--help") {
      command.show_help = true;
    } else if (argument == "--index") {
      command.index_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--queries") {
      command.query_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--ground-truth") {
      command.ground_truth_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--k") {
      const int64_t value =
          ParseInteger(RequireValue(argc, argv, &position, argument), "--k");
      if (value <= 0) {
        throw std::runtime_error("--k must be positive");
      }
      command.k = static_cast<hypervec::idx_t>(value);
    } else if (argument == "--warmup-runs") {
      command.warmup_runs = ParseRunCount(
          RequireValue(argc, argv, &position, argument), argument, true);
    } else if (argument == "--measured-runs") {
      command.measured_runs = ParseRunCount(
          RequireValue(argc, argv, &position, argument), argument, false);
    } else if (argument == "--search-param") {
      ParseSearchParameter(RequireValue(argc, argv, &position, argument),
                           &command.search_config);
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  return command;
}

void ValidateRequiredPaths(const CommandLine& command) {
  if (command.index_path.empty() || command.query_path.empty() ||
      command.ground_truth_path.empty()) {
    throw std::runtime_error(
        "--index, --queries, and --ground-truth are required");
  }
}

int Run(const CommandLine& command) {
  ValidateRequiredPaths(command);
  std::unique_ptr<hypervec::Index> index =
      hypervec::ReadIndexUp(command.index_path.c_str());
  const hypervec::FloatVectorDataset queries =
      hypervec::ReadFvecsFile(command.query_path);
  const hypervec::IntegerVectorDataset ground_truth =
      hypervec::ReadIvecsFile(command.ground_truth_path);
  if (queries.dimension != index->d) {
    throw std::runtime_error("query dimension does not match index dimension");
  }
  if (queries.vector_count != ground_truth.vector_count) {
    throw std::runtime_error("query and ground-truth vector counts must match");
  }

  std::unique_ptr<hypervec::SearchParameters> parameters =
      hypervec::CreateSearchParameters(*index, command.search_config);
  const hypervec::SearchEvaluationInput input{
      queries.values.data(),
      queries.vector_count,
      {ground_truth.values.data(), ground_truth.vector_count,
       ground_truth.dimension}};
  hypervec::SearchEvaluationOptions options;
  options.k = command.k;
  options.warmup_runs = command.warmup_runs;
  options.measured_runs = command.measured_runs;
  const hypervec::SearchEvaluationResult result =
      hypervec::EvaluateSearch(*index, input, options, parameters.get());
  const hypervec::SearchParameterDescriptor descriptor =
      hypervec::DescribeSearchParameters(*index);

  std::cout << std::setprecision(10);
  std::cout << "index_family=" << descriptor.name << '\n';
  std::cout << "query_count=" << result.query_count << '\n';
  std::cout << "k=" << result.k << '\n';
  std::cout << "measured_runs=" << result.measured_runs << '\n';
  std::cout << "recall_at_k=" << result.recall_at_k << '\n';
  std::cout << "elapsed_seconds=" << result.elapsed_seconds << '\n';
  std::cout << "mean_latency_ms=" << result.mean_latency_ms << '\n';
  std::cout << "queries_per_second=" << result.queries_per_second << '\n';
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
    std::cerr << "hypervec_eval: " << error.what() << '\n';
    PrintUsage(std::cerr);
    return 1;
  }
}
