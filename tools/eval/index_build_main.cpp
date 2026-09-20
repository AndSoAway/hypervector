/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/semantic_metric.h>
#include <index/index_factory.h>
#include <persistence/index_io.h>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "eval/artifact_fingerprint.h"
#include "eval/json_util.h"

namespace {

enum class ParameterType { kInteger, kDouble, kBoolean, kString };

struct ConfigParameter {
  std::string name;
  ParameterType type = ParameterType::kInteger;
  int64_t integer = 0;
  double floating = 0.0;
  bool boolean = false;
  std::string string;
};

struct CommandLine {
  std::string input_path;
  std::string training_query_path;
  std::string output_path;
  std::string json_output_path;
  std::string index_type;
  hypervec::SemanticMetric semantic_metric = hypervec::SemanticMetric::kL2;
  bool has_metric = false;
  std::vector<ConfigParameter> parameters;
  std::vector<std::string> parameter_assignments;
  bool list_indexes = false;
  bool show_help = false;
};

void PrintUsage(std::ostream& output) {
  output
      << "Usage: hypervec_build [options]\n"
      << "Options:\n"
      << "  --input BASE.fvecs              Prepared base vectors\n"
      << "  --training-queries TRAIN.fvecs  Optional representative queries\n"
      << "  --output INDEX                  Persisted index output\n"
      << "  --index-type NAME               Registered index name or alias\n"
      << "  --metric METRIC                 l2, inner_product, or cosine\n"
      << "  --index-param NAME=TYPE:VALUE   Repeatable typed factory option\n"
      << "                                   TYPE: int, double, bool, string\n"
      << "  --json-output REPORT.json       Optional build manifest\n"
      << "  --list-indexes                  List registered indexes and "
         "options\n"
      << "  --help                          Show this message\n";
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

double ParseDouble(std::string_view value, std::string_view context) {
  double parsed = 0.0;
  const auto conversion =
      std::from_chars(value.data(), value.data() + value.size(), parsed,
                      std::chars_format::general);
  if (conversion.ec != std::errc() ||
      conversion.ptr != value.data() + value.size()) {
    throw std::runtime_error(std::string(context) + " must be a double");
  }
  return parsed;
}

ConfigParameter ParseIndexParameter(std::string_view assignment) {
  const size_t equals = assignment.find('=');
  const size_t colon =
      assignment.find(':', equals == std::string_view::npos ? 0 : equals + 1);
  if (equals == std::string_view::npos || equals == 0 ||
      colon == std::string_view::npos || colon == equals + 1) {
    throw std::runtime_error("--index-param must use NAME=TYPE:VALUE format");
  }

  ConfigParameter parameter;
  parameter.name = assignment.substr(0, equals);
  const std::string_view type =
      assignment.substr(equals + 1, colon - equals - 1);
  const std::string_view value = assignment.substr(colon + 1);
  if (type == "int") {
    parameter.type = ParameterType::kInteger;
    parameter.integer = ParseInteger(value, "integer index parameter");
  } else if (type == "double") {
    parameter.type = ParameterType::kDouble;
    parameter.floating = ParseDouble(value, "double index parameter");
  } else if (type == "bool") {
    parameter.type = ParameterType::kBoolean;
    if (value == "true") {
      parameter.boolean = true;
    } else if (value == "false") {
      parameter.boolean = false;
    } else {
      throw std::runtime_error("boolean index parameter must be true or false");
    }
  } else if (type == "string") {
    parameter.type = ParameterType::kString;
    parameter.string = value;
  } else {
    throw std::runtime_error(
        "index parameter type must be int, double, bool, or string");
  }
  return parameter;
}

hypervec::SemanticMetric ParseMetric(std::string_view value) {
  if (value == "l2") {
    return hypervec::SemanticMetric::kL2;
  }
  if (value == "inner_product") {
    return hypervec::SemanticMetric::kInnerProduct;
  }
  if (value == "cosine") {
    return hypervec::SemanticMetric::kCosine;
  }
  throw std::runtime_error("--metric must be l2, inner_product, or cosine");
}

CommandLine ParseCommandLine(int argc, char** argv) {
  CommandLine command;
  for (int position = 1; position < argc; ++position) {
    const std::string_view argument = argv[position];
    if (argument == "--help") {
      command.show_help = true;
    } else if (argument == "--list-indexes") {
      command.list_indexes = true;
    } else if (argument == "--input") {
      command.input_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--training-queries") {
      command.training_query_path =
          RequireValue(argc, argv, &position, argument);
    } else if (argument == "--output") {
      command.output_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--index-type") {
      command.index_type = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--json-output") {
      command.json_output_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--metric") {
      command.semantic_metric =
          ParseMetric(RequireValue(argc, argv, &position, argument));
      command.has_metric = true;
    } else if (argument == "--index-param") {
      const std::string_view assignment =
          RequireValue(argc, argv, &position, argument);
      command.parameters.push_back(ParseIndexParameter(assignment));
      command.parameter_assignments.emplace_back(assignment);
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
  if (command.input_path.empty() || command.output_path.empty() ||
      command.index_type.empty() || !command.has_metric) {
    throw std::runtime_error(
        "--input, --output, --index-type, and --metric are required");
  }
  const std::filesystem::path output = NormalizedPath(command.output_path);
  if (output == NormalizedPath(command.input_path) ||
      (!command.training_query_path.empty() &&
       output == NormalizedPath(command.training_query_path))) {
    throw std::runtime_error("index output must differ from dataset inputs");
  }
  if (!command.json_output_path.empty()) {
    const std::filesystem::path report =
        NormalizedPath(command.json_output_path);
    if (report == output || report == NormalizedPath(command.input_path) ||
        (!command.training_query_path.empty() &&
         report == NormalizedPath(command.training_query_path))) {
      throw std::runtime_error(
          "build manifest must differ from index and dataset paths");
    }
  }
  std::unordered_set<std::string> names;
  for (const ConfigParameter& parameter : command.parameters) {
    if (!names.insert(parameter.name).second) {
      throw std::runtime_error("duplicate index parameter: " + parameter.name);
    }
  }
}

void ApplyParameter(const ConfigParameter& parameter,
                    hypervec::IndexConfig* config) {
  switch (parameter.type) {
    case ParameterType::kInteger:
      config->SetInteger(parameter.name, parameter.integer);
      break;
    case ParameterType::kDouble:
      config->SetDouble(parameter.name, parameter.floating);
      break;
    case ParameterType::kBoolean:
      config->SetBoolean(parameter.name, parameter.boolean);
      break;
    case ParameterType::kString:
      config->SetString(parameter.name, parameter.string);
      break;
  }
}

void PrintList(std::ostream& output) {
  for (const hypervec::IndexDescriptor& descriptor :
       hypervec::GetIndexRegistry().List()) {
    output << descriptor.name;
    if (!descriptor.aliases.empty()) {
      output << " aliases=";
      for (size_t offset = 0; offset < descriptor.aliases.size(); ++offset) {
        output << (offset == 0 ? "" : ",") << descriptor.aliases[offset];
      }
    }
    output << " parameters=";
    if (descriptor.parameter_names.empty()) {
      output << "none";
    } else {
      for (size_t offset = 0; offset < descriptor.parameter_names.size();
           ++offset) {
        output << (offset == 0 ? "" : ",")
               << descriptor.parameter_names[offset];
      }
    }
    output << '\n';
  }
}

std::string_view IndexMetricName(hypervec::MetricType metric) {
  if (metric == hypervec::kMetricL2) {
    return "l2";
  }
  if (metric == hypervec::kMetricInnerProduct) {
    return "inner_product";
  }
  throw std::runtime_error("unsupported build index metric");
}

void WriteJsonManifest(const CommandLine& command, const hypervec::Index& index,
                       const hypervec::FloatVectorDataset& base,
                       hypervec::idx_t training_query_count,
                       double build_seconds, double write_seconds) {
  if (command.json_output_path.empty()) {
    return;
  }
  const hypervec::eval_cli::ArtifactFingerprint base_fingerprint =
      hypervec::eval_cli::FingerprintFile(command.input_path);
  const hypervec::eval_cli::ArtifactFingerprint index_fingerprint =
      hypervec::eval_cli::FingerprintFile(command.output_path);
  hypervec::eval_cli::ArtifactFingerprint training_fingerprint;
  if (!command.training_query_path.empty()) {
    training_fingerprint =
        hypervec::eval_cli::FingerprintFile(command.training_query_path);
  }
  std::ofstream output(command.json_output_path,
                       std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("cannot open build manifest output: " +
                             command.json_output_path);
  }
  output << std::setprecision(17);
  output << "{\n"
         << "  \"format\": \"hypervec-build-report-v1\",\n"
         << "  \"library_version\": \"" << VERSION_STRING << "\",\n"
         << "  \"dataset\": {\n"
         << "    \"base\": \""
         << hypervec::eval_cli::JsonEscape(
                NormalizedPath(command.input_path).generic_string())
         << "\",\n"
         << "    \"training_queries\": ";
  if (command.training_query_path.empty()) {
    output << "null";
  } else {
    output << "\""
           << hypervec::eval_cli::JsonEscape(
                  NormalizedPath(command.training_query_path).generic_string())
           << "\"";
  }
  output << ",\n"
         << "    \"semantic_metric\": \""
         << hypervec::SemanticMetricName(command.semantic_metric) << "\",\n"
         << "    \"dimension\": " << base.dimension << ",\n"
         << "    \"vector_count\": " << base.vector_count << ",\n"
         << "    \"training_query_count\": " << training_query_count << "\n"
         << "  },\n"
         << "  \"index\": {\n"
         << "    \"path\": \""
         << hypervec::eval_cli::JsonEscape(
                NormalizedPath(command.output_path).generic_string())
         << "\",\n"
         << "    \"requested_type\": \""
         << hypervec::eval_cli::JsonEscape(command.index_type) << "\",\n"
         << "    \"metric\": \"" << IndexMetricName(index.metric_type)
         << "\",\n"
         << "    \"metric_type\": " << static_cast<int>(index.metric_type)
         << ",\n"
         << "    \"requested_parameters\": [";
  for (size_t offset = 0; offset < command.parameter_assignments.size();
       ++offset) {
    output << (offset == 0 ? "" : ", ") << "\""
           << hypervec::eval_cli::JsonEscape(
                  command.parameter_assignments[offset])
           << "\"";
  }
  output << "]\n"
         << "  },\n"
         << "  \"artifacts\": {\n"
         << "    \"base\": ";
  hypervec::eval_cli::WriteArtifactJson(output, command.input_path,
                                        base_fingerprint, "    ");
  output << ",\n"
         << "    \"training_queries\": ";
  if (command.training_query_path.empty()) {
    output << "null";
  } else {
    hypervec::eval_cli::WriteArtifactJson(output, command.training_query_path,
                                          training_fingerprint, "    ");
  }
  output << ",\n"
         << "    \"index\": ";
  hypervec::eval_cli::WriteArtifactJson(output, command.output_path,
                                        index_fingerprint, "    ");
  output << "\n"
         << "  },\n"
         << "  \"timing\": {\n"
         << "    \"build_seconds\": " << build_seconds << ",\n"
         << "    \"write_seconds\": " << write_seconds << "\n"
         << "  }\n"
         << "}\n";
  output.close();
  if (output.fail()) {
    throw std::runtime_error("cannot write build manifest output: " +
                             command.json_output_path);
  }
}

int Run(const CommandLine& command) {
  ValidateCommand(command);
  const hypervec::FloatVectorDataset base =
      hypervec::ReadFvecsFile(command.input_path);
  hypervec::ValidateSemanticMetricDataset(base, command.semantic_metric);

  hypervec::IndexConfig config(
      command.index_type, base.dimension,
      hypervec::IndexMetricForSemanticMetric(command.semantic_metric));
  for (const ConfigParameter& parameter : command.parameters) {
    ApplyParameter(parameter, &config);
  }
  std::unique_ptr<hypervec::Index> index = hypervec::CreateIndex(config);

  hypervec::FloatVectorDataset training_queries;
  if (!command.training_query_path.empty()) {
    training_queries = hypervec::ReadFvecsFile(command.training_query_path);
    hypervec::ValidateSemanticMetricDataset(training_queries,
                                            command.semantic_metric);
    if (training_queries.dimension != base.dimension) {
      throw std::runtime_error(
          "training-query dimension does not match base dimension");
    }
  }

  const auto build_start = std::chrono::steady_clock::now();
  if (training_queries.vector_count == 0) {
    index->Build(base.vector_count, base.values.data());
  } else {
    index->Build(base.vector_count, base.values.data(),
                 training_queries.vector_count, training_queries.values.data());
  }
  const auto build_end = std::chrono::steady_clock::now();
  if (index->n_total != base.vector_count) {
    throw std::runtime_error("built index vector count does not match input");
  }

  const auto write_start = std::chrono::steady_clock::now();
  hypervec::WriteIndex(index.get(), command.output_path.c_str());
  const auto write_end = std::chrono::steady_clock::now();

  const double build_seconds =
      std::chrono::duration<double>(build_end - build_start).count();
  const double write_seconds =
      std::chrono::duration<double>(write_end - write_start).count();
  WriteJsonManifest(command, *index, base, training_queries.vector_count,
                    build_seconds, write_seconds);
  std::cout << std::setprecision(10);
  std::cout << "index_type=" << command.index_type << '\n';
  std::cout << "metric="
            << hypervec::SemanticMetricName(command.semantic_metric) << '\n';
  std::cout << "index_metric="
            << (index->metric_type == hypervec::kMetricL2 ? "l2"
                                                          : "inner_product")
            << '\n';
  std::cout << "dimension=" << base.dimension << '\n';
  std::cout << "vector_count=" << base.vector_count << '\n';
  std::cout << "training_query_count=" << training_queries.vector_count << '\n';
  std::cout << "build_seconds=" << build_seconds << '\n';
  std::cout << "write_seconds=" << write_seconds << '\n';
  if (!command.json_output_path.empty()) {
    std::cout << "json_output=" << command.json_output_path << '\n';
  }
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
    if (command.list_indexes) {
      PrintList(std::cout);
      return 0;
    }
    return Run(command);
  } catch (const std::exception& error) {
    std::cerr << "hypervec_build: " << error.what() << '\n';
    PrintUsage(std::cerr);
    return 1;
  }
}
