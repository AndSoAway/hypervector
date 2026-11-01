/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/search_evaluator.h>
#include <eval/semantic_metric.h>
#include <index/refine/index_rerank.h>
#include <index/search_parameters_factory.h>
#include <persistence/index_io.h>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "eval/artifact_fingerprint.h"
#include "eval/json_util.h"
#include "eval/process_memory.h"

namespace {

struct CommandLine {
  std::string index_path;
  std::string query_path;
  std::string ground_truth_path;
  std::string rerank_base_path;
  hypervec::idx_t rerank_candidates = 0;
  std::string json_output_path;
  hypervec::SemanticMetric semantic_metric = hypervec::SemanticMetric::kL2;
  bool has_metric = false;
  hypervec::idx_t k = 10;
  size_t warmup_runs = 1;
  size_t measured_runs = 3;
  hypervec::idx_t query_batch_size = 0;
  size_t concurrency = 1;
  hypervec::SearchConfig search_config;
  std::vector<std::string> search_parameters;
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
      << "  --query-batch-size N     Queries per timed Search call\n"
      << "                           (default: all queries; 1 if concurrent)\n"
      << "  --concurrency N        Concurrent independent Search callers "
         "(default: 1)\n"
      << "  --metric METRIC          l2, inner_product, or cosine\n"
      << "  --search-param NAME=VALUE  Repeatable integer/bool runtime option\n"
      << "  --rerank-base BASE.fvecs  Exact vectors, in index label order\n"
      << "  --rerank-candidates N    Approximate candidates before exact "
         "top-k\n"
      << "  --json-output REPORT.json  Optional reproducible result report\n"
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
    } else if (argument == "--rerank-base") {
      command.rerank_base_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--rerank-candidates") {
      const int64_t count =
          ParseInteger(RequireValue(argc, argv, &position, argument), argument);
      if (count <= 0 || count > (std::numeric_limits<hypervec::idx_t>::max)()) {
        throw std::runtime_error("--rerank-candidates must be positive");
      }
      command.rerank_candidates = count;
    } else if (argument == "--json-output") {
      command.json_output_path = RequireValue(argc, argv, &position, argument);
    } else if (argument == "--metric") {
      command.semantic_metric =
          ParseMetric(RequireValue(argc, argv, &position, argument));
      command.has_metric = true;
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
    } else if (argument == "--query-batch-size") {
      const int64_t value =
          ParseInteger(RequireValue(argc, argv, &position, argument), argument);
      if (value <= 0 || static_cast<uint64_t>(value) >
                            static_cast<uint64_t>((
                                std::numeric_limits<hypervec::idx_t>::max)())) {
        throw std::runtime_error("--query-batch-size must be positive");
      }
      command.query_batch_size = static_cast<hypervec::idx_t>(value);
    } else if (argument == "--concurrency") {
      command.concurrency = ParseRunCount(
          RequireValue(argc, argv, &position, argument), argument, false);
    } else if (argument == "--search-param") {
      const std::string_view assignment =
          RequireValue(argc, argv, &position, argument);
      ParseSearchParameter(assignment, &command.search_config);
      command.search_parameters.emplace_back(assignment);
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
  if (command.index_path.empty() || command.query_path.empty() ||
      command.ground_truth_path.empty() || !command.has_metric) {
    throw std::runtime_error(
        "--index, --queries, --ground-truth, and --metric are required");
  }
  if (command.rerank_base_path.empty() != (command.rerank_candidates == 0) ||
      (command.rerank_candidates > 0 &&
       command.rerank_candidates < command.k)) {
    throw std::runtime_error(
        "rerank requires both --rerank-base and --rerank-candidates >= k");
  }
  if (!command.json_output_path.empty()) {
    const std::filesystem::path output =
        NormalizedPath(command.json_output_path);
    if (output == NormalizedPath(command.index_path) ||
        output == NormalizedPath(command.query_path) ||
        output == NormalizedPath(command.ground_truth_path) ||
        (!command.rerank_base_path.empty() &&
         output == NormalizedPath(command.rerank_base_path))) {
      throw std::runtime_error(
          "JSON output must differ from evaluation inputs");
    }
  }
}

std::string_view MetricName(hypervec::MetricType metric) {
  switch (metric) {
    case hypervec::kMetricInnerProduct:
      return "inner_product";
    case hypervec::kMetricL2:
      return "l2";
    case hypervec::kMetricL1:
      return "l1";
    case hypervec::kMetricLinf:
      return "linf";
    case hypervec::kMetricLp:
      return "lp";
    case hypervec::kMetricCanberra:
      return "canberra";
    case hypervec::kMetricBrayCurtis:
      return "bray_curtis";
    case hypervec::kMetricJensenShannon:
      return "jensen_shannon";
    case hypervec::kMetricJaccard:
      return "jaccard";
    case hypervec::kMetricNaNEuclidean:
      return "nan_euclidean";
    case hypervec::kMetricGower:
      return "gower";
  }
  throw std::runtime_error("unknown index metric");
}

void WriteJsonNumber(std::ostream& output, double value) {
  if (std::isfinite(value)) {
    output << value;
  } else {
    output << "null";
  }
}

void WriteJsonReport(const CommandLine& command, const hypervec::Index& index,
                     const hypervec::SearchParameterDescriptor& descriptor,
                     const hypervec::IntegerVectorDataset& ground_truth,
                     const hypervec::SearchEvaluationResult& result,
                     const std::optional<uint64_t>& peak_rss_bytes) {
  if (command.json_output_path.empty()) {
    return;
  }
  const hypervec::eval_cli::ArtifactFingerprint index_fingerprint =
      hypervec::eval_cli::FingerprintFile(command.index_path);
  const hypervec::eval_cli::ArtifactFingerprint query_fingerprint =
      hypervec::eval_cli::FingerprintFile(command.query_path);
  const hypervec::eval_cli::ArtifactFingerprint ground_truth_fingerprint =
      hypervec::eval_cli::FingerprintFile(command.ground_truth_path);
  const std::optional<hypervec::eval_cli::ArtifactFingerprint>
      rerank_fingerprint =
          command.rerank_base_path.empty()
              ? std::nullopt
              : std::make_optional(hypervec::eval_cli::FingerprintFile(
                    command.rerank_base_path));
  std::ofstream output(command.json_output_path,
                       std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("cannot open JSON report output: " +
                             command.json_output_path);
  }
  output << std::setprecision(17);
  output << "{\n"
         << "  \"format\": \"hypervec-eval-report-v1\",\n"
         << "  \"library_version\": \"" << VERSION_STRING << "\",\n"
         << "  \"index\": {\n"
         << "    \"path\": \""
         << hypervec::eval_cli::JsonEscape(
                NormalizedPath(command.index_path).generic_string())
         << "\",\n"
         << "    \"family\": \""
         << hypervec::eval_cli::JsonEscape(descriptor.name) << "\",\n"
         << "    \"dimension\": " << index.d << ",\n"
         << "    \"vector_count\": " << index.n_total << ",\n"
         << "    \"metric\": \"" << MetricName(index.metric_type) << "\",\n"
         << "    \"metric_type\": " << static_cast<int>(index.metric_type)
         << ",\n"
         << "    \"metric_arg\": ";
  WriteJsonNumber(output, index.metric_arg);
  output << "\n"
         << "  },\n"
         << "  \"rerank\": {\n"
         << "    \"enabled\": "
         << (rerank_fingerprint.has_value() ? "true" : "false") << ",\n"
         << "    \"candidates\": " << command.rerank_candidates << ",\n"
         << "    \"exact_base\": ";
  if (rerank_fingerprint.has_value()) {
    hypervec::eval_cli::WriteArtifactJson(output, command.rerank_base_path,
                                          *rerank_fingerprint, "    ");
  } else {
    output << "null";
  }
  output << "\n  },\n"
         << "  \"workload\": {\n"
         << "    \"queries\": \""
         << hypervec::eval_cli::JsonEscape(
                NormalizedPath(command.query_path).generic_string())
         << "\",\n"
         << "    \"ground_truth\": \""
         << hypervec::eval_cli::JsonEscape(
                NormalizedPath(command.ground_truth_path).generic_string())
         << "\",\n"
         << "    \"query_count\": " << result.query_count << ",\n"
         << "    \"ground_truth_width\": " << ground_truth.dimension << ",\n"
         << "    \"semantic_metric\": \""
         << hypervec::SemanticMetricName(command.semantic_metric) << "\",\n"
         << "    \"k\": " << result.k << "\n"
         << "  },\n"
         << "  \"execution\": {\n"
         << "    \"warmup_runs\": " << command.warmup_runs << ",\n"
         << "    \"measured_runs\": " << result.measured_runs << ",\n"
         << "    \"search_parameters\": [";
  for (size_t offset = 0; offset < command.search_parameters.size(); ++offset) {
    output << (offset == 0 ? "" : ", ") << "\""
           << hypervec::eval_cli::JsonEscape(command.search_parameters[offset])
           << "\"";
  }
  output << "],\n"
         << "    \"concurrency\": " << result.concurrency << "\n"
         << "  },\n"
         << "  \"artifacts\": {\n"
         << "    \"index\": ";
  hypervec::eval_cli::WriteArtifactJson(output, command.index_path,
                                        index_fingerprint, "    ");
  output << ",\n"
         << "    \"queries\": ";
  hypervec::eval_cli::WriteArtifactJson(output, command.query_path,
                                        query_fingerprint, "    ");
  output << ",\n"
         << "    \"ground_truth\": ";
  hypervec::eval_cli::WriteArtifactJson(output, command.ground_truth_path,
                                        ground_truth_fingerprint, "    ");
  output << "\n"
         << "  },\n"
         << "  \"resources\": {\n"
         << "    \"memory_scope\": \"process\",\n"
         << "    \"peak_rss_bytes\": ";
  if (peak_rss_bytes.has_value()) {
    output << *peak_rss_bytes;
  } else {
    output << "null";
  }
  output << ",\n"
         << "    \"peak_rss_source\": \""
         << hypervec::eval_cli::PeakResidentSetSource() << "\",\n"
         << "    \"measurement_point\": \"after_search\"\n"
         << "  },\n"
         << "  \"metrics\": {\n"
         << "    \"recall_at_k\": ";
  WriteJsonNumber(output, result.recall_at_k);
  output << ",\n    \"elapsed_seconds\": ";
  WriteJsonNumber(output, result.elapsed_seconds);
  output << ",\n    \"mean_latency_ms\": ";
  WriteJsonNumber(output, result.mean_latency_ms);
  output << ",\n    \"queries_per_second\": ";
  WriteJsonNumber(output, result.queries_per_second);
  output << ",\n    \"batch_latency_ms\": {\n"
         << "      \"query_batch_size\": " << result.query_batch_size << ",\n"
         << "      \"sample_count\": " << result.latency_sample_count << ",\n"
         << "      \"percentile_method\": \"nearest-rank\",\n"
         << "      \"p50\": ";
  WriteJsonNumber(output, result.batch_latency_p50_ms);
  output << ",\n      \"p95\": ";
  WriteJsonNumber(output, result.batch_latency_p95_ms);
  output << ",\n      \"p99\": ";
  WriteJsonNumber(output, result.batch_latency_p99_ms);
  output << "\n    }";
  output << "\n  }\n}\n";
  output.close();
  if (output.fail()) {
    throw std::runtime_error("cannot write JSON report output: " +
                             command.json_output_path);
  }
}

int Run(const CommandLine& command) {
  ValidateCommand(command);
  std::unique_ptr<hypervec::Index> index =
      hypervec::ReadIndexUp(command.index_path.c_str());
  const hypervec::FloatVectorDataset queries =
      hypervec::ReadFvecsFile(command.query_path);
  const hypervec::IntegerVectorDataset ground_truth =
      hypervec::ReadIvecsFile(command.ground_truth_path);
  hypervec::ValidateSemanticMetricDataset(queries, command.semantic_metric);
  hypervec::ValidateSemanticMetricIndex(index->metric_type,
                                        command.semantic_metric);
  if (queries.dimension != index->d) {
    throw std::runtime_error("query dimension does not match index dimension");
  }
  if (queries.vector_count != ground_truth.vector_count) {
    throw std::runtime_error("query and ground-truth vector counts must match");
  }

  std::unique_ptr<hypervec::SearchParameters> parameters =
      hypervec::CreateSearchParameters(*index, command.search_config);
  hypervec::FloatVectorDataset rerank_base;
  std::unique_ptr<hypervec::IndexRerank> reranked;
  const hypervec::Index* search_index = index.get();
  if (!command.rerank_base_path.empty()) {
    rerank_base = hypervec::ReadFvecsFile(command.rerank_base_path);
    if (rerank_base.dimension != index->d ||
        rerank_base.vector_count != index->n_total) {
      throw std::runtime_error("exact rerank base shape does not match index");
    }
    reranked = std::make_unique<hypervec::IndexRerank>(
        *index, rerank_base.values.data(), rerank_base.vector_count,
        command.rerank_candidates);
    search_index = reranked.get();
  }
  const hypervec::SearchEvaluationInput input{
      queries.values.data(),
      queries.vector_count,
      {ground_truth.values.data(), ground_truth.vector_count,
       ground_truth.dimension}};
  hypervec::SearchEvaluationOptions options;
  options.k = command.k;
  options.warmup_runs = command.warmup_runs;
  options.measured_runs = command.measured_runs;
  options.query_batch_size = command.query_batch_size;
  options.concurrency = command.concurrency;
  const hypervec::SearchEvaluationResult result =
      hypervec::EvaluateSearch(*search_index, input, options, parameters.get());
  const hypervec::SearchParameterDescriptor descriptor =
      hypervec::DescribeSearchParameters(*index);
  const std::optional<uint64_t> peak_rss_bytes =
      hypervec::eval_cli::PeakResidentSetBytes();
  WriteJsonReport(command, *index, descriptor, ground_truth, result,
                  peak_rss_bytes);

  std::cout << std::setprecision(10);
  std::cout << "index_family=" << descriptor.name << '\n';
  std::cout << "metric="
            << hypervec::SemanticMetricName(command.semantic_metric) << '\n';
  std::cout << "query_count=" << result.query_count << '\n';
  std::cout << "k=" << result.k << '\n';
  std::cout << "rerank_candidates=" << command.rerank_candidates << '\n';
  std::cout << "measured_runs=" << result.measured_runs << '\n';
  std::cout << "recall_at_k=" << result.recall_at_k << '\n';
  std::cout << "elapsed_seconds=" << result.elapsed_seconds << '\n';
  std::cout << "mean_latency_ms=" << result.mean_latency_ms << '\n';
  std::cout << "queries_per_second=" << result.queries_per_second << '\n';
  std::cout << "query_batch_size=" << result.query_batch_size << '\n';
  std::cout << "concurrency=" << result.concurrency << '\n';
  std::cout << "latency_sample_count=" << result.latency_sample_count << '\n';
  std::cout << "batch_latency_p50_ms=" << result.batch_latency_p50_ms << '\n';
  std::cout << "batch_latency_p95_ms=" << result.batch_latency_p95_ms << '\n';
  std::cout << "batch_latency_p99_ms=" << result.batch_latency_p99_ms << '\n';
  std::cout << "peak_rss_bytes=";
  if (peak_rss_bytes.has_value()) {
    std::cout << *peak_rss_bytes;
  } else {
    std::cout << "unsupported";
  }
  std::cout << '\n';
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
    return Run(command);
  } catch (const std::exception& error) {
    std::cerr << "hypervec_eval: " << error.what() << '\n';
    PrintUsage(std::cerr);
    return 1;
  }
}
