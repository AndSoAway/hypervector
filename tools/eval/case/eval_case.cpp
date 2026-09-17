/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/case/eval_case.h"

#include <factory/engine.h>
#include <factory/index_config.h>
#include <omp.h>
#include <persistence/index_io.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eval/monitor/duration_monitor.h"
#include "eval/monitor/latency_monitor.h"
#include "eval/monitor/recall_monitor.h"
#include "eval/platform_memory.h"

namespace hypervec::eval {
namespace {

class ScopedOmpThreads {
 public:
  explicit ScopedOmpThreads(int threads) : previous_(omp_get_max_threads()) {
    omp_set_num_threads(threads);
  }
  ~ScopedOmpThreads() { omp_set_num_threads(previous_); }
  ScopedOmpThreads(const ScopedOmpThreads&) = delete;
  ScopedOmpThreads& operator=(const ScopedOmpThreads&) = delete;

 private:
  int previous_;
};

void RequireCapability(const hypervec::Index& index,
                       hypervec::IndexCapability capability,
                       const char* operation) {
  if (!hypervec::HasCapability(index.Capabilities(), capability)) {
    throw std::invalid_argument(std::string("index type '") +
                                std::string(index.TypeName()) +
                                "' does not support " + operation);
  }
}

void Merge(nlohmann::json& target, const nlohmann::json& source) {
  for (const auto& [key, value] : source.items()) {
    target[key] = value;
  }
}

}  // namespace

EvalCase::EvalCase(
    EvalConfig config, std::shared_ptr<EvalDataset> dataset,
    std::unique_ptr<hypervec::Index> index,
    std::unique_ptr<hypervec::SearchParameters> search_parameters)
    : config_(std::move(config)),
      dataset_(std::move(dataset)),
      index_(std::move(index)),
      search_parameters_(std::move(search_parameters)) {
  // Intentionally empty.
}

nlohmann::json EvalCase::Metadata() const {
  return {
      {"name", config_.name},
      {"action", config_.action},
      {"index_type", index_->TypeName()},
      {"index_config", nlohmann::json::parse(config_.canonical_index_config)},
      {"search_config", nlohmann::json::parse(config_.canonical_search_config)},
      {"dataset", dataset_->GetPath()},
      {"dimension", dataset_->GetDim()},
      {"base_count", dataset_->GetNumberOfBase()},
      {"query_count", dataset_->GetNumberOfQuery()},
      {"top_k", config_.top_k}};
}

nlohmann::json BuildEvalCase::Run() {
  RequireCapability(*index_, hypervec::IndexCapability::BatchBuild,
                    "batch build");
  ScopedOmpThreads scoped_threads(config_.num_threads_building);
  const std::uint64_t rss_before = GetCurrentRssBytes();
  DurationMonitor duration(
      static_cast<std::uint64_t>(dataset_->GetNumberOfBase()));
  duration.Start();
  index_->Build({dataset_->GetNumberOfBase(), dataset_->GetDim(),
                 dataset_->GetTrain(), nullptr});
  duration.Stop();
  const std::uint64_t rss_after = GetCurrentRssBytes();
  if (!config_.index_path.empty()) {
    hypervec::WriteIndex(index_.get(), config_.index_path.c_str());
  }
  nlohmann::json result = Metadata();
  result["action"] = "build";
  const nlohmann::json duration_result = duration.GetResult();
  if (config_.enable_tps) {
    result["tps"] = duration_result["tps"];
  }
  result["duration(s)"] = duration_result["duration(s)"];
  if (config_.enable_memory) {
    result["index_memory(B)"] =
        rss_after >= rss_before ? rss_after - rss_before : 0;
  }
  return result;
}

nlohmann::json SearchEvalCase::Run() {
  RequireCapability(*index_, hypervec::IndexCapability::KnnSearch,
                    "k-nearest-neighbor search");
  const std::uint64_t available =
      static_cast<std::uint64_t>(dataset_->GetNumberOfQuery());
  const std::uint64_t query_count =
      config_.search_query_count == 0
          ? available
          : std::min(config_.search_query_count, available);
  if (query_count == 0) {
    throw std::invalid_argument("the dataset contains no search queries");
  }
  if (config_.enable_recall && config_.top_k > dataset_->GetGroundTruthK()) {
    throw std::invalid_argument("top_k exceeds the available ground truth");
  }

  const std::size_t top_k = static_cast<std::size_t>(config_.top_k);
  std::vector<float> distances(query_count * top_k);
  std::vector<hypervec::idx_t> labels(query_count * top_k);
  std::vector<double> latency_ms(query_count, -1.0);
  std::atomic<std::uint64_t> successful_queries{0};
  std::exception_ptr search_error;
  std::mutex error_mutex;
  ScopedOmpThreads scoped_threads(config_.num_threads_searching);
  LatencyMonitor latency;
  latency.Start();
#pragma omp parallel for schedule(static)
  for (std::int64_t query = 0; query < static_cast<std::int64_t>(query_count);
       ++query) {
    try {
      const auto start = std::chrono::steady_clock::now();
      index_->Search(1, dataset_->GetTest() + query * dataset_->GetDim(),
                     config_.top_k, distances.data() + query * top_k,
                     labels.data() + query * top_k, search_parameters_.get());
      const auto stop = std::chrono::steady_clock::now();
      latency_ms[query] =
          std::chrono::duration<double, std::milli>(stop - start).count();
      successful_queries.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (search_error == nullptr) {
        search_error = std::current_exception();
      }
    }
  }
  latency.Stop();
  if (search_error != nullptr) {
    std::rethrow_exception(search_error);
  }
  latency.SetRecords(std::move(latency_ms), successful_queries.load());

  nlohmann::json result = Metadata();
  result["action"] = "search";
  const nlohmann::json latency_result = latency.GetResult();
  if (config_.enable_qps) {
    result["qps"] = latency_result["qps"];
  }
  if (config_.enable_latency) {
    result["latency_avg(ms)"] = latency_result["latency_avg(ms)"];
  }
  if (config_.enable_percent_latency) {
    for (const char* key :
         {"latency_p50(ms)", "latency_p80(ms)", "latency_p90(ms)",
          "latency_p95(ms)", "latency_p99(ms)"}) {
      result[key] = latency_result[key];
    }
  }
  if (config_.enable_recall) {
    RecallMonitor recall;
    recall.Start();
    for (std::uint64_t query = 0; query < query_count; ++query) {
      recall.Record(labels.data() + query * top_k,
                    dataset_->GetNeighbors(query), top_k);
    }
    recall.Stop();
    const nlohmann::json recall_result = recall.GetResult();
    result["recall_avg"] = recall_result["recall_avg"];
    if (config_.enable_percent_recall) {
      for (const char* key : {"recall_p0", "recall_p10", "recall_p30",
                              "recall_p50", "recall_p70", "recall_p90"}) {
        result[key] = recall_result[key];
      }
    }
  }
  result["measured_query_count"] = successful_queries.load();
  return result;
}

nlohmann::json BuildSearchEvalCase::Run() {
  BuildEvalCase build(config_, dataset_, std::move(index_));
  nlohmann::json build_result = build.Run();
  index_ = build.TakeIndex();
  SearchEvalCase search(config_, dataset_, std::move(index_),
                        std::move(search_parameters_));
  nlohmann::json search_result = search.Run();
  index_ = search.TakeIndex();
  Merge(build_result, search_result);
  build_result["action"] = "build_search";
  return build_result;
}

nlohmann::json RunEvalCase(const EvalConfig& config) {
  ValidateEvalConfig(config);
  auto dataset = EvalDataset::Load(config.base_path, config.query_path,
                                   config.groundtruth_path);
  hypervec::Engine engine;
  const hypervec::IndexConfig index_request =
      hypervec::ParseIndexConfig(ReadTextFile(config.index_config_path));
  const hypervec::IndexConfig index_config =
      engine.NormalizeConfig(index_request);
  if (index_config.dimension != dataset->GetDim()) {
    throw std::invalid_argument(
        "index configuration dimension does not match the dataset");
  }
  hypervec::SearchConfig search_request;
  if (!config.search_config_path.empty()) {
    search_request =
        hypervec::ParseSearchConfig(ReadTextFile(config.search_config_path));
  }
  const hypervec::SearchConfig search_config =
      engine.NormalizeSearchConfig(index_config, search_request);
  EvalConfig normalized_config = config;
  normalized_config.canonical_index_config =
      hypervec::SerializeIndexConfig(index_config);
  normalized_config.canonical_search_config =
      hypervec::SerializeSearchConfig(search_config);

  std::unique_ptr<hypervec::Index> index;
  if (config.action == "search") {
    index = hypervec::ReadIndexUp(config.index_path.c_str());
    if (index->d != index_config.dimension) {
      throw std::invalid_argument(
          "loaded index dimension does not match the index configuration");
    }
    const std::string expected_type =
        index_config.id_mode == hypervec::IdMode::External ? "idmap"
                                                           : index_config.type;
    if (index->TypeName() != expected_type) {
      throw std::invalid_argument(
          "loaded index type does not match the index configuration");
    }
    if (index->metric_type != index_config.metric) {
      throw std::invalid_argument(
          "loaded index metric does not match the index configuration");
    }
  } else {
    index = engine.CreateIndex(index_config);
  }
  if (config.action == "build") {
    return BuildEvalCase(normalized_config, std::move(dataset),
                         std::move(index))
        .Run();
  }
  auto search_parameters =
      engine.CreateSearchParameters(index_config, search_config);
  if (config.action == "search") {
    return SearchEvalCase(normalized_config, std::move(dataset),
                          std::move(index), std::move(search_parameters))
        .Run();
  }
  return BuildSearchEvalCase(normalized_config, std::move(dataset),
                             std::move(index), std::move(search_parameters))
      .Run();
}

}  // namespace hypervec::eval
