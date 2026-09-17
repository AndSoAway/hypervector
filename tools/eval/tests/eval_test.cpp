/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "eval/case/eval_case.h"
#include "eval/eval_config.h"
#include "eval/eval_dataset.h"
#include "eval/monitor/latency_monitor.h"
#include "eval/monitor/recall_monitor.h"

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectNear(double actual, double expected, double tolerance,
                const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message);
  }
}

std::string TemporaryDirectory() {
  const char* directory = std::getenv("TMPDIR");
#if defined(_WIN32)
  if (directory == nullptr) {
    directory = std::getenv("TEMP");
  }
#endif
  std::string result = directory == nullptr ? "." : directory;
  if (!result.empty() && result.back() != '/' && result.back() != '\\') {
    result.push_back('/');
  }
  return result;
}

void WriteFvecs(const std::string& path, const std::vector<float>& vectors,
                std::int32_t dimension) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create fvecs file: " + path);
  }
  for (std::size_t i = 0; i < vectors.size();
       i += static_cast<std::size_t>(dimension)) {
    output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    output.write(reinterpret_cast<const char*>(vectors.data() + i),
                 static_cast<std::streamsize>(dimension * sizeof(float)));
  }
}

void WriteIvecs(const std::string& path, const std::vector<std::int64_t>& ids,
                std::int32_t dimension) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create ivecs file: " + path);
  }
  for (std::size_t i = 0; i < ids.size();
       i += static_cast<std::size_t>(dimension)) {
    output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    for (std::int32_t j = 0; j < dimension; ++j) {
      const std::int32_t id = static_cast<std::int32_t>(ids[i + j]);
      output.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
  }
}

struct DatasetPaths {
  std::string base;
  std::string query;
  std::string groundtruth;
};

DatasetPaths MakeDataset() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string prefix =
      TemporaryDirectory() + "hypervec_eval_test_" + std::to_string(stamp);
  DatasetPaths paths{prefix + ".base.fvecs", prefix + ".query.fvecs",
                     prefix + ".groundtruth.ivecs"};
  const std::vector<float> train = {0.0F, 0.0F, 1.0F, 0.0F,
                                    0.0F, 1.0F, 1.0F, 1.0F};
  const std::vector<float> test = {0.1F, 0.1F, 0.9F, 0.9F};
  const std::vector<std::int64_t> neighbors = {0, 1, 3, 1};
  WriteFvecs(paths.base, train, 2);
  WriteFvecs(paths.query, test, 2);
  WriteIvecs(paths.groundtruth, neighbors, 2);
  return paths;
}

void TestRecall() {
  const std::int64_t result_ids[] = {7, 7};
  const std::int64_t truth_ids[] = {7, 8};
  ExpectNear(hypervec::eval::RecallMonitor::IdRecall(result_ids, truth_ids, 2),
             0.5, 1e-12, "duplicate IDs must not count twice");
}

void TestGroundTruthValidation() {
  const DatasetPaths paths = MakeDataset();
  const std::vector<std::int64_t> bad_neighbors = {0, 1, 3, 9999};
  WriteIvecs(paths.groundtruth, bad_neighbors, 2);
  bool rejected = false;
  try {
    hypervec::eval::EvalDataset::Load(paths.base, paths.query,
                                      paths.groundtruth);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  std::remove(paths.base.c_str());
  std::remove(paths.query.c_str());
  std::remove(paths.groundtruth.c_str());
  Expect(rejected, "out-of-range ground-truth id must be rejected");
}

void TestLatency() {
  hypervec::eval::LatencyMonitor monitor;
  monitor.Start();
  monitor.Stop();
  monitor.SetRecords({1.0, 2.0, 3.0, 4.0, 5.0}, 5);
  const auto result = monitor.GetResult();
  ExpectNear(result["latency_avg(ms)"], 3.0, 1e-12,
             "average latency is incorrect");
  ExpectNear(result["latency_p80(ms)"], 4.0, 1e-12, "p80 latency is incorrect");
}

void TestEndToEnd() {
  const DatasetPaths paths = MakeDataset();
  const std::string index_config_path = paths.base + ".index.json";
  {
    std::ofstream output(index_config_path);
    output << R"({"type":"flat","dimension":2,"metric":"l2"})";
  }
  try {
    const auto dataset = hypervec::eval::EvalDataset::Load(
        paths.base, paths.query, paths.groundtruth);
    Expect(dataset->GetDim() == 2, "dataset dimension is incorrect");
    Expect(dataset->GetNumberOfBase() == 4, "dataset base count is incorrect");
    Expect(dataset->GetNumberOfQuery() == 2,
           "dataset query count is incorrect");
    Expect(dataset->GetGroundTruthK() == 2,
           "dataset ground-truth k is incorrect");
    hypervec::eval::EvalConfig config;
    config.base_path = paths.base;
    config.query_path = paths.query;
    config.groundtruth_path = paths.groundtruth;
    config.index_config_path = index_config_path;
    config.action = "build_search";
    config.top_k = 2;
    const auto result = hypervec::eval::RunEvalCase(config);
    ExpectNear(result["recall_avg"], 1.0, 1e-12,
               "flat index must have exact recall");
    Expect(result["index_config"]["type"] == "flat",
           "result must contain the normalized public index config");
    for (const char* key :
         {"duration(s)", "tps", "index_memory(B)", "qps", "latency_avg(ms)",
          "latency_p50(ms)", "latency_p80(ms)", "latency_p90(ms)",
          "latency_p95(ms)", "latency_p99(ms)", "recall_avg", "recall_p0",
          "recall_p10", "recall_p30", "recall_p50", "recall_p70",
          "recall_p90"}) {
      Expect(result.contains(key), "an expected metric is missing");
    }
  } catch (...) {
    std::remove(paths.base.c_str());
    std::remove(paths.query.c_str());
    std::remove(paths.groundtruth.c_str());
    std::remove(index_config_path.c_str());
    throw;
  }
  std::remove(paths.base.c_str());
  std::remove(paths.query.c_str());
  std::remove(paths.groundtruth.c_str());
  std::remove(index_config_path.c_str());
}

}  // namespace

int main() {
  try {
    TestRecall();
    TestLatency();
    TestGroundTruthValidation();
    TestEndToEnd();
    std::cout << "hypervec_eval tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hypervec_eval tests failed: " << error.what() << '\n';
    return 1;
  }
}
