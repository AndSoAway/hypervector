/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <H5Cpp.h>

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

std::string MakeDataset() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const char* temporary_directory = std::getenv("TMPDIR");
#if defined(_WIN32)
  if (temporary_directory == nullptr) {
    temporary_directory = std::getenv("TEMP");
  }
#endif
  std::string path = temporary_directory == nullptr ? "." : temporary_directory;
  if (!path.empty() && path.back() != '/' && path.back() != '\\') {
    path.push_back('/');
  }
  path += "hypervec_eval_test_" + std::to_string(stamp) + ".hdf5";
  H5::H5File file(path, H5F_ACC_TRUNC);
  const hsize_t train_dimensions[2] = {4, 2};
  const hsize_t test_dimensions[2] = {2, 2};
  const hsize_t truth_dimensions[2] = {2, 2};
  const float train[] = {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
  const float test[] = {0.1F, 0.1F, 0.9F, 0.9F};
  const std::int64_t neighbors[] = {0, 1, 3, 1};
  const float distances[] = {0.02F, 0.82F, 0.02F, 0.82F};
  H5::DataSpace train_space(2, train_dimensions);
  H5::DataSpace test_space(2, test_dimensions);
  H5::DataSpace truth_space(2, truth_dimensions);
  file.createDataSet("train", H5::PredType::NATIVE_FLOAT, train_space)
      .write(train, H5::PredType::NATIVE_FLOAT);
  file.createDataSet("test", H5::PredType::NATIVE_FLOAT, test_space)
      .write(test, H5::PredType::NATIVE_FLOAT);
  file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, truth_space)
      .write(neighbors, H5::PredType::NATIVE_INT64);
  file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, truth_space)
      .write(distances, H5::PredType::NATIVE_FLOAT);
  return path;
}

void TestRecall() {
  const float results[] = {0.1F, 1.0F};
  const float ground_truth[] = {0.1F, 0.9F};
  ExpectNear(
      hypervec::eval::RecallMonitor::DistanceRecall(results, ground_truth, 2),
      0.5, 1e-12, "distance recall is incorrect");
  const std::int64_t result_ids[] = {7, 7};
  const std::int64_t truth_ids[] = {7, 8};
  ExpectNear(hypervec::eval::RecallMonitor::IdRecall(result_ids, truth_ids, 2),
             0.5, 1e-12, "duplicate IDs must not count twice");
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
  const std::string path = MakeDataset();
  const std::string index_config_path = path + ".index.json";
  {
    std::ofstream output(index_config_path);
    output << R"({"type":"flat","dimension":2,"metric":"l2"})";
  }
  try {
    const auto dataset = hypervec::eval::EvalDataset::Load(path);
    Expect(dataset->GetDim() == 2, "dataset dimension is incorrect");
    Expect(dataset->GetNumberOfBase() == 4, "dataset base count is incorrect");
    hypervec::eval::EvalConfig config;
    config.dataset_path = path;
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
    std::remove(path.c_str());
    std::remove(index_config_path.c_str());
    throw;
  }
  std::remove(path.c_str());
  std::remove(index_config_path.c_str());
}

}  // namespace

int main() {
  try {
    TestRecall();
    TestLatency();
    TestEndToEnd();
    std::cout << "hypervec_eval tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hypervec_eval tests failed: " << error.what() << '\n';
    return 1;
  }
}
