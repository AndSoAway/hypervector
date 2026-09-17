/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/monitor/latency_monitor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace hypervec::eval {

LatencyMonitor::LatencyMonitor() : Monitor("latency") {}

void LatencyMonitor::Start() {
  latency_ms_.clear();
  successful_query_count_ = 0;
  wall_time_seconds_ = 0.0;
  start_time_ = Clock::now();
}

void LatencyMonitor::Stop() {
  wall_time_seconds_ =
      std::chrono::duration<double>(Clock::now() - start_time_).count();
}

void LatencyMonitor::SetRecords(std::vector<double> latency_ms,
                                std::uint64_t successful_query_count) {
  latency_ms.erase(std::remove_if(latency_ms.begin(), latency_ms.end(),
                                  [](double value) {
                                    return !std::isfinite(value) || value < 0.0;
                                  }),
                   latency_ms.end());
  latency_ms_ = std::move(latency_ms);
  std::sort(latency_ms_.begin(), latency_ms_.end());
  successful_query_count_ = successful_query_count;
}

double LatencyMonitor::Percentile(double rate) const {
  if (latency_ms_.empty()) {
    return 0.0;
  }
  const auto position = static_cast<std::size_t>(
      rate * static_cast<double>(latency_ms_.size() - 1));
  return latency_ms_[position];
}

nlohmann::json LatencyMonitor::GetResult() const {
  const double total =
      std::accumulate(latency_ms_.begin(), latency_ms_.end(), 0.0);
  const double average = latency_ms_.empty()
                             ? 0.0
                             : total / static_cast<double>(latency_ms_.size());
  const double qps =
      wall_time_seconds_ > 0.0
          ? static_cast<double>(successful_query_count_) / wall_time_seconds_
          : 0.0;
  return {{"qps", qps},
          {"latency_avg(ms)", average},
          {"latency_p50(ms)", Percentile(0.50)},
          {"latency_p80(ms)", Percentile(0.80)},
          {"latency_p90(ms)", Percentile(0.90)},
          {"latency_p95(ms)", Percentile(0.95)},
          {"latency_p99(ms)", Percentile(0.99)}};
}

}  // namespace hypervec::eval
