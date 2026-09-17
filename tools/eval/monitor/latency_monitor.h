/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "eval/monitor/monitor.h"

namespace hypervec::eval {

class LatencyMonitor : public Monitor {
 public:
  LatencyMonitor();
  void Start() override;
  void Stop() override;
  nlohmann::json GetResult() const override;
  void SetRecords(std::vector<double> latency_ms,
                  std::uint64_t successful_query_count);

 private:
  double Percentile(double rate) const;
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_time_{};
  std::vector<double> latency_ms_;
  std::uint64_t successful_query_count_ = 0;
  double wall_time_seconds_ = 0.0;
};

}  // namespace hypervec::eval
