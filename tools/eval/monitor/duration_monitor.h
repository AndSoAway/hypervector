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

#include "eval/monitor/monitor.h"

namespace hypervec::eval {

class DurationMonitor : public Monitor {
 public:
  explicit DurationMonitor(std::uint64_t item_count = 0);
  void Start() override;
  void Stop() override;
  nlohmann::json GetResult() const override;
  double DurationSeconds() const { return duration_seconds_; }

 private:
  using Clock = std::chrono::steady_clock;
  std::uint64_t item_count_;
  Clock::time_point start_time_{};
  double duration_seconds_ = 0.0;
};

}  // namespace hypervec::eval
