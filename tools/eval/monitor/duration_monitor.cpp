/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/monitor/duration_monitor.h"

namespace hypervec::eval {

DurationMonitor::DurationMonitor(std::uint64_t item_count)
    : Monitor("duration"), item_count_(item_count) {}

void DurationMonitor::Start() { start_time_ = Clock::now(); }

void DurationMonitor::Stop() {
  duration_seconds_ =
      std::chrono::duration<double>(Clock::now() - start_time_).count();
}

nlohmann::json DurationMonitor::GetResult() const {
  const double tps = duration_seconds_ > 0.0
                         ? static_cast<double>(item_count_) / duration_seconds_
                         : 0.0;
  return {{"duration(s)", duration_seconds_}, {"tps", tps}};
}

}  // namespace hypervec::eval
