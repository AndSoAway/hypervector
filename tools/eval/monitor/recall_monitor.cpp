/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/monitor/recall_monitor.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace hypervec::eval {

RecallMonitor::RecallMonitor() : Monitor("recall") {}

void RecallMonitor::Start() { records_.clear(); }
void RecallMonitor::Stop() { std::sort(records_.begin(), records_.end()); }

double RecallMonitor::IdRecall(const std::int64_t* result_ids,
                               const std::int64_t* ground_truth_ids,
                               std::uint64_t top_k) {
  if (top_k == 0) {
    return 0.0;
  }
  std::unordered_set<std::int64_t> remaining(ground_truth_ids,
                                             ground_truth_ids + top_k);
  std::uint64_t matches = 0;
  for (std::uint64_t i = 0; i < top_k; ++i) {
    if (remaining.erase(result_ids[i]) > 0) {
      ++matches;
    }
  }
  return static_cast<double>(matches) / static_cast<double>(top_k);
}

void RecallMonitor::Record(const std::int64_t* result_ids,
                           const std::int64_t* ground_truth_ids,
                           std::uint64_t top_k) {
  if (result_ids == nullptr || ground_truth_ids == nullptr) {
    throw std::invalid_argument("recall inputs must not be null");
  }
  records_.push_back(IdRecall(result_ids, ground_truth_ids, top_k));
}

double RecallMonitor::Percentile(double rate) const {
  if (records_.empty()) {
    return 0.0;
  }
  const auto position =
      static_cast<std::size_t>(rate * static_cast<double>(records_.size() - 1));
  return records_[position];
}

nlohmann::json RecallMonitor::GetResult() const {
  const double total = std::accumulate(records_.begin(), records_.end(), 0.0);
  const double average =
      records_.empty() ? 0.0 : total / static_cast<double>(records_.size());
  return {{"recall_avg", average},          {"recall_p0", Percentile(0.0)},
          {"recall_p10", Percentile(0.10)}, {"recall_p30", Percentile(0.30)},
          {"recall_p50", Percentile(0.50)}, {"recall_p70", Percentile(0.70)},
          {"recall_p90", Percentile(0.90)}};
}

}  // namespace hypervec::eval
