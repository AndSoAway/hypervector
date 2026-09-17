/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "eval/monitor/monitor.h"

namespace hypervec::eval {

class RecallMonitor : public Monitor {
 public:
  explicit RecallMonitor(bool use_id_based_recall = false);
  void Start() override;
  void Stop() override;
  nlohmann::json GetResult() const override;
  void Record(const float* result_distances, const std::int64_t* result_ids,
              const float* ground_truth_distances,
              const std::int64_t* ground_truth_ids, std::uint64_t top_k);
  static double DistanceRecall(const float* result_distances,
                               const float* ground_truth_distances,
                               std::uint64_t top_k);
  static double IdRecall(const std::int64_t* result_ids,
                         const std::int64_t* ground_truth_ids,
                         std::uint64_t top_k);

 private:
  double Percentile(double rate) const;
  bool use_id_based_recall_;
  std::vector<double> records_;
};

}  // namespace hypervec::eval
