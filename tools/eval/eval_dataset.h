/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hypervec::eval {

class EvalDataset {
 public:
  static std::shared_ptr<EvalDataset> Load(const std::string& path);

  const float* GetTrain() const { return train_.data(); }
  const float* GetTest() const { return test_.data(); }
  const std::int64_t* GetNeighbors(std::int64_t query) const;
  const float* GetDistances(std::int64_t query) const;
  std::int64_t GetDim() const { return dim_; }
  std::int64_t GetNumberOfBase() const { return n_base_; }
  std::int64_t GetNumberOfQuery() const { return n_query_; }
  std::int64_t GetGroundTruthK() const { return ground_truth_k_; }
  const std::string& GetPath() const { return path_; }

 private:
  std::string path_;
  std::vector<float> train_;
  std::vector<float> test_;
  std::vector<std::int64_t> neighbors_;
  std::vector<float> distances_;
  std::int64_t dim_ = 0;
  std::int64_t n_base_ = 0;
  std::int64_t n_query_ = 0;
  std::int64_t ground_truth_k_ = 0;
};

}  // namespace hypervec::eval
