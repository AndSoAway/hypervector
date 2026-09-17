/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/eval_dataset.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hypervec::eval {
namespace {

std::vector<float> ReadFvecs(const std::string& path, std::int64_t& dimension) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open fvecs file: " + path);
  }
  std::vector<float> vectors;
  dimension = -1;
  for (;;) {
    std::int32_t record_dim = 0;
    input.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
    if (input.gcount() == 0) {
      break;  // clean end of file
    }
    if (input.gcount() != static_cast<std::streamsize>(sizeof(record_dim))) {
      throw std::runtime_error("truncated fvecs header: " + path);
    }
    if (record_dim <= 0) {
      throw std::runtime_error("invalid dimension in fvecs file: " + path);
    }
    if (dimension == -1) {
      dimension = record_dim;
    } else if (dimension != record_dim) {
      throw std::runtime_error("inconsistent dimensions in fvecs file: " +
                               path);
    }
    const std::size_t offset = vectors.size();
    vectors.resize(offset + static_cast<std::size_t>(record_dim));
    input.read(reinterpret_cast<char*>(vectors.data() + offset),
               static_cast<std::streamsize>(record_dim * sizeof(float)));
    if (input.gcount() !=
        static_cast<std::streamsize>(record_dim * sizeof(float))) {
      throw std::runtime_error("truncated fvecs body: " + path);
    }
  }
  if (vectors.empty() || dimension <= 0) {
    throw std::runtime_error("empty fvecs file: " + path);
  }
  return vectors;
}

std::vector<std::int64_t> ReadIvecs(const std::string& path,
                                    std::int64_t& dimension) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open ivecs file: " + path);
  }
  std::vector<std::int64_t> ids;
  dimension = -1;
  for (;;) {
    std::int32_t record_dim = 0;
    input.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
    if (input.gcount() == 0) {
      break;  // clean end of file
    }
    if (input.gcount() != static_cast<std::streamsize>(sizeof(record_dim))) {
      throw std::runtime_error("truncated ivecs header: " + path);
    }
    if (record_dim <= 0) {
      throw std::runtime_error("invalid dimension in ivecs file: " + path);
    }
    if (dimension == -1) {
      dimension = record_dim;
    } else if (dimension != record_dim) {
      throw std::runtime_error("inconsistent dimensions in ivecs file: " +
                               path);
    }
    std::vector<std::int32_t> row(static_cast<std::size_t>(record_dim));
    input.read(reinterpret_cast<char*>(row.data()),
               static_cast<std::streamsize>(record_dim * sizeof(std::int32_t)));
    if (input.gcount() !=
        static_cast<std::streamsize>(record_dim * sizeof(std::int32_t))) {
      throw std::runtime_error("truncated ivecs body: " + path);
    }
    for (const std::int32_t id : row) {
      ids.push_back(static_cast<std::int64_t>(id));
    }
  }
  if (ids.empty() || dimension <= 0) {
    throw std::runtime_error("empty ivecs file: " + path);
  }
  return ids;
}

}  // namespace

std::shared_ptr<EvalDataset> EvalDataset::Load(
    const std::string& base_path, const std::string& query_path,
    const std::string& groundtruth_path) {
  auto result = std::make_shared<EvalDataset>();
  std::int64_t base_dim = 0;
  std::int64_t query_dim = 0;
  std::int64_t ground_truth_k = 0;
  result->train_ = ReadFvecs(base_path, base_dim);
  result->test_ = ReadFvecs(query_path, query_dim);
  result->neighbors_ = ReadIvecs(groundtruth_path, ground_truth_k);
  if (base_dim != query_dim) {
    throw std::invalid_argument("base and query dimensions do not match");
  }
  result->path_ = base_path;
  result->dim_ = base_dim;
  result->ground_truth_k_ = ground_truth_k;
  result->n_base_ = static_cast<std::int64_t>(result->train_.size()) / base_dim;
  result->n_query_ = static_cast<std::int64_t>(result->test_.size()) / base_dim;
  if (result->n_query_ * ground_truth_k !=
      static_cast<std::int64_t>(result->neighbors_.size())) {
    throw std::invalid_argument("ground-truth size does not match query count");
  }
  for (const std::int64_t id : result->neighbors_) {
    if (id < 0 || id >= result->n_base_) {
      throw std::invalid_argument(
          "ground-truth contains an out-of-range neighbor id: " +
          std::to_string(id));
    }
  }
  return result;
}

const std::int64_t* EvalDataset::GetNeighbors(std::int64_t query) const {
  if (query < 0 || query >= n_query_) {
    throw std::out_of_range("query index is outside the dataset");
  }
  return neighbors_.data() + query * ground_truth_k_;
}

}  // namespace hypervec::eval
