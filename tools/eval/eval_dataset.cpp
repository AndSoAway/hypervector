/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/eval_dataset.h"

#include <H5Cpp.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace hypervec::eval {
namespace {

struct MatrixShape {
  hsize_t rows;
  hsize_t columns;
};

MatrixShape GetMatrixShape(const H5::DataSet& dataset, const char* name) {
  const H5::DataSpace space = dataset.getSpace();
  if (space.getSimpleExtentNdims() != 2) {
    throw std::invalid_argument(std::string(name) +
                                " must be a rank-2 HDF5 dataset");
  }
  hsize_t dimensions[2] = {0, 0};
  space.getSimpleExtentDims(dimensions);
  if (dimensions[0] == 0 || dimensions[1] == 0) {
    throw std::invalid_argument(std::string(name) + " must not be empty");
  }
  return {dimensions[0], dimensions[1]};
}

std::size_t CheckedElementCount(const MatrixShape& shape, const char* name) {
  if (shape.rows > std::numeric_limits<std::size_t>::max() / shape.columns) {
    throw std::overflow_error(std::string(name) + " is too large");
  }
  return static_cast<std::size_t>(shape.rows * shape.columns);
}

}  // namespace

std::shared_ptr<EvalDataset> EvalDataset::Load(const std::string& path) {
  H5::Exception::dontPrint();
  try {
    H5::H5File file(path, H5F_ACC_RDONLY);
    const H5::DataSet train_dataset = file.openDataSet("train");
    const H5::DataSet test_dataset = file.openDataSet("test");
    const H5::DataSet neighbors_dataset = file.openDataSet("neighbors");
    const H5::DataSet distances_dataset = file.openDataSet("distances");
    const MatrixShape train_shape = GetMatrixShape(train_dataset, "train");
    const MatrixShape test_shape = GetMatrixShape(test_dataset, "test");
    const MatrixShape neighbors_shape =
        GetMatrixShape(neighbors_dataset, "neighbors");
    const MatrixShape distances_shape =
        GetMatrixShape(distances_dataset, "distances");
    if (train_shape.columns != test_shape.columns) {
      throw std::invalid_argument("train and test dimensions do not match");
    }
    if (test_shape.rows != neighbors_shape.rows ||
        neighbors_shape.rows != distances_shape.rows ||
        neighbors_shape.columns != distances_shape.columns) {
      throw std::invalid_argument(
          "ground-truth shapes do not match test queries");
    }
    auto result = std::make_shared<EvalDataset>();
    result->path_ = path;
    result->n_base_ = static_cast<std::int64_t>(train_shape.rows);
    result->n_query_ = static_cast<std::int64_t>(test_shape.rows);
    result->dim_ = static_cast<std::int64_t>(train_shape.columns);
    result->ground_truth_k_ =
        static_cast<std::int64_t>(neighbors_shape.columns);
    result->train_.resize(CheckedElementCount(train_shape, "train"));
    result->test_.resize(CheckedElementCount(test_shape, "test"));
    result->neighbors_.resize(
        CheckedElementCount(neighbors_shape, "neighbors"));
    result->distances_.resize(
        CheckedElementCount(distances_shape, "distances"));
    train_dataset.read(result->train_.data(), H5::PredType::NATIVE_FLOAT);
    test_dataset.read(result->test_.data(), H5::PredType::NATIVE_FLOAT);
    neighbors_dataset.read(result->neighbors_.data(),
                           H5::PredType::NATIVE_INT64);
    distances_dataset.read(result->distances_.data(),
                           H5::PredType::NATIVE_FLOAT);
    return result;
  } catch (const H5::Exception& error) {
    throw std::runtime_error("failed to load HDF5 dataset '" + path +
                             "': " + error.getDetailMsg());
  }
}

const std::int64_t* EvalDataset::GetNeighbors(std::int64_t query) const {
  if (query < 0 || query >= n_query_) {
    throw std::out_of_range("query index is outside the dataset");
  }
  return neighbors_.data() + query * ground_truth_k_;
}

const float* EvalDataset::GetDistances(std::int64_t query) const {
  if (query < 0 || query >= n_query_) {
    throw std::out_of_range("query index is outside the dataset");
  }
  return distances_.data() + query * ground_truth_k_;
}

}  // namespace hypervec::eval
