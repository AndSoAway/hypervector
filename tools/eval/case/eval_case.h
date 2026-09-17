/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <memory>
#include <utility>

// Third-party dependencies.
#include <nlohmann/json.hpp>

#include "eval/eval_config.h"
#include "eval/eval_dataset.h"

namespace hypervec::eval {

class EvalCase {
 public:
  EvalCase(
      EvalConfig config, std::shared_ptr<EvalDataset> dataset,
      std::unique_ptr<hypervec::Index> index,
      std::unique_ptr<hypervec::SearchParameters> search_parameters = nullptr);
  virtual ~EvalCase() = default;
  virtual nlohmann::json Run() = 0;
  std::unique_ptr<hypervec::Index> TakeIndex() { return std::move(index_); }

 protected:
  nlohmann::json Metadata() const;
  EvalConfig config_;
  std::shared_ptr<EvalDataset> dataset_;
  std::unique_ptr<hypervec::Index> index_;
  std::unique_ptr<hypervec::SearchParameters> search_parameters_;
};

class BuildEvalCase : public EvalCase {
 public:
  using EvalCase::EvalCase;
  nlohmann::json Run() override;
};

class SearchEvalCase : public EvalCase {
 public:
  using EvalCase::EvalCase;
  nlohmann::json Run() override;
};

class BuildSearchEvalCase : public EvalCase {
 public:
  using EvalCase::EvalCase;
  nlohmann::json Run() override;
};

nlohmann::json RunEvalCase(const EvalConfig& config);

}  // namespace hypervec::eval
