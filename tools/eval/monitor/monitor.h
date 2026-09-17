/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <string>

// Third-party dependencies.
#include <nlohmann/json.hpp>

namespace hypervec::eval {

class Monitor {
 public:
  explicit Monitor(std::string name);
  virtual ~Monitor() = default;
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual nlohmann::json GetResult() const = 0;
  const std::string& GetName() const { return name_; }

 private:
  std::string name_;
};

}  // namespace hypervec::eval
