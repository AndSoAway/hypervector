/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <memory>
#include <string>

// Third-party dependencies.
#include <nlohmann/json.hpp>

namespace hypervec::eval {

class Formatter {
 public:
  virtual ~Formatter() = default;
  virtual std::string Format(const nlohmann::json& value) const = 0;
};

class FormatterJson : public Formatter {
 public:
  std::string Format(const nlohmann::json& value) const override;
};

class FormatterTable : public Formatter {
 public:
  std::string Format(const nlohmann::json& value) const override;
};

std::unique_ptr<Formatter> MakeFormatter(const std::string& name);

}  // namespace hypervec::eval
