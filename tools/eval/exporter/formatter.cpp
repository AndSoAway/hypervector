/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/exporter/formatter.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Third-party dependencies.
#include <tabulate/table.hpp>

namespace hypervec::eval {
namespace {

void FlattenJson(const nlohmann::json& value, const std::string& prefix,
                 std::vector<std::pair<std::string, std::string>>& output) {
  if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      FlattenJson(child, prefix.empty() ? key : prefix + "." + key, output);
    }
    return;
  }
  if (value.is_array()) {
    output.emplace_back(prefix, value.dump());
    return;
  }
  output.emplace_back(
      prefix, value.is_string() ? value.get<std::string>() : value.dump());
}

}  // namespace

std::string FormatterJson::Format(const nlohmann::json& value) const {
  return value.dump(2) + "\n";
}

std::string FormatterTable::Format(const nlohmann::json& value) const {
  std::vector<std::pair<std::string, std::string>> values;
  FlattenJson(value, "", values);
  tabulate::Table table;
  table.add_row({"metric", "value"});
  for (const auto& [key, item] : values) {
    table.add_row({key, item});
  }
  table[0].format().font_style({tabulate::FontStyle::bold});
  std::ostringstream output;
  output << table << '\n';
  return output.str();
}

std::unique_ptr<Formatter> MakeFormatter(const std::string& name) {
  if (name == "json") {
    return std::make_unique<FormatterJson>();
  }
  if (name == "table") {
    return std::make_unique<FormatterTable>();
  }
  throw std::invalid_argument("unsupported output format: " + name);
}

}  // namespace hypervec::eval
