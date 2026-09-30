/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "provenance.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace hypervec::eval_cli {

void WriteProvenance(const std::filesystem::path& path,
                     const ProvenanceRecord& record) {
  std::vector<std::string> keys;
  keys.reserve(record.size());
  for (const auto& [key, value] : record) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  std::ofstream output(path, std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot open provenance output: " + path.string());
  }
  for (const std::string& key : keys) {
    output << key << "=" << record.at(key) << "\n";
  }
  output.close();
  if (output.fail()) {
    throw std::runtime_error("cannot write provenance output: " +
                             path.string());
  }
}

std::optional<ProvenanceRecord> ReadProvenance(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  ProvenanceRecord record;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const size_t separator = line.find('=');
    if (separator == std::string::npos || separator == 0) {
      throw std::runtime_error("malformed provenance line in " + path.string());
    }
    record[line.substr(0, separator)] = line.substr(separator + 1);
  }
  return record;
}

}  // namespace hypervec::eval_cli
