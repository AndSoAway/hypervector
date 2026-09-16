/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <factory/index_descriptor.h>

#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hypervec {

/** Thread-safe mapping from stable names and aliases to index descriptors. */
class IndexRegistry {
 public:
  static IndexRegistry& Global();

  void Register(IndexDescriptor descriptor);

  std::optional<IndexDescriptor> Find(std::string_view name) const;

  std::vector<IndexDescriptor> List() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, IndexDescriptor> descriptors_;
  std::unordered_map<std::string, std::string> aliases_;
};

/** Register all built-in CPU indexes exactly once. */
void RegisterBuiltinIndexes();

}  // namespace hypervec
