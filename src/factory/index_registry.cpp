/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/index_registry.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <utility>

namespace hypervec {

namespace {

std::string NormalizeName(std::string_view name) {
  if (name.empty()) {
    HYPERVEC_THROW_MSG("index name must not be empty");
  }

  std::string normalized;
  normalized.reserve(name.size());
  for (const unsigned char ch : name) {
    if (ch == '-') {
      normalized.push_back('_');
    } else if (std::isalnum(ch) != 0 || ch == '_') {
      normalized.push_back(static_cast<char>(std::tolower(ch)));
    } else {
      HYPERVEC_THROW_FMT("invalid character in index name '%.*s'",
                         static_cast<int>(name.size()), name.data());
    }
  }
  return normalized;
}

}  // namespace

IndexRegistry& IndexRegistry::Global() {
  static IndexRegistry registry;
  return registry;
}

void IndexRegistry::Register(IndexDescriptor descriptor) {
  descriptor.canonical_name = NormalizeName(descriptor.canonical_name);
  HYPERVEC_THROW_IF_NOT_MSG(static_cast<bool>(descriptor.normalize),
                            "index descriptor requires a config normalizer");
  HYPERVEC_THROW_IF_NOT_MSG(static_cast<bool>(descriptor.creator),
                            "index descriptor requires a creator");
  HYPERVEC_THROW_IF_NOT_MSG(!descriptor.metrics.empty(),
                            "index descriptor requires supported metrics");
  HYPERVEC_THROW_IF_NOT_MSG(!descriptor.backends.empty(),
                            "index descriptor requires supported backends");

  std::unordered_set<std::string> registration_names;
  registration_names.insert(descriptor.canonical_name);
  for (std::string& alias : descriptor.aliases) {
    alias = NormalizeName(alias);
    HYPERVEC_THROW_IF_NOT_MSG(
        registration_names.insert(alias).second,
        "index descriptor contains a duplicate canonical name or alias");
  }

  for (std::string& backend : descriptor.backends) {
    backend = NormalizeName(backend);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (const std::string& name : registration_names) {
    HYPERVEC_THROW_IF_NOT_MSG(descriptors_.find(name) == descriptors_.end() &&
                                  aliases_.find(name) == aliases_.end(),
                              "index name or alias is already registered");
  }

  const std::string canonical_name = descriptor.canonical_name;
  descriptors_.emplace(canonical_name, std::move(descriptor));
  const IndexDescriptor& stored = descriptors_.at(canonical_name);
  for (const std::string& alias : stored.aliases) {
    aliases_.emplace(alias, canonical_name);
  }
}

std::optional<IndexDescriptor> IndexRegistry::Find(
    std::string_view name) const {
  const std::string normalized = NormalizeName(name);
  std::lock_guard<std::mutex> lock(mutex_);

  auto descriptor = descriptors_.find(normalized);
  if (descriptor != descriptors_.end()) {
    return descriptor->second;
  }

  const auto alias = aliases_.find(normalized);
  if (alias == aliases_.end()) {
    return std::nullopt;
  }
  return descriptors_.at(alias->second);
}

std::vector<IndexDescriptor> IndexRegistry::List() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<IndexDescriptor> descriptors;
  descriptors.reserve(descriptors_.size());
  for (const auto& entry : descriptors_) {
    descriptors.push_back(entry.second);
  }
  std::sort(descriptors.begin(), descriptors.end(),
            [](const IndexDescriptor& lhs, const IndexDescriptor& rhs) {
              return lhs.canonical_name < rhs.canonical_name;
            });
  return descriptors;
}

}  // namespace hypervec
