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
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "name_utils.h"

namespace hypervec {

IndexRegistry& IndexRegistry::Global() {
  static IndexRegistry registry;
  return registry;
}

void IndexRegistry::Register(IndexDescriptor descriptor) {
  descriptor.canonical_name = factory_detail::NormalizeName(
      descriptor.canonical_name, "index descriptor canonical name");
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
    alias = factory_detail::NormalizeName(alias, "index descriptor alias");
    HYPERVEC_THROW_IF_NOT_MSG(
        registration_names.insert(alias).second,
        "index descriptor contains a duplicate canonical name or alias");
  }

  for (std::string& backend : descriptor.backends) {
    backend =
        factory_detail::NormalizeName(backend, "index descriptor backend");
  }

  const std::string canonical_name = descriptor.canonical_name;
  const IndexDescriptorPtr stored =
      std::make_shared<const IndexDescriptor>(std::move(descriptor));

  std::lock_guard<std::mutex> lock(mutex_);
  for (const std::string& name : registration_names) {
    HYPERVEC_THROW_IF_NOT_MSG(descriptors_.find(name) == descriptors_.end() &&
                                  aliases_.find(name) == aliases_.end(),
                              "index name or alias is already registered");
  }

  descriptors_.emplace(canonical_name, stored);
  for (const std::string& alias : stored->aliases) {
    aliases_.emplace(alias, canonical_name);
  }
}

IndexDescriptorPtr IndexRegistry::Find(std::string_view name) const {
  const std::string normalized =
      factory_detail::NormalizeName(name, "index name");
  std::lock_guard<std::mutex> lock(mutex_);

  auto descriptor = descriptors_.find(normalized);
  if (descriptor != descriptors_.end()) {
    return descriptor->second;
  }

  const auto alias = aliases_.find(normalized);
  if (alias == aliases_.end()) {
    return nullptr;
  }
  return descriptors_.at(alias->second);
}

std::vector<IndexDescriptorPtr> IndexRegistry::List() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<IndexDescriptorPtr> descriptors;
  descriptors.reserve(descriptors_.size());
  for (const auto& entry : descriptors_) {
    descriptors.push_back(entry.second);
  }
  std::sort(descriptors.begin(), descriptors.end(),
            [](const IndexDescriptorPtr& lhs, const IndexDescriptorPtr& rhs) {
              return lhs->canonical_name < rhs->canonical_name;
            });
  return descriptors;
}

}  // namespace hypervec
