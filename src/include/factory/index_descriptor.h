/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <factory/index_config.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hypervec {

/** Placeholder for process-owned resources used by future backends. */
struct ResourceContext {};

using IndexCreator = std::function<std::unique_ptr<Index>(
    const IndexConfig&, const ResourceContext&)>;
using IndexConfigNormalizer = std::function<void(IndexConfig&)>;
using SearchConfigNormalizer =
    std::function<void(const IndexConfig&, SearchConfig&)>;
using SearchParametersCreator = std::function<std::unique_ptr<SearchParameters>(
    const IndexConfig&, const SearchConfig&)>;

/** Registry metadata for one concrete, user-creatable index variant.
 *
 * guaranteed_capabilities is the subset available for every valid config of
 * this descriptor. normalize owns algorithm-specific validation and defaults;
 * keeping it with the creator lets new index types avoid changes in Engine.
 * normalize_search and create_search_parameters provide the equivalent
 * algorithm-specific path for per-search overrides.
 * The created instance remains the authority for dynamic or metric-dependent
 * capabilities.
 */
struct IndexDescriptor {
  std::string canonical_name;
  std::vector<std::string> aliases;
  IndexCapabilities guaranteed_capabilities = 0;
  std::vector<MetricType> metrics;
  std::vector<std::string> backends;
  IndexConfigNormalizer normalize;
  IndexCreator creator;
  SearchConfigNormalizer normalize_search;
  SearchParametersCreator create_search_parameters;
};

using IndexDescriptorPtr = std::shared_ptr<const IndexDescriptor>;

}  // namespace hypervec
