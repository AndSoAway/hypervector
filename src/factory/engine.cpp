/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/engine.h>
#include <index/idmap/index_id_map.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "name_utils.h"

namespace hypervec {

namespace {

std::string NormalizeStorage(std::string_view storage) {
  const std::string normalized =
      factory_detail::NormalizeName(storage, "storage.type", true);
  if (normalized.empty()) {
    return normalized;
  }
  if (normalized == "flat" || normalized == "raw" || normalized == "fp32") {
    return "flat";
  }
  if (normalized == "pq" || normalized == "product_quantization") {
    return "pq";
  }
  if (normalized == "lvq" || normalized == "local_vector_quantization") {
    return "lvq";
  }
  throw HypervecException("unsupported storage type '" + std::string(storage) +
                          "'");
}

bool SupportsMetric(const IndexDescriptor& descriptor, MetricType metric) {
  return std::find(descriptor.metrics.begin(), descriptor.metrics.end(),
                   metric) != descriptor.metrics.end();
}

bool SupportsBackend(const IndexDescriptor& descriptor,
                     const std::string& backend) {
  return std::find(descriptor.backends.begin(), descriptor.backends.end(),
                   backend) != descriptor.backends.end();
}

struct NormalizedRequest {
  IndexConfig config;
  IndexDescriptorPtr descriptor;
};

NormalizedRequest NormalizeRequest(const IndexConfig& config) {
  IndexConfig normalized = config;
  if (normalized.dimension <= 0 ||
      normalized.dimension > static_cast<idx_t>(INT_MAX)) {
    throw HypervecException("dimension must be in [1, INT_MAX]");
  }

  normalized.backend =
      factory_detail::NormalizeName(normalized.backend, "backend");
  normalized.type = factory_detail::NormalizeName(normalized.type, "type");
  normalized.storage.type = NormalizeStorage(normalized.storage.type);

  // These two family spellings are convenience inputs. Registry entries are
  // always concrete variants, so the remainder of the creation path has no
  // family-specific dispatch.
  if (normalized.type == "hnsw" || normalized.type == "indexhnsw") {
    if (normalized.storage.type.empty()) {
      normalized.storage.type = "flat";
    }
    normalized.type = "hnsw_" + normalized.storage.type;
  } else if (normalized.type == "ivf" || normalized.type == "indexivf") {
    if (normalized.storage.type.empty()) {
      normalized.storage.type = "flat";
    }
    normalized.type = "ivf_" + normalized.storage.type;
  }

  IndexDescriptorPtr descriptor = IndexRegistry::Global().Find(normalized.type);
  if (descriptor == nullptr) {
    throw HypervecException("unknown index type '" + config.type + "'");
  }
  normalized.type = descriptor->canonical_name;

  if (!SupportsMetric(*descriptor, normalized.metric)) {
    throw HypervecException("metric is not supported by index type '" +
                            normalized.type + "'");
  }
  if (!SupportsBackend(*descriptor, normalized.backend)) {
    throw HypervecException("backend '" + normalized.backend +
                            "' is not supported by index type '" +
                            normalized.type + "'");
  }

  if (normalized.metric == kMetricLp) {
    if (!normalized.metric_arg.has_value() ||
        !std::isfinite(*normalized.metric_arg) ||
        *normalized.metric_arg <= 0.0F) {
      throw HypervecException(
          "metric_arg must be finite and positive for the lp metric");
    }
  } else if (normalized.metric_arg.has_value()) {
    throw HypervecException("metric_arg is valid only for the lp metric");
  }

  descriptor->normalize(normalized);
  if (normalized.type != descriptor->canonical_name) {
    throw HypervecException("config normalizer changed the canonical type");
  }

  switch (normalized.id_mode) {
    case IdMode::Sequential:
    case IdMode::External:
      break;
    default:
      throw HypervecException("invalid id_mode enum value");
  }
  return {std::move(normalized), std::move(descriptor)};
}

}  // namespace

Engine::Engine(ResourceContext resources) : resources_(std::move(resources)) {
  RegisterBuiltinIndexes();
}

IndexConfig Engine::NormalizeConfig(const IndexConfig& config) const {
  return NormalizeRequest(config).config;
}

std::unique_ptr<Index> Engine::CreateIndex(const IndexConfig& config) const {
  NormalizedRequest request = NormalizeRequest(config);
  const IndexConfig& normalized = request.config;
  const IndexDescriptor& descriptor = *request.descriptor;

  std::unique_ptr<Index> index = descriptor.creator(normalized, resources_);
  if (index == nullptr) {
    throw HypervecException("index creator returned null for type '" +
                            normalized.type + "'");
  }
  if (index->TypeName() != descriptor.canonical_name) {
    throw HypervecException("index creator returned the wrong concrete type");
  }
  if ((descriptor.guaranteed_capabilities & ~index->Capabilities()) != 0) {
    throw HypervecException(
        "index instance is missing registry-guaranteed capabilities");
  }

  if (normalized.id_mode == IdMode::External) {
    return std::make_unique<IndexIDMap>(std::move(index));
  }
  return index;
}

std::unique_ptr<Index> Engine::CreateIndex(std::string_view json) const {
  return CreateIndex(ParseIndexConfig(json));
}

std::unique_ptr<Index> CreateIndex(std::string_view json) {
  return Engine().CreateIndex(json);
}

}  // namespace hypervec
