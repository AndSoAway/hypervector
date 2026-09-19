/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/diskann/index_diskann.h>
#include <index/hnsw/index_hnsw.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf.h>
#include <index/lsh/index_lsh.h>
#include <index/nsg/index_nsg.h>
#include <index/nsw/index_nsw.h>
#include <index/pretransform/index_pre_transform.h>
#include <index/search_parameters_factory.h>
#include <index/vamana/index_vamana.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

void ValidateParameterName(std::string_view name) {
  HYPERVEC_THROW_IF_NOT_MSG(!name.empty(),
                            "search parameter name must not be empty");
}

const Index& ResolveSearchIndex(const Index& index) {
  const Index* current = &index;
  for (size_t depth = 0; depth < 64; ++depth) {
    if (const auto* mapped = dynamic_cast<const IndexIDMap*>(current)) {
      HYPERVEC_THROW_IF_NOT_MSG(
          mapped->index != nullptr,
          "cannot create search parameters for an empty IndexIDMap");
      current = mapped->index;
      continue;
    }
    if (const auto* transformed =
            dynamic_cast<const IndexPreTransform*>(current)) {
      HYPERVEC_THROW_IF_NOT_MSG(
          transformed->index != nullptr,
          "cannot create search parameters for an empty IndexPreTransform");
      current = transformed->index.get();
      continue;
    }
    return *current;
  }
  HYPERVEC_THROW_MSG("index decorator nesting is too deep");
}

void ValidateNames(const SearchConfig& config,
                   const std::vector<std::string>& allowed) {
  const std::unordered_set<std::string> allowed_names(allowed.begin(),
                                                      allowed.end());
  for (const std::string& name : config.ParameterNames()) {
    HYPERVEC_THROW_IF_NOT_FMT(allowed_names.contains(name),
                              "unknown search parameter '%s'", name.c_str());
  }
}

size_t PositiveSize(const SearchConfig& config, std::string_view name,
                    size_t default_value) {
  if (!config.HasParameter(name)) {
    HYPERVEC_THROW_IF_NOT_FMT(
        default_value > 0,
        "default search parameter '%.*s' must be a positive integer",
        static_cast<int>(name.size()), name.data());
    return default_value;
  }
  const int64_t value = config.GetInteger(name, 0);
  HYPERVEC_THROW_IF_NOT_FMT(
      value > 0 &&
          static_cast<uint64_t>(value) <= (std::numeric_limits<size_t>::max)(),
      "search parameter '%.*s' must be a positive integer",
      static_cast<int>(name.size()), name.data());
  return static_cast<size_t>(value);
}

size_t NonnegativeSize(const SearchConfig& config, std::string_view name,
                       size_t default_value) {
  if (!config.HasParameter(name)) {
    return default_value;
  }
  const int64_t value = config.GetInteger(name, 0);
  HYPERVEC_THROW_IF_NOT_FMT(
      value >= 0 &&
          static_cast<uint64_t>(value) <= (std::numeric_limits<size_t>::max)(),
      "search parameter '%.*s' must be a non-negative integer",
      static_cast<int>(name.size()), name.data());
  return static_cast<size_t>(value);
}

int PositiveInt(const SearchConfig& config, std::string_view name,
                int default_value) {
  const int64_t value = config.GetInteger(name, default_value);
  HYPERVEC_THROW_IF_NOT_FMT(
      value > 0 && value <= (std::numeric_limits<int>::max)(),
      "search parameter '%.*s' must be in [1, %d]",
      static_cast<int>(name.size()), name.data(),
      (std::numeric_limits<int>::max)());
  return static_cast<int>(value);
}

std::string EfParameterName(const SearchConfig& config) {
  HYPERVEC_THROW_IF_NOT_MSG(
      !(config.HasParameter("ef_search") && config.HasParameter("ef")),
      "search parameters 'ef_search' and 'ef' cannot both be set");
  return config.HasParameter("ef") ? "ef" : "ef_search";
}

template <typename Parameters>
std::unique_ptr<SearchParameters> FinishParameters(
    std::unique_ptr<Parameters> parameters, const SearchConfig& config) {
  parameters->sel = config.selector;
  return parameters;
}

std::unique_ptr<SearchParameters> MakeDiskAnnParameters(
    const IndexDiskANN& index, const SearchConfig& config) {
  ValidateNames(config, {"search_width", "check_relative_distance"});
  auto parameters = std::make_unique<SearchParametersDiskANN>();
  parameters->search_width =
      PositiveSize(config, "search_width", index.Options().search_width);
  parameters->check_relative_distance = config.GetBoolean(
      "check_relative_distance", index.Options().check_relative_distance);
  return FinishParameters(std::move(parameters), config);
}

std::unique_ptr<SearchParameters> MakeHnswParameters(
    const IndexHNSW& index, const SearchConfig& config) {
  ValidateNames(
      config, {"ef", "ef_search", "check_relative_distance", "bounded_queue"});
  auto parameters = std::make_unique<SearchParametersHNSW>();
  const std::string ef_name = EfParameterName(config);
  parameters->ef_search = PositiveInt(config, ef_name, index.hnsw.ef_search);
  parameters->check_relative_distance = config.GetBoolean(
      "check_relative_distance", index.hnsw.check_relative_distance);
  parameters->bounded_queue =
      config.GetBoolean("bounded_queue", index.hnsw.search_bounded_queue);
  return FinishParameters(std::move(parameters), config);
}

std::unique_ptr<SearchParameters> MakeNswParameters(
    const IndexNSW& index, const SearchConfig& config) {
  ValidateNames(config, {"ef", "ef_search", "check_relative_distance"});
  auto parameters = std::make_unique<SearchParametersNSW>();
  const std::string ef_name = EfParameterName(config);
  parameters->ef_search =
      PositiveSize(config, ef_name, index.Options().ef_search);
  parameters->check_relative_distance = config.GetBoolean(
      "check_relative_distance", index.Options().check_relative_distance);
  return FinishParameters(std::move(parameters), config);
}

std::unique_ptr<SearchParameters> MakeNsgParameters(
    const IndexNSG& index, const SearchConfig& config) {
  ValidateNames(config, {"ef", "ef_search", "check_relative_distance"});
  auto parameters = std::make_unique<SearchParametersNSG>();
  const std::string ef_name = EfParameterName(config);
  parameters->ef_search =
      PositiveSize(config, ef_name, index.Options().ef_search);
  parameters->check_relative_distance = config.GetBoolean(
      "check_relative_distance", index.Options().check_relative_distance);
  return FinishParameters(std::move(parameters), config);
}

std::unique_ptr<SearchParameters> MakeVamanaParameters(
    const IndexVamana& index, const SearchConfig& config) {
  ValidateNames(config, {"search_width", "check_relative_distance"});
  auto parameters = std::make_unique<SearchParametersVamana>();
  parameters->search_width =
      PositiveSize(config, "search_width", index.Options().search_width);
  parameters->check_relative_distance = config.GetBoolean(
      "check_relative_distance", index.Options().check_relative_distance);
  return FinishParameters(std::move(parameters), config);
}

std::unique_ptr<SearchParameters> MakeLshParameters(
    const SearchConfig& config) {
  ValidateNames(config, {"probe_count", "candidate_limit"});
  auto parameters = std::make_unique<SearchParametersLSH>();
  if (config.HasParameter("probe_count")) {
    parameters->probe_count = PositiveSize(config, "probe_count", 1);
  }
  if (config.HasParameter("candidate_limit")) {
    parameters->candidate_limit = NonnegativeSize(config, "candidate_limit", 0);
  }
  return FinishParameters(std::move(parameters), config);
}

std::unique_ptr<SearchParameters> MakeIvfParameters(
    const IndexIVF& index, const SearchConfig& config) {
  ValidateNames(config, {"nprobe"});
  auto parameters = std::make_unique<IVFSearchParameters>();
  const int64_t nprobe = config.GetInteger("nprobe", index.nprobe);
  HYPERVEC_THROW_IF_NOT_MSG(nprobe > 0,
                            "search parameter 'nprobe' must be positive");
  parameters->nprobe = static_cast<idx_t>(nprobe);
  return FinishParameters(std::move(parameters), config);
}

}  // namespace

void SearchConfig::Set(std::string name, Value value) {
  ValidateParameterName(name);
  auto entry =
      std::find_if(parameters_.begin(), parameters_.end(),
                   [&](const auto& item) { return item.first == name; });
  if (entry == parameters_.end()) {
    parameters_.emplace_back(std::move(name), std::move(value));
  } else {
    entry->second = std::move(value);
  }
}

SearchConfig& SearchConfig::SetInteger(std::string name, int64_t value) {
  Value stored;
  stored.type = ValueType::kInteger;
  stored.integer = value;
  Set(std::move(name), std::move(stored));
  return *this;
}

SearchConfig& SearchConfig::SetBoolean(std::string name, bool value) {
  Value stored;
  stored.type = ValueType::kBoolean;
  stored.boolean = value;
  Set(std::move(name), std::move(stored));
  return *this;
}

const SearchConfig::Value* SearchConfig::Find(std::string_view name) const {
  const auto entry =
      std::find_if(parameters_.begin(), parameters_.end(),
                   [&](const auto& item) { return item.first == name; });
  return entry == parameters_.end() ? nullptr : &entry->second;
}

bool SearchConfig::HasParameter(std::string_view name) const {
  return Find(name) != nullptr;
}

int64_t SearchConfig::GetInteger(std::string_view name,
                                 int64_t default_value) const {
  const Value* value = Find(name);
  if (value == nullptr) {
    return default_value;
  }
  HYPERVEC_THROW_IF_NOT_FMT(value->type == ValueType::kInteger,
                            "search parameter '%.*s' must be an integer",
                            static_cast<int>(name.size()), name.data());
  return value->integer;
}

bool SearchConfig::GetBoolean(std::string_view name, bool default_value) const {
  const Value* value = Find(name);
  if (value == nullptr) {
    return default_value;
  }
  HYPERVEC_THROW_IF_NOT_FMT(value->type == ValueType::kBoolean,
                            "search parameter '%.*s' must be a boolean",
                            static_cast<int>(name.size()), name.data());
  return value->boolean;
}

std::vector<std::string> SearchConfig::ParameterNames() const {
  std::vector<std::string> names;
  names.reserve(parameters_.size());
  for (const auto& [name, value] : parameters_) {
    (void)value;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

SearchParameterDescriptor DescribeSearchParameters(const Index& index) {
  const Index& target = ResolveSearchIndex(index);
  if (dynamic_cast<const IndexDiskANN*>(&target) != nullptr) {
    return {"diskann", {"check_relative_distance", "search_width"}};
  }
  if (dynamic_cast<const IndexHNSW*>(&target) != nullptr) {
    return {"hnsw", {"bounded_queue", "check_relative_distance", "ef_search"}};
  }
  if (dynamic_cast<const IndexNSW*>(&target) != nullptr) {
    return {"nsw", {"check_relative_distance", "ef_search"}};
  }
  if (dynamic_cast<const IndexNSG*>(&target) != nullptr) {
    return {"nsg", {"check_relative_distance", "ef_search"}};
  }
  if (dynamic_cast<const IndexVamana*>(&target) != nullptr) {
    return {"vamana", {"check_relative_distance", "search_width"}};
  }
  if (dynamic_cast<const IndexLSH*>(&target) != nullptr) {
    return {"lsh", {"candidate_limit", "probe_count"}};
  }
  if (dynamic_cast<const IndexIVF*>(&target) != nullptr) {
    return {"ivf", {"nprobe"}};
  }
  return {"generic", {}};
}

std::unique_ptr<SearchParameters> CreateSearchParameters(
    const Index& index, const SearchConfig& config) {
  const Index& target = ResolveSearchIndex(index);
  if (const auto* diskann = dynamic_cast<const IndexDiskANN*>(&target)) {
    return MakeDiskAnnParameters(*diskann, config);
  }
  if (const auto* hnsw = dynamic_cast<const IndexHNSW*>(&target)) {
    return MakeHnswParameters(*hnsw, config);
  }
  if (const auto* nsw = dynamic_cast<const IndexNSW*>(&target)) {
    return MakeNswParameters(*nsw, config);
  }
  if (const auto* nsg = dynamic_cast<const IndexNSG*>(&target)) {
    return MakeNsgParameters(*nsg, config);
  }
  if (const auto* vamana = dynamic_cast<const IndexVamana*>(&target)) {
    return MakeVamanaParameters(*vamana, config);
  }
  if (dynamic_cast<const IndexLSH*>(&target) != nullptr) {
    return MakeLshParameters(config);
  }
  if (const auto* ivf = dynamic_cast<const IndexIVF*>(&target)) {
    return MakeIvfParameters(*ivf, config);
  }
  ValidateNames(config, {});
  auto parameters = std::make_unique<SearchParameters>();
  parameters->sel = config.selector;
  return parameters;
}

}  // namespace hypervec
