/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/diskann/index_diskann.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/index_factory.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/lsh/index_lsh.h>
#include <index/nsg/index_nsg.h>
#include <index/nsw/index_nsw.h>
#include <index/pretransform/index_pre_transform.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/rabitq/index_ivf_rabitq.h>
#include <transform/opq_matrix.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

std::string NormalizeName(std::string_view name) {
  std::string normalized(name);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  return normalized;
}

void ValidateParameterName(std::string_view name) {
  HYPERVEC_THROW_IF_NOT_MSG(!name.empty(),
                            "index parameter name must not be empty");
}

void ValidateBaseConfig(const IndexConfig& config) {
  HYPERVEC_THROW_IF_NOT_FMT(
      config.dimension > 0 &&
          config.dimension <= std::numeric_limits<int>::max(),
      "index dimension must be in [1, %d], got %" PRId64,
      std::numeric_limits<int>::max(), static_cast<int64_t>(config.dimension));
  MetricTypeFromInt(static_cast<int>(config.metric_type));
  HYPERVEC_THROW_IF_NOT_MSG(
      config.metric_type != kMetricLp ||
          (std::isfinite(config.metric_arg) && config.metric_arg > 0.0F),
      "kMetricLp requires a finite, positive metric_arg");
}

void RequireL2(const IndexConfig& config, std::string_view index_name) {
  HYPERVEC_THROW_IF_NOT_FMT(
      config.metric_type == kMetricL2, "%.*s supports kMetricL2 only",
      static_cast<int>(index_name.size()), index_name.data());
}

void RequireL2OrInnerProduct(const IndexConfig& config,
                             std::string_view index_name) {
  HYPERVEC_THROW_IF_NOT_FMT(
      config.metric_type == kMetricL2 ||
          config.metric_type == kMetricInnerProduct,
      "%.*s supports kMetricL2 and kMetricInnerProduct only",
      static_cast<int>(index_name.size()), index_name.data());
}

idx_t PositiveIndexParameter(const IndexConfig& config, std::string_view name,
                             idx_t default_value) {
  const int64_t value = config.GetInteger(name, default_value);
  HYPERVEC_THROW_IF_NOT_FMT(value > 0,
                            "index parameter '%.*s' must be positive",
                            static_cast<int>(name.size()), name.data());
  return static_cast<idx_t>(value);
}

int PositiveIntParameter(const IndexConfig& config, std::string_view name,
                         int default_value) {
  const int64_t value = config.GetInteger(name, default_value);
  HYPERVEC_THROW_IF_NOT_FMT(
      value > 0 && value <= std::numeric_limits<int>::max(),
      "index parameter '%.*s' must be in [1, %d]",
      static_cast<int>(name.size()), name.data(),
      std::numeric_limits<int>::max());
  return static_cast<int>(value);
}

int HnswDegree(const IndexConfig& config) {
  const int degree = PositiveIntParameter(config, "m_hnsw", 32);
  HYPERVEC_THROW_IF_NOT_FMT(
      degree >= 2, "index parameter 'm_hnsw' must be at least 2, got %d",
      degree);
  return degree;
}

std::unique_ptr<Index> MakeFlat(const IndexConfig& config) {
  if (config.metric_type == kMetricL2) {
    return std::make_unique<IndexFlatL2>(config.dimension);
  }
  if (config.metric_type == kMetricInnerProduct) {
    return std::make_unique<IndexFlatIP>(config.dimension);
  }
  return std::make_unique<IndexFlat>(config.dimension, config.metric_type);
}

std::unique_ptr<Index> MakePQ(const IndexConfig& config) {
  RequireL2(config, "pq");
  const idx_t m_pq = PositiveIndexParameter(config, "m_pq", 8);
  const int nbits = PositiveIntParameter(config, "nbits", 8);
  return std::make_unique<IndexPQ>(config.dimension, m_pq, nbits,
                                   config.metric_type);
}

std::unique_ptr<Index> MakeOPQPQ(const IndexConfig& config) {
  RequireL2(config, "opq_pq");
  const idx_t m_pq = PositiveIndexParameter(config, "m_pq", 8);
  const int nbits = PositiveIntParameter(config, "nbits", 8);
  const int opq_iterations = PositiveIntParameter(config, "opq_iterations", 8);
  auto opq = std::make_unique<OPQMatrix>(config.dimension, m_pq, nbits);
  opq->parameters.iterations = opq_iterations;
  auto pq = std::make_unique<IndexPQ>(config.dimension, m_pq, nbits,
                                      config.metric_type);
  return std::make_unique<IndexPreTransform>(std::move(opq), std::move(pq));
}

std::unique_ptr<Index> MakeLVQ(const IndexConfig& config) {
  RequireL2(config, "lvq");
  const idx_t nlocal = PositiveIndexParameter(config, "nlocal", 16);
  const int nbits = PositiveIntParameter(config, "nbits", 8);
  return std::make_unique<IndexLVQ>(config.dimension, nlocal, nbits,
                                    config.metric_type);
}

std::unique_ptr<Index> MakeIVFFlat(const IndexConfig& config) {
  RequireL2OrInnerProduct(config, "ivf_flat");
  const idx_t nlist = PositiveIndexParameter(config, "nlist", 1024);
  return std::make_unique<IndexIVFFlat>(config.dimension, nlist,
                                        config.metric_type);
}

std::unique_ptr<Index> MakeIVFPQ(const IndexConfig& config) {
  RequireL2(config, "ivf_pq");
  const idx_t nlist = PositiveIndexParameter(config, "nlist", 1024);
  const idx_t m_pq = PositiveIndexParameter(config, "m_pq", 8);
  const int nbits = PositiveIntParameter(config, "nbits", 8);
  return std::make_unique<IndexIVFPQ>(config.dimension, nlist, m_pq, nbits,
                                      config.metric_type);
}

std::unique_ptr<Index> MakeIVFLVQ(const IndexConfig& config) {
  RequireL2(config, "ivf_lvq");
  const idx_t nlist = PositiveIndexParameter(config, "nlist", 1024);
  const idx_t nlocal = PositiveIndexParameter(config, "nlocal", 16);
  const int nbits = PositiveIntParameter(config, "nbits", 8);
  return std::make_unique<IndexIVFLVQ>(config.dimension, nlist, nlocal, nbits,
                                       config.metric_type);
}

std::unique_ptr<Index> MakeIVFRaBitQ(const IndexConfig& config) {
  RequireL2(config, "ivf_rabitq");
  const idx_t nlist = PositiveIndexParameter(config, "nlist", 1024);
  const int64_t random_seed = config.GetInteger(
      "random_seed", static_cast<int64_t>(HYPERVEC_RABITQ_DEFAULT_SEED));
  HYPERVEC_THROW_IF_NOT_MSG(
      random_seed >= 0, "index parameter 'random_seed' must be non-negative");
  const int rotation_rounds = PositiveIntParameter(
      config, "rotation_rounds", HYPERVEC_RABITQ_DEFAULT_ROTATION_ROUNDS);
  return std::make_unique<IndexIVFRaBitQ>(config.dimension, nlist,
                                          static_cast<uint64_t>(random_seed),
                                          rotation_rounds, config.metric_type);
}

std::unique_ptr<Index> MakeHNSWFlat(const IndexConfig& config) {
  const int degree = HnswDegree(config);
  auto index = std::make_unique<IndexHNSWFlat>(
      static_cast<int>(config.dimension), degree, config.metric_type);
  index->storage->metric_arg = config.metric_arg;
  return index;
}

std::unique_ptr<Index> MakeDiskANN(const IndexConfig& config) {
  RequireL2(config, "diskann");
  DiskAnnIndexOptions options;
  options.max_degree =
      static_cast<size_t>(PositiveIntParameter(config, "max_degree", 32));
  options.build_search_width = static_cast<size_t>(
      PositiveIntParameter(config, "build_search_width", 64));
  options.candidate_pool_size = static_cast<size_t>(
      PositiveIntParameter(config, "candidate_pool_size", 200));
  options.alpha = static_cast<float>(config.GetDouble("alpha", 1.2));
  options.build_passes =
      static_cast<size_t>(PositiveIntParameter(config, "build_passes", 2));
  if (config.HasParameter("random_seed")) {
    const int64_t random_seed = config.GetInteger("random_seed", 0);
    HYPERVEC_THROW_IF_NOT_MSG(
        random_seed >= 0, "index parameter 'random_seed' must be non-negative");
    options.random_seed = static_cast<uint64_t>(random_seed);
  }
  options.search_width =
      static_cast<size_t>(PositiveIntParameter(config, "search_width", 64));
  options.check_relative_distance =
      config.GetBoolean("check_relative_distance", true);
  options.page_size =
      static_cast<size_t>(PositiveIntParameter(config, "page_size", 4096));
  options.cache_capacity_pages = static_cast<size_t>(
      PositiveIntParameter(config, "cache_capacity_pages", 1024));
  options.node_data_path = config.GetString("node_data_path", "");
  return std::make_unique<IndexDiskANNFlat>(config.dimension,
                                            config.metric_type, options);
}

std::unique_ptr<Index> MakeNSWFlat(const IndexConfig& config) {
  NSWIndexOptions options;
  options.max_degree =
      static_cast<size_t>(PositiveIntParameter(config, "max_degree", 32));
  options.ef_construction =
      static_cast<size_t>(PositiveIntParameter(config, "ef_construction", 64));
  options.ef_search =
      static_cast<size_t>(PositiveIntParameter(config, "ef_search", 16));
  options.check_relative_distance =
      config.GetBoolean("check_relative_distance", true);
  options.fill_to_max_degree = config.GetBoolean("fill_to_max_degree", true);
  return std::make_unique<IndexNSWFlat>(config.dimension, config.metric_type,
                                        options, config.metric_arg);
}

std::unique_ptr<Index> MakeNSGFlat(const IndexConfig& config) {
  NSGIndexOptions options;
  options.knn_degree =
      static_cast<size_t>(PositiveIntParameter(config, "knn_degree", 64));
  options.nn_descent_iterations = static_cast<size_t>(
      PositiveIntParameter(config, "nn_descent_iterations", 10));
  options.nn_descent_convergence_threshold =
      config.GetDouble("nn_descent_convergence_threshold", 0.001);
  if (config.HasParameter("random_seed")) {
    const int64_t random_seed = config.GetInteger("random_seed", 0);
    HYPERVEC_THROW_IF_NOT_MSG(random_seed >= 0,
                              "index parameter 'random_seed' must be "
                              "non-negative");
    options.random_seed = static_cast<uint64_t>(random_seed);
  }
  options.max_degree =
      static_cast<size_t>(PositiveIntParameter(config, "max_degree", 32));
  options.build_search_width = static_cast<size_t>(
      PositiveIntParameter(config, "build_search_width", 40));
  options.candidate_pool_size = static_cast<size_t>(
      PositiveIntParameter(config, "candidate_pool_size", 200));
  options.ef_search =
      static_cast<size_t>(PositiveIntParameter(config, "ef_search", 40));
  options.check_relative_distance =
      config.GetBoolean("check_relative_distance", true);
  return std::make_unique<IndexNSGFlat>(config.dimension, config.metric_type,
                                        options, config.metric_arg);
}

std::unique_ptr<Index> MakeLSH(const IndexConfig& config) {
  HYPERVEC_THROW_IF_NOT_MSG(config.metric_type == kMetricInnerProduct,
                            "lsh supports kMetricInnerProduct only");
  LSHIndexOptions options;
  options.table_count =
      static_cast<size_t>(PositiveIntParameter(config, "table_count", 8));
  options.bits_per_table =
      static_cast<size_t>(PositiveIntParameter(config, "bits_per_table", 16));
  options.probe_count =
      static_cast<size_t>(PositiveIntParameter(config, "probe_count", 4));
  const int64_t candidate_limit = config.GetInteger("candidate_limit", 0);
  HYPERVEC_THROW_IF_NOT_MSG(
      candidate_limit >= 0 && static_cast<uint64_t>(candidate_limit) <=
                                  std::numeric_limits<size_t>::max(),
      "index parameter 'candidate_limit' must be non-negative and fit in "
      "size_t");
  options.candidate_limit = static_cast<size_t>(candidate_limit);
  if (config.HasParameter("random_seed")) {
    const int64_t random_seed = config.GetInteger("random_seed", 0);
    HYPERVEC_THROW_IF_NOT_MSG(random_seed >= 0,
                              "index parameter 'random_seed' must be "
                              "non-negative");
    options.random_seed = static_cast<uint64_t>(random_seed);
  }
  return std::make_unique<IndexLSH>(config.dimension, config.metric_type,
                                    options);
}

std::unique_ptr<Index> MakeHNSWPQ(const IndexConfig& config) {
  RequireL2(config, "hnsw_pq");
  const idx_t m_pq = PositiveIndexParameter(config, "m_pq", 8);
  HYPERVEC_THROW_IF_NOT_FMT(m_pq <= std::numeric_limits<int>::max(),
                            "index parameter 'm_pq' must be at most %d",
                            std::numeric_limits<int>::max());
  const int nbits = PositiveIntParameter(config, "nbits", 8);
  const int degree = HnswDegree(config);
  return std::make_unique<IndexHNSWPQ>(static_cast<int>(config.dimension),
                                       static_cast<int>(m_pq), nbits, degree,
                                       config.metric_type);
}

std::unique_ptr<Index> MakeHNSWLVQ(const IndexConfig& config) {
  RequireL2(config, "hnsw_lvq");
  const idx_t nlocal = PositiveIndexParameter(config, "nlocal", 16);
  HYPERVEC_THROW_IF_NOT_FMT(nlocal <= std::numeric_limits<int>::max(),
                            "index parameter 'nlocal' must be at most %d",
                            std::numeric_limits<int>::max());
  const int nbits = PositiveIntParameter(config, "nbits", 8);
  const int degree = HnswDegree(config);
  return std::make_unique<IndexHNSWLVQ>(static_cast<int>(config.dimension),
                                        static_cast<int>(nlocal), nbits, degree,
                                        config.metric_type);
}

void RegisterBuiltins(IndexRegistry* registry) {
  registry->Register({"diskann",
                      {"disk_ann", "IndexDiskANN", "IndexDiskANNFlat"},
                      {"max_degree", "build_search_width",
                       "candidate_pool_size", "alpha", "build_passes",
                       "random_seed", "search_width", "check_relative_distance",
                       "page_size", "cache_capacity_pages", "node_data_path"}},
                     MakeDiskANN);
  registry->Register({"flat", {"IndexFlat"}, {}}, MakeFlat);
  registry->Register({"pq", {"IndexPQ"}, {"m_pq", "nbits"}}, MakePQ);
  registry->Register(
      {"opq_pq", {"opqpq", "IndexOPQPQ"}, {"m_pq", "nbits", "opq_iterations"}},
      MakeOPQPQ);
  registry->Register({"lvq", {"IndexLVQ"}, {"nlocal", "nbits"}}, MakeLVQ);
  registry->Register(
      {"ivf_flat", {"ivf", "ivfflat", "IndexIVFFlat"}, {"nlist"}}, MakeIVFFlat);
  registry->Register(
      {"ivf_pq", {"ivfpq", "IndexIVFPQ"}, {"nlist", "m_pq", "nbits"}},
      MakeIVFPQ);
  registry->Register(
      {"ivf_lvq", {"ivflvq", "IndexIVFLVQ"}, {"nlist", "nlocal", "nbits"}},
      MakeIVFLVQ);
  registry->Register({"ivf_rabitq",
                      {"ivfrabitq", "IndexIVFRaBitQ"},
                      {"nlist", "random_seed", "rotation_rounds"}},
                     MakeIVFRaBitQ);
  registry->Register({"hnsw_flat",
                      {"hnsw", "hnswflat", "IndexHNSWFlat", "autoindex"},
                      {"m_hnsw"}},
                     MakeHNSWFlat);
  registry->Register(
      {"hnsw_pq", {"hnswpq", "IndexHNSWPQ"}, {"m_pq", "nbits", "m_hnsw"}},
      MakeHNSWPQ);
  registry->Register(
      {"hnsw_lvq", {"hnswlvq", "IndexHNSWLVQ"}, {"nlocal", "nbits", "m_hnsw"}},
      MakeHNSWLVQ);
  registry->Register({"nsw_flat",
                      {"nsw", "nswflat", "IndexNSWFlat"},
                      {"max_degree", "ef_construction", "ef_search",
                       "check_relative_distance", "fill_to_max_degree"}},
                     MakeNSWFlat);
  registry->Register(
      {"nsg_flat",
       {"nsg", "nsgflat", "IndexNSGFlat"},
       {"knn_degree", "nn_descent_iterations",
        "nn_descent_convergence_threshold", "random_seed", "max_degree",
        "build_search_width", "candidate_pool_size", "ef_search",
        "check_relative_distance"}},
      MakeNSGFlat);
  registry->Register({"lsh",
                      {"IndexLSH"},
                      {"table_count", "bits_per_table", "probe_count",
                       "candidate_limit", "random_seed"}},
                     MakeLSH);
}

}  // namespace

IndexConfig::IndexConfig(std::string index_type, idx_t dimension,
                         MetricType metric_type) {
  this->index_type = std::move(index_type);
  this->dimension = dimension;
  this->metric_type = metric_type;
}

void IndexConfig::Set(std::string name, Value value) {
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

IndexConfig& IndexConfig::SetInteger(std::string name, int64_t value) {
  Value stored;
  stored.type = ValueType::kInteger;
  stored.integer = value;
  Set(std::move(name), std::move(stored));
  return *this;
}

IndexConfig& IndexConfig::SetDouble(std::string name, double value) {
  Value stored;
  stored.type = ValueType::kDouble;
  stored.floating = value;
  Set(std::move(name), std::move(stored));
  return *this;
}

IndexConfig& IndexConfig::SetBoolean(std::string name, bool value) {
  Value stored;
  stored.type = ValueType::kBoolean;
  stored.boolean = value;
  Set(std::move(name), std::move(stored));
  return *this;
}

IndexConfig& IndexConfig::SetString(std::string name, std::string value) {
  Value stored;
  stored.type = ValueType::kString;
  stored.string = std::move(value);
  Set(std::move(name), std::move(stored));
  return *this;
}

const IndexConfig::Value* IndexConfig::Find(std::string_view name) const {
  const auto entry =
      std::find_if(parameters_.begin(), parameters_.end(),
                   [&](const auto& item) { return item.first == name; });
  return entry == parameters_.end() ? nullptr : &entry->second;
}

bool IndexConfig::HasParameter(std::string_view name) const {
  return Find(name) != nullptr;
}

int64_t IndexConfig::GetInteger(std::string_view name,
                                int64_t default_value) const {
  const Value* value = Find(name);
  if (value == nullptr) {
    return default_value;
  }
  HYPERVEC_THROW_IF_NOT_FMT(value->type == ValueType::kInteger,
                            "index parameter '%.*s' must be an integer",
                            static_cast<int>(name.size()), name.data());
  return value->integer;
}

double IndexConfig::GetDouble(std::string_view name,
                              double default_value) const {
  const Value* value = Find(name);
  if (value == nullptr) {
    return default_value;
  }
  HYPERVEC_THROW_IF_NOT_FMT(value->type == ValueType::kDouble,
                            "index parameter '%.*s' must be a double",
                            static_cast<int>(name.size()), name.data());
  return value->floating;
}

bool IndexConfig::GetBoolean(std::string_view name, bool default_value) const {
  const Value* value = Find(name);
  if (value == nullptr) {
    return default_value;
  }
  HYPERVEC_THROW_IF_NOT_FMT(value->type == ValueType::kBoolean,
                            "index parameter '%.*s' must be a boolean",
                            static_cast<int>(name.size()), name.data());
  return value->boolean;
}

std::string IndexConfig::GetString(std::string_view name,
                                   std::string default_value) const {
  const Value* value = Find(name);
  if (value == nullptr) {
    return default_value;
  }
  HYPERVEC_THROW_IF_NOT_FMT(value->type == ValueType::kString,
                            "index parameter '%.*s' must be a string",
                            static_cast<int>(name.size()), name.data());
  return value->string;
}

std::vector<std::string> IndexConfig::ParameterNames() const {
  std::vector<std::string> names;
  names.reserve(parameters_.size());
  for (const auto& [name, value] : parameters_) {
    (void)value;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

struct IndexRegistry::Impl {
  struct Entry {
    IndexDescriptor descriptor;
    IndexCreator creator;
  };

  mutable std::shared_mutex mutex;
  std::map<std::string, Entry> entries;
  std::unordered_map<std::string, std::string> names;
};

IndexRegistry::IndexRegistry() : impl_(std::make_unique<Impl>()) {}

IndexRegistry::~IndexRegistry() = default;

void IndexRegistry::Register(IndexDescriptor descriptor, IndexCreator creator) {
  HYPERVEC_THROW_IF_NOT_MSG(!descriptor.name.empty(),
                            "registered index name must not be empty");
  HYPERVEC_THROW_IF_NOT_MSG(static_cast<bool>(creator),
                            "registered index creator must not be empty");

  const std::string canonical = NormalizeName(descriptor.name);
  std::vector<std::string> normalized_names = {canonical};
  normalized_names.reserve(descriptor.aliases.size() + 1);
  for (const std::string& alias : descriptor.aliases) {
    HYPERVEC_THROW_IF_NOT_MSG(!alias.empty(),
                              "registered index alias must not be empty");
    normalized_names.push_back(NormalizeName(alias));
  }

  std::unordered_set<std::string> unique_names;
  for (const std::string& name : normalized_names) {
    HYPERVEC_THROW_IF_NOT_FMT(unique_names.insert(name).second,
                              "duplicate name '%s' within index registration",
                              name.c_str());
  }
  for (const std::string& parameter : descriptor.parameter_names) {
    ValidateParameterName(parameter);
  }
  std::sort(descriptor.parameter_names.begin(),
            descriptor.parameter_names.end());
  HYPERVEC_THROW_IF_NOT_MSG(
      std::adjacent_find(descriptor.parameter_names.begin(),
                         descriptor.parameter_names.end()) ==
          descriptor.parameter_names.end(),
      "registered index parameter names must be unique");

  std::unique_lock lock(impl_->mutex);
  for (const std::string& name : normalized_names) {
    HYPERVEC_THROW_IF_NOT_FMT(impl_->names.find(name) == impl_->names.end(),
                              "index name or alias '%s' is already registered",
                              name.c_str());
  }

  descriptor.name = canonical;
  impl_->entries.emplace(
      canonical, Impl::Entry{std::move(descriptor), std::move(creator)});
  for (const std::string& name : normalized_names) {
    impl_->names.emplace(name, canonical);
  }
}

bool IndexRegistry::Contains(std::string_view name) const {
  const std::string normalized = NormalizeName(name);
  std::shared_lock lock(impl_->mutex);
  return impl_->names.find(normalized) != impl_->names.end();
}

std::vector<IndexDescriptor> IndexRegistry::List() const {
  std::shared_lock lock(impl_->mutex);
  std::vector<IndexDescriptor> descriptors;
  descriptors.reserve(impl_->entries.size());
  for (const auto& [name, entry] : impl_->entries) {
    (void)name;
    descriptors.push_back(entry.descriptor);
  }
  return descriptors;
}

std::unique_ptr<Index> IndexRegistry::Create(const IndexConfig& config) const {
  ValidateBaseConfig(config);
  const std::string requested_name = NormalizeName(config.index_type);

  IndexCreator creator;
  std::vector<std::string> parameter_names;
  {
    std::shared_lock lock(impl_->mutex);
    const auto name = impl_->names.find(requested_name);
    HYPERVEC_THROW_IF_NOT_FMT(name != impl_->names.end(),
                              "unknown index type '%s'",
                              config.index_type.c_str());
    const auto entry = impl_->entries.find(name->second);
    creator = entry->second.creator;
    parameter_names = entry->second.descriptor.parameter_names;
  }

  for (const std::string& parameter : config.ParameterNames()) {
    HYPERVEC_THROW_IF_NOT_FMT(
        std::binary_search(parameter_names.begin(), parameter_names.end(),
                           parameter),
        "unknown parameter '%s' for index type '%s'", parameter.c_str(),
        config.index_type.c_str());
  }

  std::unique_ptr<Index> index = creator(config);
  HYPERVEC_THROW_IF_NOT_FMT(index != nullptr,
                            "creator for index type '%s' returned null",
                            config.index_type.c_str());
  index->metric_arg = config.metric_arg;
  if (!config.use_id_map) {
    return index;
  }

  auto mapped = std::make_unique<IndexIDMap>(index.get());
  mapped->own_fields = true;
  index.release();
  return mapped;
}

IndexRegistry& GetIndexRegistry() {
  static IndexRegistry registry;
  static const bool initialized = [] {
    RegisterBuiltins(&registry);
    return true;
  }();
  (void)initialized;
  return registry;
}

std::unique_ptr<Index> CreateIndex(const IndexConfig& config) {
  return GetIndexRegistry().Create(config);
}

}  // namespace hypervec
