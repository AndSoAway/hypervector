/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hypervec {

/** Complete construction configuration consumed by the index registry.
 *
 * Algorithm-specific values are typed and named. Built-in creators reject
 * unknown parameters so misspelled configuration cannot be silently ignored.
 */
class IndexConfig {
 public:
  IndexConfig() = default;
  IndexConfig(std::string index_type, idx_t dimension,
              MetricType metric_type = kMetricL2);

  std::string index_type = "flat";
  idx_t dimension = 0;
  MetricType metric_type = kMetricL2;
  float metric_arg = 0.0F;
  bool use_id_map = false;

  IndexConfig& SetInteger(std::string name, int64_t value);
  IndexConfig& SetDouble(std::string name, double value);
  IndexConfig& SetBoolean(std::string name, bool value);
  IndexConfig& SetString(std::string name, std::string value);

  bool HasParameter(std::string_view name) const;
  int64_t GetInteger(std::string_view name, int64_t default_value) const;
  double GetDouble(std::string_view name, double default_value) const;
  bool GetBoolean(std::string_view name, bool default_value) const;
  std::string GetString(std::string_view name, std::string default_value) const;
  std::vector<std::string> ParameterNames() const;

 private:
  enum class ValueType { kInteger, kDouble, kBoolean, kString };

  struct Value {
    ValueType type = ValueType::kInteger;
    int64_t integer = 0;
    double floating = 0.0;
    bool boolean = false;
    std::string string;
  };

  const Value* Find(std::string_view name) const;
  void Set(std::string name, Value value);

  std::vector<std::pair<std::string, Value>> parameters_;
};

/** Public metadata for one registered index implementation. */
struct IndexDescriptor {
  std::string name;
  std::vector<std::string> aliases;
  std::vector<std::string> parameter_names;
};

using IndexCreator = std::function<std::unique_ptr<Index>(const IndexConfig&)>;

/** Thread-safe mapping from stable index names to construction callbacks. */
class IndexRegistry {
 public:
  IndexRegistry();
  ~IndexRegistry();

  IndexRegistry(const IndexRegistry&) = delete;
  IndexRegistry& operator=(const IndexRegistry&) = delete;

  void Register(IndexDescriptor descriptor, IndexCreator creator);
  bool Contains(std::string_view name) const;
  std::vector<IndexDescriptor> List() const;
  std::unique_ptr<Index> Create(const IndexConfig& config) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** Process-wide registry pre-populated with HyperVec's built-in indexes. */
IndexRegistry& GetIndexRegistry();

/** Construct an index from one complete configuration object. */
std::unique_ptr<Index> CreateIndex(const IndexConfig& config);

}  // namespace hypervec
