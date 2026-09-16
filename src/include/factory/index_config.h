/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <optional>
#include <string>
#include <string_view>

namespace hypervec {

enum class IdMode {
  Sequential,
  External,
};

struct StorageConfig {
  std::string type;
  std::optional<idx_t> pq_m;
  std::optional<idx_t> nlocal;
  std::optional<int> nbits;
};

struct BuildConfig {
  std::optional<idx_t> nlist;
  std::optional<int> graph_degree;
  std::optional<int> ef_construction;
  std::optional<bool> by_residual;
  std::optional<int> use_precomputed_table;
};

struct SearchConfig {
  std::optional<idx_t> nprobe;
  std::optional<int> ef_search;
};

/** Typed, dependency-free representation of an index creation request.
 *
 * Optional algorithm parameters are left unset by the JSON parser. Engine
 * validation applies defaults only after the concrete registry descriptor is
 * known, so parameters for one algorithm cannot silently affect another.
 */
struct IndexConfig {
  std::string type;
  idx_t dimension = 0;
  MetricType metric = kMetricL2;
  std::optional<float> metric_arg;
  std::string backend = "cpu";
  StorageConfig storage;
  BuildConfig build;
  SearchConfig defaults;
  IdMode id_mode = IdMode::Sequential;
};

/** Parse strict JSON into typed configuration without applying defaults. */
IndexConfig ParseIndexConfig(std::string_view json);

/** Convert a supported metric name to the C++ enum. */
MetricType ParseMetricType(std::string_view name);

/** Stable lower-case name for a metric enum. */
std::string_view MetricTypeName(MetricType metric);

}  // namespace hypervec
