/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index_type.h>
#include <utils/common/platform_macros.h>
#include <utils/distances/metric_type.h>

#include <string>
#include <vector>

namespace hypervec {

enum class ConfigValueType {
  kBool,
  kInt,
  kString,
};

enum class ConfigSource {
  kDefault,
  kFile,
  kCli,
};

struct LoggingConfig {
  bool enable_logging = true;
  std::string log_level = "info";
  bool log_to_stderr = true;
  bool log_to_file = false;
  std::string log_file_path = "logs/hypervec.log";
};

struct ServerConfig {
  std::string data_root = "./data";
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string server_mode = "http";
  bool enable_http2 = true;
};

struct IndexDefaultsConfig {
  IndexType default_index_type = IndexType::kHnswFlat;
  MetricType default_metric_type = kMetricL2;
};

struct HypervecConfig {
  LoggingConfig logging;
  ServerConfig server;
  IndexDefaultsConfig index_defaults;
};

struct ConfigOption {
  const char* section;
  const char* key;
  ConfigValueType type;
  const char* default_value;
  const char* description;
  bool optional;
};

HYPERVEC_API const std::vector<ConfigOption>& GetConfigOptions();
HYPERVEC_API HypervecConfig DefaultRuntimeConfig();
HYPERVEC_API std::string RenderSampleConfig();
HYPERVEC_API std::string MetricTypeName(MetricType metric);
HYPERVEC_API bool ParseMetricType(const std::string& value,
                                  MetricType* metric);

}  // namespace hypervec
