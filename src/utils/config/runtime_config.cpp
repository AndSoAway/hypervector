/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/config/runtime_config.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace hypervec {
namespace {

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

}  // namespace

const std::vector<ConfigOption>& GetConfigOptions() {
  static const std::vector<ConfigOption> options = {
    {"server", "data_root", ConfigValueType::kString, "./data",
     "Root directory for server collection data.", false},
    {"server", "host", ConfigValueType::kString, "127.0.0.1",
     "Server bind host.", false},
    {"server", "port", ConfigValueType::kInt, "8080",
     "Server bind port.", false},
    {"server", "server_mode", ConfigValueType::kString, "http",
     "Startup mode: http, grpc, or dual.", false},
    {"server", "enable_http2", ConfigValueType::kBool, "true",
     "Enable HTTP/2 when the selected ASGI server supports it.", false},
    {"logging", "enable_logging", ConfigValueType::kBool, "true",
     "Global runtime logging switch.", false},
    {"logging", "log_level", ConfigValueType::kString, "info",
     "Global minimum log level.", false},
    {"logging", "log_to_stderr", ConfigValueType::kBool, "true",
     "Write runtime logs to stderr.", false},
    {"logging", "log_to_file", ConfigValueType::kBool, "false",
     "Write runtime logs to a file.", false},
    {"logging", "log_file_path", ConfigValueType::kString,
     "logs/hypervec.log", "Runtime log file path.", true},
    {"index", "default_index_type", ConfigValueType::kString, "HNSWFlat",
     "Default index type used when a collection omits index_params.", false},
    {"index", "default_metric_type", ConfigValueType::kString, "L2",
     "Default metric used when a collection omits metric_type.", false},
  };
  return options;
}

HypervecConfig DefaultRuntimeConfig() {
  return HypervecConfig{};
}

std::string RenderSampleConfig() {
  std::ostringstream out;
  const char* current_section = "";
  for (const auto& option : GetConfigOptions()) {
    if (std::string(current_section) != option.section) {
      if (current_section[0] != '\0') {
        out << "\n";
      }
      current_section = option.section;
      out << "[" << current_section << "]\n";
    }
    out << "# " << option.description << "\n";
    out << option.key << " = " << option.default_value << "\n";
  }
  return out.str();
}

bool ParseIndexType(const std::string& value, IndexType* index_type) {
  const std::string key = Lower(value);
  IndexType parsed = IndexType::kHnswFlat;
  if (key == "flat" || key == "indexflat") {
    parsed = IndexType::kFlat;
  } else if (key == "ivf" || key == "ivfflat" || key == "indexivfflat") {
    parsed = IndexType::kIvfFlat;
  } else if (key == "ivflvq" || key == "indexivflvq") {
    parsed = IndexType::kIvfLvq;
  } else if (key == "ivfpq" || key == "indexivfpq") {
    parsed = IndexType::kIvfPq;
  } else if (key == "hnsw" || key == "hnswflat" ||
             key == "indexhnswflat" || key == "autoindex") {
    parsed = IndexType::kHnswFlat;
  } else if (key == "hnswlvq" || key == "indexhnswlvq") {
    parsed = IndexType::kHnswLvq;
  } else if (key == "hnswpq" || key == "indexhnswpq") {
    parsed = IndexType::kHnswPq;
  } else {
    return false;
  }
  if (index_type != nullptr) {
    *index_type = parsed;
  }
  return true;
}

const char* IndexTypeName(IndexType index_type) {
  switch (index_type) {
    case IndexType::kFlat:
      return "Flat";
    case IndexType::kIvfFlat:
      return "IVFFlat";
    case IndexType::kIvfLvq:
      return "IVFLVQ";
    case IndexType::kIvfPq:
      return "IVFPQ";
    case IndexType::kHnswFlat:
      return "HNSWFlat";
    case IndexType::kHnswLvq:
      return "HNSWLVQ";
    case IndexType::kHnswPq:
      return "HNSWPQ";
  }
  return "HNSWFlat";
}

std::string MetricTypeName(MetricType metric) {
  switch (metric) {
    case kMetricInnerProduct:
      return "IP";
    case kMetricL2:
      return "L2";
    case kMetricL1:
      return "L1";
    case kMetricLinf:
      return "Linf";
    case kMetricLp:
      return "Lp";
    case kMetricCanberra:
      return "Canberra";
    case kMetricBrayCurtis:
      return "BrayCurtis";
    case kMetricJensenShannon:
      return "JensenShannon";
    case kMetricJaccard:
      return "Jaccard";
    case kMetricNaNEuclidean:
      return "NaNEuclidean";
    case kMetricGower:
      return "Gower";
  }
  throw std::invalid_argument("unknown metric type");
}

bool ParseMetricType(const std::string& value, MetricType* metric) {
  const std::string key = Lower(value);
  MetricType parsed = kMetricL2;
  if (key == "ip" || key == "inner_product" || key == "innerproduct" ||
      key == "cosine") {
    parsed = kMetricInnerProduct;
  } else if (key == "l2" || key == "euclidean") {
    parsed = kMetricL2;
  } else if (key == "l1") {
    parsed = kMetricL1;
  } else if (key == "linf" || key == "inf") {
    parsed = kMetricLinf;
  } else if (key == "lp") {
    parsed = kMetricLp;
  } else if (key == "canberra") {
    parsed = kMetricCanberra;
  } else if (key == "braycurtis" || key == "bray_curtis") {
    parsed = kMetricBrayCurtis;
  } else if (key == "jensenshannon" || key == "jensen_shannon") {
    parsed = kMetricJensenShannon;
  } else if (key == "jaccard") {
    parsed = kMetricJaccard;
  } else if (key == "naneuclidean" || key == "nan_euclidean") {
    parsed = kMetricNaNEuclidean;
  } else if (key == "gower") {
    parsed = kMetricGower;
  } else {
    return false;
  }
  if (metric != nullptr) {
    *metric = parsed;
  }
  return true;
}

}  // namespace hypervec
