/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/config/runtime_config.h>

#include <sstream>
#include <string>

namespace hypervec {

namespace {

std::string ToString(bool value) { return value ? "true" : "false"; }

std::string ToString(int value) { return std::to_string(value); }

}  // namespace

const std::vector<ConfigOption>& GetConfigOptions() {
  // HypervecConfig field defaults are the single source of truth for default
  // values; this metadata table only mirrors them so the sample config and the
  // later INI/CLI layering cannot drift apart.
  static const HypervecConfig defaults;
  static const std::vector<ConfigOption> options = {
    {"server", "data_root", ConfigValueType::kString, defaults.server.data_root,
     "Root directory for server collection data.", false},
    {"server", "host", ConfigValueType::kString, defaults.server.host,
     "Server bind host.", false},
    {"server", "port", ConfigValueType::kInt, ToString(defaults.server.port),
     "Server bind port.", false},
    {"server", "server_mode", ConfigValueType::kString,
     defaults.server.server_mode, "Startup mode: http, grpc, or dual.", false},
    {"server", "enable_http2", ConfigValueType::kBool,
     ToString(defaults.server.enable_http2),
     "Enable HTTP/2 when the selected ASGI server supports it.", false},
    {"logging", "enable_logging", ConfigValueType::kBool,
     ToString(defaults.logging.enable_logging), "Global runtime logging switch.",
     false},
    {"logging", "log_level", ConfigValueType::kString, defaults.logging.log_level,
     "Global minimum log level.", false},
    {"logging", "log_to_stderr", ConfigValueType::kBool,
     ToString(defaults.logging.log_to_stderr), "Write runtime logs to stderr.",
     false},
    {"logging", "log_to_file", ConfigValueType::kBool,
     ToString(defaults.logging.log_to_file), "Write runtime logs to a file.",
     false},
    {"logging", "log_file_path", ConfigValueType::kString,
     defaults.logging.log_file_path, "Runtime log file path.", true},
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

}  // namespace hypervec
