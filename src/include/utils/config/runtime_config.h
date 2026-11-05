/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/common/platform_macros.h>

#include <string>
#include <vector>

namespace hypervec {

/// Value type of a ConfigOption; used by later parsers to pick a conversion.
enum class ConfigValueType {
  kBool,    ///< "true" or "false".
  kInt,     ///< Signed integer literal.
  kString,  ///< String, possibly empty when `optional` is true.
};

/// System-level logging policy for the process.
///
/// The fields mirror the five logging keys accepted by the legacy Python
/// entry points; no index/algorithm setting belongs here.
struct LoggingConfig {
  bool enable_logging = true;  ///< Master runtime logging switch.
  std::string log_level = "info";  ///< Global minimum level.
  bool log_to_stderr = true;  ///< Write to standard error.
  bool log_to_file = false;  ///< Write to the file sink.
  std::string log_file_path = "logs/hypervec.log";  ///< File sink path.
};

/// System-level server startup settings shared by all transports.
///
/// One listener is modelled here. Per-transport defaults (gRPC/dual) stay in
/// the adapter layer until the server startup chain consumes this snapshot.
struct ServerConfig {
  std::string data_root = "./data";  ///< Collection data root.
  std::string host = "127.0.0.1";  ///< Server bind host.
  int port = 8080;  ///< Server bind port.
};

/// Immutable-by-convention startup snapshot of all system-level settings.
struct HypervecConfig {
  LoggingConfig logging;  ///< Process logging policy.
  ServerConfig server;    ///< Server startup settings.
};

/// Metadata for a single INI key of the runtime configuration model.
///
/// `default_value` is rendered from the corresponding `HypervecConfig` field
/// so the typed model stays the single source of truth. `optional` marks a key
/// that may be omitted or left empty in the INI file: the value is parsed as
/// "unset" and any cross-field requirement (for example `log_file_path` when
/// `log_to_file=true`) is enforced by validation, not by the metadata table.
struct ConfigOption {
  const char* section;  ///< INI section, e.g. "server".
  const char* key;  ///< INI key inside `section`, e.g. "port".
  ConfigValueType type;  ///< Conversion for the raw value.
  std::string default_value;  ///< Default from HypervecConfig.
  const char* description;  ///< Help line for the sample config.
  bool optional;  ///< May be unset or empty; false is required.
};

/// Returns the ordered metadata table for every system-level option.
///
/// The table is static and never mutated after the first call. Returning a
/// reference is intentionally safe under the current build: `hypervec` is a
/// static library, and the SWIG bindings link it into their own shared module,
/// so no `std::vector` crosses a DLL boundary. If the core is ever shipped as
/// a shared library, this should become an out-param or a `std::span`.
HYPERVEC_API const std::vector<ConfigOption>& GetConfigOptions();

/// Returns the typed defaults that back `GetConfigOptions()`.
HYPERVEC_API HypervecConfig DefaultRuntimeConfig();

/// Renders the sample INI document from the metadata table.
HYPERVEC_API std::string RenderSampleConfig();

}  // namespace hypervec
