/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/exporter/exporter.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypervec::eval {
namespace {

void WriteFile(const std::string& path, const std::string& value,
               std::ios_base::openmode mode) {
  std::ofstream output(path, mode);
  if (!output) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  output << value;
  if (!output) {
    throw std::runtime_error("failed to write output file: " + path);
  }
}

}  // namespace

void ExporterStdout::Export(const std::string& value) const {
  std::cout << value;
}

ExporterFile::ExporterFile(std::string path) : path_(std::move(path)) {}

void ExporterFile::Export(const std::string& value) const {
  WriteFile(path_, value, std::ios::out | std::ios::trunc);
}

ExporterAppendFile::ExporterAppendFile(std::string path)
    : ExporterFile(std::move(path)) {}

void ExporterAppendFile::Export(const std::string& value) const {
  WriteFile(path_, value, std::ios::out | std::ios::app);
}

std::unique_ptr<Exporter> MakeExporter(const std::string& path, bool append) {
  if (path.empty()) {
    return std::make_unique<ExporterStdout>();
  }
  if (append) {
    return std::make_unique<ExporterAppendFile>(path);
  }
  return std::make_unique<ExporterFile>(path);
}

}  // namespace hypervec::eval
