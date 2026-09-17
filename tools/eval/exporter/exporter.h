/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <memory>
#include <string>

namespace hypervec::eval {

class Exporter {
 public:
  virtual ~Exporter() = default;
  virtual void Export(const std::string& value) const = 0;
};

class ExporterStdout : public Exporter {
 public:
  void Export(const std::string& value) const override;
};

class ExporterFile : public Exporter {
 public:
  explicit ExporterFile(std::string path);
  void Export(const std::string& value) const override;

 protected:
  std::string path_;
};

class ExporterAppendFile : public ExporterFile {
 public:
  explicit ExporterAppendFile(std::string path);
  void Export(const std::string& value) const override;
};

std::unique_ptr<Exporter> MakeExporter(const std::string& path, bool append);

}  // namespace hypervec::eval
