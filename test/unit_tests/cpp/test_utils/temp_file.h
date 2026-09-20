/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <stdexcept>
#include <string>
#include <system_error>

namespace hypervec::test {

/** A test output path isolated inside an exclusively created directory. */
class ScopedTempFile {
 public:
  ScopedTempFile() {
    static std::atomic<uint64_t> sequence{0};
    const auto temp_root = std::filesystem::temp_directory_path();
    const uint64_t stamp = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());

    for (size_t attempt = 0; attempt < 128; ++attempt) {
      const uint64_t suffix = sequence.fetch_add(1, std::memory_order_relaxed);
      const auto candidate =
          temp_root / ("hypervec-test-" + std::to_string(stamp) + "-" +
                       std::to_string(suffix));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        directory_ = candidate;
        path = (directory_ / "payload.bin").string();
        return;
      }
      if (error) {
        throw std::filesystem::filesystem_error(
            "could not create temporary test directory", candidate, error);
      }
    }

    throw std::runtime_error("could not allocate a unique test file path");
  }

  ScopedTempFile(const ScopedTempFile&) = delete;
  ScopedTempFile& operator=(const ScopedTempFile&) = delete;
  ScopedTempFile(ScopedTempFile&&) = delete;
  ScopedTempFile& operator=(ScopedTempFile&&) = delete;

  ~ScopedTempFile() {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  std::string path;

 private:
  std::filesystem::path directory_;
};

}  // namespace hypervec::test
