/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hypervec {

struct RandomAccessReadStats {
  uint64_t read_operations = 0;
  uint64_t bytes_read = 0;
};

/** Thread-safe positional reader for page-oriented index data.
 *
 * ReadAt has exact-read semantics: the complete requested range is copied or
 * an exception is thrown. Successful non-empty reads update the counters.
 * Unlike IOReader, this contract has no shared sequential cursor.
 */
class RandomAccessReader {
 public:
  RandomAccessReader() = default;
  virtual ~RandomAccessReader() = default;

  RandomAccessReader(const RandomAccessReader&) = delete;
  RandomAccessReader& operator=(const RandomAccessReader&) = delete;

  virtual uint64_t Size() const noexcept = 0;

  void ReadAt(uint64_t offset, void* destination, size_t bytes) const;

  /** Best-effort hint. Invalid or empty ranges are ignored. */
  virtual void Prefetch(uint64_t offset, size_t bytes) const noexcept;

  RandomAccessReadStats Stats() const noexcept;
  void ResetStats() noexcept;

 protected:
  virtual void ReadAtImpl(uint64_t offset, void* destination,
                          size_t bytes) const = 0;

 private:
  mutable std::atomic<uint64_t> read_operations_{0};
  mutable std::atomic<uint64_t> bytes_read_{0};
};

/** Owning in-memory implementation used by tests and embedded indexes. */
class VectorRandomAccessReader final : public RandomAccessReader {
 public:
  explicit VectorRandomAccessReader(std::vector<uint8_t> data);

  uint64_t Size() const noexcept override;
  const std::vector<uint8_t>& Data() const noexcept { return data_; }

 protected:
  void ReadAtImpl(uint64_t offset, void* destination,
                  size_t bytes) const override;

 private:
  std::vector<uint8_t> data_;
};

/** File-backed positional reader.
 *
 * POSIX builds use pread. Windows builds serialize seek/read operations on an
 * owned descriptor so callers still observe cursor-free thread-safe reads.
 */
class FileRandomAccessReader final : public RandomAccessReader {
 public:
  explicit FileRandomAccessReader(const std::string& filename);
  ~FileRandomAccessReader() override;

  uint64_t Size() const noexcept override;
  void Prefetch(uint64_t offset, size_t bytes) const noexcept override;

 protected:
  void ReadAtImpl(uint64_t offset, void* destination,
                  size_t bytes) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hypervec
