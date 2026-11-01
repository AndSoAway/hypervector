/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <persistence/random_access_io.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hypervec {

void RandomAccessReader::ReadAt(uint64_t offset, void* destination,
                                size_t bytes) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      bytes == 0 || destination != nullptr,
      "RandomAccessReader::ReadAt: destination must not be null");
  const uint64_t size = Size();
  HYPERVEC_THROW_IF_NOT_MSG(
      offset <= size,
      "RandomAccessReader::ReadAt: offset is outside the data source");
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<uint64_t>(bytes) <= size - offset,
      "RandomAccessReader::ReadAt: requested range exceeds the data source");
  if (bytes == 0) {
    return;
  }

  ReadAtImpl(offset, destination, bytes);
  read_operations_.fetch_add(1, std::memory_order_relaxed);
  bytes_read_.fetch_add(static_cast<uint64_t>(bytes),
                        std::memory_order_relaxed);
}

void RandomAccessReader::Prefetch(uint64_t /*offset*/,
                                  size_t /*bytes*/) const noexcept {}

RandomAccessReadStats RandomAccessReader::Stats() const noexcept {
  return {read_operations_.load(std::memory_order_relaxed),
          bytes_read_.load(std::memory_order_relaxed)};
}

void RandomAccessReader::ResetStats() noexcept {
  read_operations_.store(0, std::memory_order_relaxed);
  bytes_read_.store(0, std::memory_order_relaxed);
}

VectorRandomAccessReader::VectorRandomAccessReader(std::vector<uint8_t> data)
    : data_(std::move(data)) {}

uint64_t VectorRandomAccessReader::Size() const noexcept {
  return static_cast<uint64_t>(data_.size());
}

void VectorRandomAccessReader::ReadAtImpl(uint64_t offset, void* destination,
                                          size_t bytes) const {
  std::memcpy(destination, data_.data() + static_cast<size_t>(offset), bytes);
}

struct FileRandomAccessReader::Impl {
  explicit Impl(const std::string& filename) {
#if defined(_WIN32)
    descriptor = _open(filename.c_str(), _O_RDONLY | _O_BINARY);
#else
    int flags = O_RDONLY;
#if defined(O_CLOEXEC)
    flags |= O_CLOEXEC;
#endif
    descriptor = open(filename.c_str(), flags);
#endif
    HYPERVEC_THROW_IF_NOT_FMT(descriptor >= 0,
                              "could not open %s for random reading: %s",
                              filename.c_str(), std::strerror(errno));

#if defined(_WIN32)
    struct _stat64 status {};
    const int result = _fstat64(descriptor, &status);
#else
    struct stat status {};
    const int result = fstat(descriptor, &status);
#endif
    if (result != 0 || status.st_size < 0) {
      const int error = errno;
      Close();
      HYPERVEC_THROW_FMT("could not inspect %s for random reading: %s",
                         filename.c_str(), std::strerror(error));
    }
    size = static_cast<uint64_t>(status.st_size);
  }

  ~Impl() { Close(); }

  void Close() noexcept {
    if (descriptor < 0) {
      return;
    }
#if defined(_WIN32)
    _close(descriptor);
#else
    close(descriptor);
#endif
    descriptor = -1;
  }

  int descriptor = -1;
  uint64_t size = 0;
#if defined(_WIN32)
  mutable std::mutex mutex;
#endif
};

FileRandomAccessReader::FileRandomAccessReader(const std::string& filename)
    : impl_(std::make_unique<Impl>(filename)) {}

FileRandomAccessReader::~FileRandomAccessReader() = default;

uint64_t FileRandomAccessReader::Size() const noexcept { return impl_->size; }

void FileRandomAccessReader::ReadAtImpl(uint64_t offset, void* destination,
                                        size_t bytes) const {
  uint8_t* output = static_cast<uint8_t*>(destination);
  size_t completed = 0;
#if defined(_WIN32)
  std::lock_guard<std::mutex> lock(impl_->mutex);
  HYPERVEC_THROW_IF_NOT_MSG(
      offset <= static_cast<uint64_t>((std::numeric_limits<int64_t>::max)()),
      "FileRandomAccessReader::ReadAt: offset exceeds Windows file limits");
  HYPERVEC_THROW_IF_NOT_FMT(
      _lseeki64(impl_->descriptor, static_cast<int64_t>(offset), SEEK_SET) >= 0,
      "FileRandomAccessReader::ReadAt: seek failed: %s", std::strerror(errno));
  while (completed < bytes) {
    const size_t remaining = bytes - completed;
    const unsigned int chunk = static_cast<unsigned int>(std::min(
        remaining, static_cast<size_t>((std::numeric_limits<int>::max)())));
    const int count = _read(impl_->descriptor, output + completed, chunk);
    HYPERVEC_THROW_IF_NOT_FMT(
        count > 0, "FileRandomAccessReader::ReadAt: read failed: %s",
        count == 0 ? "unexpected end of file" : std::strerror(errno));
    completed += static_cast<size_t>(count);
  }
#else
  HYPERVEC_THROW_IF_NOT_MSG(
      offset <= static_cast<uint64_t>((std::numeric_limits<off_t>::max)()),
      "FileRandomAccessReader::ReadAt: offset exceeds POSIX file limits");
  while (completed < bytes) {
    const size_t remaining = bytes - completed;
    const size_t chunk = std::min(
        remaining, static_cast<size_t>((std::numeric_limits<ssize_t>::max)()));
    const uint64_t current_offset = offset + static_cast<uint64_t>(completed);
    HYPERVEC_THROW_IF_NOT_MSG(
        current_offset <=
            static_cast<uint64_t>((std::numeric_limits<off_t>::max)()),
        "FileRandomAccessReader::ReadAt: range exceeds POSIX file limits");
    const ssize_t count = pread(impl_->descriptor, output + completed, chunk,
                                static_cast<off_t>(current_offset));
    if (count < 0 && errno == EINTR) {
      continue;
    }
    HYPERVEC_THROW_IF_NOT_FMT(
        count > 0, "FileRandomAccessReader::ReadAt: read failed: %s",
        count == 0 ? "unexpected end of file" : std::strerror(errno));
    completed += static_cast<size_t>(count);
  }
#endif
}

void FileRandomAccessReader::Prefetch(uint64_t offset,
                                      size_t bytes) const noexcept {
#if defined(POSIX_FADV_WILLNEED)
  if (bytes == 0 || offset >= impl_->size ||
      offset > static_cast<uint64_t>((std::numeric_limits<off_t>::max)())) {
    return;
  }
  const uint64_t available = impl_->size - offset;
  const uint64_t requested = std::min(available, static_cast<uint64_t>(bytes));
  const uint64_t max_length =
      static_cast<uint64_t>((std::numeric_limits<off_t>::max)());
  const off_t length = static_cast<off_t>(std::min(requested, max_length));
  (void)posix_fadvise(impl_->descriptor, static_cast<off_t>(offset), length,
                      POSIX_FADV_WILLNEED);
#else
  (void)offset;
  (void)bytes;
#endif
}

}  // namespace hypervec
