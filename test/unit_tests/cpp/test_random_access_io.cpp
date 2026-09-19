/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <persistence/random_access_io.h>
#include <utils/log/assert.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

class TempFile {
 public:
  explicit TempFile(const std::vector<uint8_t>& data) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t stamp = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    path_ = std::filesystem::temp_directory_path() /
            ("hypervec-random-access-" + std::to_string(stamp) + "-" +
             std::to_string(sequence.fetch_add(1)));
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
    HYPERVEC_THROW_IF_NOT_MSG(output.good(),
                              "TempFile: failed to write test data");
  }

  ~TempFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path& Path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

std::vector<uint8_t> Sequence(size_t count) {
  std::vector<uint8_t> result(count);
  for (size_t index = 0; index < count; ++index) {
    result[index] = static_cast<uint8_t>(index & 0xFFU);
  }
  return result;
}

}  // namespace

TEST(RandomAccessReader, VectorReadsExactRangesAndTracksSuccesses) {
  hypervec::VectorRandomAccessReader reader(Sequence(32));
  std::array<uint8_t, 5> output{};

  reader.ReadAt(7, output.data(), output.size());

  EXPECT_EQ(output, (std::array<uint8_t, 5>{7, 8, 9, 10, 11}));
  EXPECT_EQ(reader.Size(), 32U);
  EXPECT_EQ(reader.Stats().read_operations, 1U);
  EXPECT_EQ(reader.Stats().bytes_read, output.size());
  reader.ReadAt(reader.Size(), nullptr, 0);
  EXPECT_EQ(reader.Stats().read_operations, 1U);
  reader.ResetStats();
  EXPECT_EQ(reader.Stats().read_operations, 0U);
  EXPECT_EQ(reader.Stats().bytes_read, 0U);
}

TEST(RandomAccessReader, RejectsInvalidRangesWithoutChangingCounters) {
  hypervec::VectorRandomAccessReader reader(Sequence(8));
  uint8_t output = 0;

  EXPECT_THROW(reader.ReadAt(0, nullptr, 1), hypervec::HypervecException);
  EXPECT_THROW(reader.ReadAt(9, &output, 0), hypervec::HypervecException);
  EXPECT_THROW(reader.ReadAt(7, &output, 2), hypervec::HypervecException);
  EXPECT_EQ(reader.Stats().read_operations, 0U);
  EXPECT_EQ(reader.Stats().bytes_read, 0U);
}

TEST(RandomAccessReader, ConcurrentVectorReadsAreCursorFree) {
  const std::vector<uint8_t> data = Sequence(1024);
  hypervec::VectorRandomAccessReader reader(data);
  constexpr size_t kThreadCount = 8;
  constexpr size_t kReadsPerThread = 100;
  std::array<std::thread, kThreadCount> threads;
  std::atomic<bool> correct{true};
  for (size_t thread = 0; thread < kThreadCount; ++thread) {
    threads[thread] = std::thread([&, thread] {
      for (size_t read = 0; read < kReadsPerThread; ++read) {
        const size_t offset = (thread * 97 + read * 13) % (data.size() - 8);
        std::array<uint8_t, 8> output{};
        reader.ReadAt(offset, output.data(), output.size());
        if (!std::equal(output.begin(), output.end(), data.begin() + offset)) {
          correct.store(false);
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(correct.load());
  EXPECT_EQ(reader.Stats().read_operations, kThreadCount * kReadsPerThread);
  EXPECT_EQ(reader.Stats().bytes_read,
            kThreadCount * kReadsPerThread * size_t{8});
}

TEST(RandomAccessReader, FileReadsPositionsAndSupportsPrefetchHints) {
  const std::vector<uint8_t> data = Sequence(64);
  const TempFile file(data);
  hypervec::FileRandomAccessReader reader(file.Path().string());
  std::array<uint8_t, 6> tail{};
  std::array<uint8_t, 4> head{};

  reader.ReadAt(58, tail.data(), tail.size());
  reader.ReadAt(2, head.data(), head.size());
  reader.Prefetch(8, 16);
  reader.Prefetch(1000, 16);

  EXPECT_EQ(tail, (std::array<uint8_t, 6>{58, 59, 60, 61, 62, 63}));
  EXPECT_EQ(head, (std::array<uint8_t, 4>{2, 3, 4, 5}));
  EXPECT_EQ(reader.Size(), data.size());
  EXPECT_EQ(reader.Stats().read_operations, 2U);
  EXPECT_EQ(reader.Stats().bytes_read, 10U);
}

TEST(RandomAccessReader, ConcurrentFileReadsAreCursorFree) {
  const std::vector<uint8_t> data = Sequence(2048);
  const TempFile file(data);
  hypervec::FileRandomAccessReader reader(file.Path().string());
  constexpr size_t kThreadCount = 4;
  constexpr size_t kReadsPerThread = 50;
  std::array<std::thread, kThreadCount> threads;
  std::atomic<bool> correct{true};
  for (size_t thread = 0; thread < kThreadCount; ++thread) {
    threads[thread] = std::thread([&, thread] {
      for (size_t read = 0; read < kReadsPerThread; ++read) {
        const size_t offset = (thread * 193 + read * 29) % (data.size() - 16);
        std::array<uint8_t, 16> output{};
        reader.ReadAt(offset, output.data(), output.size());
        if (!std::equal(output.begin(), output.end(), data.begin() + offset)) {
          correct.store(false);
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(correct.load());
  EXPECT_EQ(reader.Stats().read_operations, kThreadCount * kReadsPerThread);
  EXPECT_EQ(reader.Stats().bytes_read,
            kThreadCount * kReadsPerThread * size_t{16});
}

TEST(RandomAccessReader, FileReportsMissingAndTruncatedData) {
  const TempFile missing_base({});
  const std::filesystem::path missing =
      missing_base.Path().string() + ".missing";
  EXPECT_THROW(
      static_cast<void>(hypervec::FileRandomAccessReader(missing.string())),
      hypervec::HypervecException);

  const TempFile file(Sequence(16));
  hypervec::FileRandomAccessReader reader(file.Path().string());
  std::ofstream truncate(file.Path(), std::ios::binary | std::ios::trunc);
  truncate.close();
  std::array<uint8_t, 4> output{};
  EXPECT_THROW(reader.ReadAt(0, output.data(), output.size()),
               hypervec::HypervecException);
  EXPECT_EQ(reader.Stats().read_operations, 0U);
  EXPECT_EQ(reader.Stats().bytes_read, 0U);
}
