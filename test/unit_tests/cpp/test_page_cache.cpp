/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <persistence/page_cache.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::vector<uint8_t> Sequence(size_t count) {
  std::vector<uint8_t> result(count);
  for (size_t index = 0; index < count; ++index) {
    result[index] = static_cast<uint8_t>(index & 0xFFU);
  }
  return result;
}

class BlockingReader final : public hypervec::RandomAccessReader {
 public:
  explicit BlockingReader(std::vector<uint8_t> data) : data_(std::move(data)) {}

  uint64_t Size() const noexcept override {
    return static_cast<uint64_t>(data_.size());
  }

  bool WaitForStarts(size_t count) const {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(lock, std::chrono::seconds(2),
                               [this, count] { return started_ >= count; });
  }

  void Release() const {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    condition_.notify_all();
  }

 protected:
  void ReadAtImpl(uint64_t offset, void* destination,
                  size_t bytes) const override {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ++started_;
      condition_.notify_all();
      condition_.wait(lock, [this] { return released_; });
    }
    std::memcpy(destination, data_.data() + static_cast<size_t>(offset), bytes);
  }

 private:
  std::vector<uint8_t> data_;
  mutable std::mutex mutex_;
  mutable std::condition_variable condition_;
  mutable size_t started_ = 0;
  mutable bool released_ = false;
};

class FailOnceReader final : public hypervec::RandomAccessReader {
 public:
  explicit FailOnceReader(std::vector<uint8_t> data) : data_(std::move(data)) {}

  uint64_t Size() const noexcept override {
    return static_cast<uint64_t>(data_.size());
  }

 protected:
  void ReadAtImpl(uint64_t offset, void* destination,
                  size_t bytes) const override {
    if (fail_.exchange(false)) {
      throw hypervec::HypervecException("injected read failure", __func__,
                                        __FILE__, __LINE__);
    }
    std::memcpy(destination, data_.data() + static_cast<size_t>(offset), bytes);
  }

 private:
  std::vector<uint8_t> data_;
  mutable std::atomic<bool> fail_{true};
};

class StartGate {
 public:
  explicit StartGate(size_t participants) : participants_(participants) {}

  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++arrived_;
    condition_.notify_all();
    condition_.wait(lock, [this] { return open_; });
  }

  void OpenWhenReady() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return arrived_ == participants_; });
    open_ = true;
    lock.unlock();
    condition_.notify_all();
  }

 private:
  size_t participants_;
  size_t arrived_ = 0;
  bool open_ = false;
  std::mutex mutex_;
  std::condition_variable condition_;
};

}  // namespace

TEST(PageCache, LoadsPagesAndKeepsTheFinalPageExact) {
  const auto reader =
      std::make_shared<hypervec::VectorRandomAccessReader>(Sequence(10));
  hypervec::PageCache cache(reader, 4, 3);

  const auto middle = cache.Get(1);
  const auto tail = cache.Get(2);
  const auto middle_again = cache.Get(1);

  EXPECT_EQ(cache.PageSize(), 4U);
  EXPECT_EQ(cache.CapacityPages(), 3U);
  EXPECT_EQ(cache.PageCount(), 3U);
  EXPECT_EQ(cache.CachedPages(), 2U);
  EXPECT_EQ(middle->id, 1U);
  EXPECT_EQ(middle->offset, 4U);
  EXPECT_EQ(middle->data, (std::vector<uint8_t>{4, 5, 6, 7}));
  EXPECT_EQ(tail->offset, 8U);
  EXPECT_EQ(tail->data, (std::vector<uint8_t>{8, 9}));
  EXPECT_EQ(middle_again, middle);

  const auto stats = cache.Stats();
  EXPECT_EQ(stats.cache_hits, 1U);
  EXPECT_EQ(stats.cache_misses, 2U);
  EXPECT_EQ(stats.pages_loaded, 2U);
  EXPECT_EQ(stats.bytes_loaded, 6U);
  EXPECT_EQ(reader->Stats().read_operations, 2U);
  EXPECT_EQ(reader->Stats().bytes_read, 6U);
}

TEST(PageCache, EvictsLeastRecentlyUsedPagesWithoutInvalidatingHandles) {
  const auto reader =
      std::make_shared<hypervec::VectorRandomAccessReader>(Sequence(16));
  hypervec::PageCache cache(reader, 4, 2);

  const auto original_zero = cache.Get(0);
  const auto original_one = cache.Get(1);
  EXPECT_EQ(cache.Get(0), original_zero);
  cache.Get(2);
  const auto reloaded_one = cache.Get(1);

  EXPECT_NE(reloaded_one, original_one);
  EXPECT_EQ(original_zero->data, (std::vector<uint8_t>{0, 1, 2, 3}));
  EXPECT_EQ(original_one->data, (std::vector<uint8_t>{4, 5, 6, 7}));
  EXPECT_EQ(cache.CachedPages(), 2U);
  const auto stats = cache.Stats();
  EXPECT_EQ(stats.cache_hits, 1U);
  EXPECT_EQ(stats.cache_misses, 4U);
  EXPECT_EQ(stats.evictions, 2U);
}

TEST(PageCache, RejectsInvalidConfigurationAndPageIds) {
  const auto reader =
      std::make_shared<hypervec::VectorRandomAccessReader>(Sequence(8));
  EXPECT_THROW(hypervec::PageCache(nullptr, 4, 1), hypervec::HypervecException);
  EXPECT_THROW(hypervec::PageCache(reader, 0, 1), hypervec::HypervecException);
  EXPECT_THROW(hypervec::PageCache(reader, 4, 0), hypervec::HypervecException);

  hypervec::PageCache cache(reader, 4, 1);
  EXPECT_THROW(cache.Get(2), hypervec::HypervecException);
  cache.Prefetch(2);

  const auto empty_reader =
      std::make_shared<hypervec::VectorRandomAccessReader>(Sequence(0));
  hypervec::PageCache empty(empty_reader, 4, 1);
  EXPECT_EQ(empty.PageCount(), 0U);
  EXPECT_THROW(empty.Get(0), hypervec::HypervecException);
}

TEST(PageCache, ClearDropsResidencyAndStatsCanBeReset) {
  const auto reader =
      std::make_shared<hypervec::VectorRandomAccessReader>(Sequence(8));
  hypervec::PageCache cache(reader, 4, 2);

  const auto original = cache.Get(0);
  cache.Clear();
  EXPECT_EQ(cache.CachedPages(), 0U);
  const auto reloaded = cache.Get(0);

  EXPECT_NE(original, reloaded);
  EXPECT_EQ(reader->Stats().read_operations, 2U);
  EXPECT_EQ(cache.Stats().cache_misses, 2U);
  cache.ResetStats();
  const auto reset = cache.Stats();
  EXPECT_EQ(reset.cache_hits, 0U);
  EXPECT_EQ(reset.cache_misses, 0U);
  EXPECT_EQ(reset.pages_loaded, 0U);
  EXPECT_EQ(reset.evictions, 0U);
}

TEST(PageCache, CoalescesConcurrentRequestsForTheSamePage) {
  constexpr size_t kThreadCount = 8;
  const auto reader = std::make_shared<BlockingReader>(Sequence(32));
  hypervec::PageCache cache(reader, 8, 2);
  StartGate gate(kThreadCount);
  std::array<std::thread, kThreadCount> threads;
  std::array<hypervec::PageCache::PageHandle, kThreadCount> pages;
  std::atomic<bool> correct{true};

  for (size_t index = 0; index < kThreadCount; ++index) {
    threads[index] = std::thread([&, index] {
      gate.ArriveAndWait();
      try {
        pages[index] = cache.Get(1);
      } catch (...) {
        correct.store(false);
      }
    });
  }
  gate.OpenWhenReady();
  EXPECT_TRUE(reader->WaitForStarts(1));
  reader->Release();
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(correct.load());
  EXPECT_TRUE(std::all_of(pages.begin(), pages.end(), [&](const auto& page) {
    return page == pages.front();
  }));
  const auto stats = cache.Stats();
  EXPECT_EQ(stats.cache_misses, 1U);
  EXPECT_EQ(stats.cache_hits + stats.coalesced_requests, kThreadCount - 1);
  EXPECT_EQ(reader->Stats().read_operations, 1U);
}

TEST(PageCache, LoadsDifferentPagesWithoutSerializingIo) {
  const auto reader = std::make_shared<BlockingReader>(Sequence(32));
  hypervec::PageCache cache(reader, 8, 2);
  std::atomic<bool> correct{true};
  std::thread first([&] {
    try {
      cache.Get(0);
    } catch (...) {
      correct.store(false);
    }
  });
  std::thread second([&] {
    try {
      cache.Get(1);
    } catch (...) {
      correct.store(false);
    }
  });

  const bool both_started = reader->WaitForStarts(2);
  reader->Release();
  first.join();
  second.join();

  EXPECT_TRUE(both_started);
  EXPECT_TRUE(correct.load());
  EXPECT_EQ(reader->Stats().read_operations, 2U);
}

TEST(PageCache, FailedLoadsWakeWaitersAndCanBeRetried) {
  const auto reader = std::make_shared<FailOnceReader>(Sequence(8));
  hypervec::PageCache cache(reader, 4, 1);

  EXPECT_THROW(cache.Get(0), hypervec::HypervecException);
  EXPECT_EQ(cache.CachedPages(), 0U);
  const auto page = cache.Get(0);

  EXPECT_EQ(page->data, (std::vector<uint8_t>{0, 1, 2, 3}));
  EXPECT_EQ(cache.Stats().cache_misses, 2U);
  EXPECT_EQ(cache.Stats().pages_loaded, 1U);
  EXPECT_EQ(reader->Stats().read_operations, 1U);
}

TEST(PageCache, ClearDuringLoadDoesNotRepopulateTheCache) {
  const auto reader = std::make_shared<BlockingReader>(Sequence(8));
  hypervec::PageCache cache(reader, 4, 1);
  hypervec::PageCache::PageHandle loaded;
  std::thread worker([&] { loaded = cache.Get(0); });

  const bool started = reader->WaitForStarts(1);
  cache.Clear();
  reader->Release();
  worker.join();

  ASSERT_TRUE(started);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->data, (std::vector<uint8_t>{0, 1, 2, 3}));
  EXPECT_EQ(cache.CachedPages(), 0U);
  EXPECT_NE(cache.Get(0), loaded);
  EXPECT_EQ(reader->Stats().read_operations, 2U);
}
