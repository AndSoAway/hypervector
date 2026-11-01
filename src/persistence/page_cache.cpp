/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <persistence/page_cache.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace hypervec {

struct PageCache::Impl {
  using PageHandle = PageCache::PageHandle;

  struct Entry {
    PageHandle page;
    std::list<uint64_t>::iterator lru_position;
  };

  struct PendingLoad {
    std::condition_variable completed;
    PageHandle page;
    std::exception_ptr error;
    uint64_t generation = 0;
    bool ready = false;
  };

  Impl(std::shared_ptr<const RandomAccessReader> source, size_t requested_size,
       size_t requested_capacity)
      : reader(std::move(source)),
        page_size(requested_size),
        capacity_pages(requested_capacity) {
    HYPERVEC_THROW_IF_NOT_MSG(reader != nullptr,
                              "PageCache: reader must not be null");
    HYPERVEC_THROW_IF_NOT_MSG(page_size > 0,
                              "PageCache: page size must be positive");
    HYPERVEC_THROW_IF_NOT_MSG(capacity_pages > 0,
                              "PageCache: capacity must be positive");
    HYPERVEC_THROW_IF_NOT_MSG(
        page_size <= (std::numeric_limits<uint64_t>::max)(),
        "PageCache: page size exceeds supported range");

    source_size = reader->Size();
    const uint64_t size = static_cast<uint64_t>(page_size);
    page_count = source_size / size + (source_size % size != 0 ? 1U : 0U);
  }

  std::pair<uint64_t, size_t> PageRange(uint64_t page_id) const {
    HYPERVEC_THROW_IF_NOT_MSG(page_id < page_count,
                              "PageCache::Get: page ID is out of range");
    const uint64_t size = static_cast<uint64_t>(page_size);
    HYPERVEC_THROW_IF_NOT_MSG(
        page_id <= (std::numeric_limits<uint64_t>::max)() / size,
        "PageCache::Get: page offset overflows");
    const uint64_t offset = page_id * size;
    const size_t bytes = static_cast<size_t>(
        std::min(size, static_cast<uint64_t>(source_size - offset)));
    return {offset, bytes};
  }

  void CompleteFailure(uint64_t page_id,
                       const std::shared_ptr<PendingLoad>& pending,
                       std::exception_ptr error) const noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex);
      pending->error = std::move(error);
      pending->ready = true;
      const auto position = in_flight.find(page_id);
      if (position != in_flight.end() && position->second == pending) {
        in_flight.erase(position);
      }
    }
    pending->completed.notify_all();
  }

  std::shared_ptr<const RandomAccessReader> reader;
  size_t page_size;
  size_t capacity_pages;
  uint64_t source_size = 0;
  uint64_t page_count = 0;

  mutable std::mutex mutex;
  mutable std::list<uint64_t> lru;
  mutable std::unordered_map<uint64_t, Entry> resident;
  mutable std::unordered_map<uint64_t, std::shared_ptr<PendingLoad>> in_flight;
  mutable PageCacheStats stats;
  mutable uint64_t generation = 0;
};

PageCache::PageCache(std::shared_ptr<const RandomAccessReader> reader,
                     size_t page_size, size_t capacity_pages)
    : impl_(new Impl(std::move(reader), page_size, capacity_pages)) {}

PageCache::~PageCache() = default;

PageCache::PageHandle PageCache::Get(uint64_t page_id) const {
  const auto range = impl_->PageRange(page_id);
  std::shared_ptr<Impl::PendingLoad> pending;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const auto resident = impl_->resident.find(page_id);
    if (resident != impl_->resident.end()) {
      ++impl_->stats.cache_hits;
      impl_->lru.splice(impl_->lru.begin(), impl_->lru,
                        resident->second.lru_position);
      return resident->second.page;
    }

    const auto in_flight = impl_->in_flight.find(page_id);
    if (in_flight != impl_->in_flight.end()) {
      ++impl_->stats.coalesced_requests;
      pending = in_flight->second;
      pending->completed.wait(lock, [&pending] { return pending->ready; });
      if (pending->error != nullptr) {
        std::rethrow_exception(pending->error);
      }
      return pending->page;
    }

    ++impl_->stats.cache_misses;
    pending = std::make_shared<Impl::PendingLoad>();
    pending->generation = impl_->generation;
    impl_->in_flight.emplace(page_id, pending);
  }

  PageHandle loaded;
  try {
    auto page = std::make_shared<Page>();
    page->id = page_id;
    page->offset = range.first;
    page->data.resize(range.second);
    impl_->reader->ReadAt(range.first, page->data.data(), page->data.size());
    loaded = std::move(page);

    std::lock_guard<std::mutex> lock(impl_->mutex);
    ++impl_->stats.pages_loaded;
    impl_->stats.bytes_loaded += static_cast<uint64_t>(range.second);
    if (pending->generation == impl_->generation) {
      impl_->lru.push_front(page_id);
      try {
        impl_->resident.emplace(page_id,
                                Impl::Entry{loaded, impl_->lru.begin()});
      } catch (...) {
        impl_->lru.pop_front();
        throw;
      }
      while (impl_->resident.size() > impl_->capacity_pages) {
        const uint64_t evicted = impl_->lru.back();
        impl_->lru.pop_back();
        impl_->resident.erase(evicted);
        ++impl_->stats.evictions;
      }
    }
    pending->page = loaded;
    pending->ready = true;
    impl_->in_flight.erase(page_id);
  } catch (...) {
    impl_->CompleteFailure(page_id, pending, std::current_exception());
    throw;
  }

  pending->completed.notify_all();
  return loaded;
}

void PageCache::Prefetch(uint64_t page_id) const noexcept {
  if (page_id >= impl_->page_count) {
    return;
  }
  const uint64_t size = static_cast<uint64_t>(impl_->page_size);
  const uint64_t offset = page_id * size;
  const size_t bytes = static_cast<size_t>(
      std::min(size, static_cast<uint64_t>(impl_->source_size - offset)));
  impl_->reader->Prefetch(offset, bytes);
}

void PageCache::Clear() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->resident.clear();
  impl_->lru.clear();
  ++impl_->generation;
}

PageCacheStats PageCache::Stats() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

void PageCache::ResetStats() noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->stats = {};
}

size_t PageCache::PageSize() const noexcept { return impl_->page_size; }

size_t PageCache::CapacityPages() const noexcept {
  return impl_->capacity_pages;
}

uint64_t PageCache::SourceSize() const noexcept { return impl_->source_size; }

uint64_t PageCache::PageCount() const noexcept { return impl_->page_count; }

size_t PageCache::CachedPages() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->resident.size();
}

}  // namespace hypervec
