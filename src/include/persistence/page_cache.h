/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <persistence/random_access_io.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace hypervec {

struct PageCacheStats {
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t coalesced_requests = 0;
  uint64_t pages_loaded = 0;
  uint64_t bytes_loaded = 0;
  uint64_t evictions = 0;
};

/** Thread-safe fixed-size page cache over a RandomAccessReader.
 *
 * The final page may be shorter than PageSize(). Concurrent requests for the
 * same absent page share one physical read, while different pages can load in
 * parallel. Returned handles keep page data alive even after LRU eviction.
 */
class PageCache {
 public:
  struct Page {
    uint64_t id = 0;
    uint64_t offset = 0;
    std::vector<uint8_t> data;
  };

  using PageHandle = std::shared_ptr<const Page>;

  PageCache(std::shared_ptr<const RandomAccessReader> reader, size_t page_size,
            size_t capacity_pages);
  ~PageCache();

  PageCache(const PageCache&) = delete;
  PageCache& operator=(const PageCache&) = delete;

  PageHandle Get(uint64_t page_id) const;

  /** Best-effort hint for a valid page. Invalid page IDs are ignored. */
  void Prefetch(uint64_t page_id) const noexcept;

  /** Drops resident pages. Existing handles and in-flight callers stay valid.
   */
  void Clear();

  PageCacheStats Stats() const noexcept;
  void ResetStats() noexcept;

  size_t PageSize() const noexcept;
  size_t CapacityPages() const noexcept;
  uint64_t PageCount() const noexcept;
  size_t CachedPages() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hypervec
