/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>
#include <persistence/page_cache.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace hypervec {

/** Read-only fixed-record graph stored in page-aligned data.
 *
 * Each record uses native byte order and contains a uint32_t degree followed
 * by MaxDegree() GraphId slots. Records never cross page boundaries. Unused
 * space at the end of every page, including the final page, is padding.
 */
class PagedGraphStorage final : public GraphStorage {
 public:
  PagedGraphStorage(std::shared_ptr<const PageCache> cache, size_t node_count,
                    size_t max_degree);
  PagedGraphStorage(std::shared_ptr<const PageCache> cache, size_t node_count,
                    size_t max_degree, size_t record_size,
                    size_t degree_offset);

  size_t NodeCount() const noexcept override;
  size_t MaxDegree() const noexcept override;
  GraphNeighborList Neighbors(GraphId node) const override;
  void Prefetch(GraphId node) const noexcept override;

  size_t RecordSize() const noexcept { return record_size_; }
  size_t DegreeOffset() const noexcept { return degree_offset_; }
  size_t RecordsPerPage() const noexcept { return records_per_page_; }
  uint64_t StorageSize() const noexcept { return storage_size_; }

 private:
  void ValidateLayout();
  std::pair<uint64_t, size_t> Locate(GraphId node) const;

  std::shared_ptr<const PageCache> cache_;
  size_t node_count_;
  size_t max_degree_;
  size_t record_size_;
  size_t degree_offset_;
  size_t records_per_page_;
  uint64_t storage_size_;
};

}  // namespace hypervec
