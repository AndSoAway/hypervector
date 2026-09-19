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
#include <span>
#include <utility>

namespace hypervec {

struct IOWriter;

struct DiskAnnNodeLocation {
  uint64_t page_id = 0;
  size_t offset = 0;
};

/** Fixed page layout for DiskANN node payloads.
 *
 * A record contains a native-order float32 vector, a uint32_t degree, and
 * MaxDegree() GraphId slots. Records are packed without crossing page
 * boundaries and every page is written in full.
 */
class DiskAnnNodeLayout {
 public:
  DiskAnnNodeLayout(size_t node_count, size_t dimension, size_t max_degree,
                    size_t page_size);

  size_t NodeCount() const noexcept { return node_count_; }
  size_t Dimension() const noexcept { return dimension_; }
  size_t MaxDegree() const noexcept { return max_degree_; }
  size_t PageSize() const noexcept { return page_size_; }
  size_t VectorBytes() const noexcept { return vector_bytes_; }
  size_t DegreeOffset() const noexcept { return vector_bytes_; }
  size_t RecordSize() const noexcept { return record_size_; }
  size_t RecordsPerPage() const noexcept { return records_per_page_; }
  uint64_t PageCount() const noexcept { return page_count_; }
  uint64_t StorageSize() const noexcept { return storage_size_; }

  DiskAnnNodeLocation Locate(GraphId node) const;

 private:
  size_t node_count_;
  size_t dimension_;
  size_t max_degree_;
  size_t page_size_;
  size_t vector_bytes_;
  size_t record_size_;
  size_t records_per_page_;
  uint64_t page_count_;
  uint64_t storage_size_;
};

/** Float vector view that pins its page while the view is in use. */
class DiskAnnVectorView {
 public:
  using const_iterator = std::span<const float>::iterator;

  DiskAnnVectorView() noexcept = default;
  DiskAnnVectorView(std::span<const float> vector,
                    std::shared_ptr<const void> owner) noexcept
      : vector_(vector), owner_(std::move(owner)) {}

  const float* data() const noexcept { return vector_.data(); }
  size_t size() const noexcept { return vector_.size(); }
  const_iterator begin() const noexcept { return vector_.begin(); }
  const_iterator end() const noexcept { return vector_.end(); }
  const float& operator[](size_t index) const noexcept {
    return vector_[index];
  }

  operator std::span<const float>() const noexcept { return vector_; }

 private:
  std::span<const float> vector_;
  std::shared_ptr<const void> owner_;
};

/** Random-access raw vectors sharing pages with the disk graph. */
class PagedVectorStorage {
 public:
  PagedVectorStorage(std::shared_ptr<const PageCache> cache,
                     DiskAnnNodeLayout layout);

  size_t NodeCount() const noexcept { return layout_.NodeCount(); }
  size_t Dimension() const noexcept { return layout_.Dimension(); }
  DiskAnnVectorView Vector(GraphId node) const;
  void Prefetch(GraphId node) const noexcept;

 private:
  std::shared_ptr<const PageCache> cache_;
  DiskAnnNodeLayout layout_;
};

/** Write validated vectors and adjacency into a page-aligned node payload. */
void WriteDiskAnnNodes(const DiskAnnNodeLayout& layout, const float* vectors,
                       const GraphStorage& graph, IOWriter* writer);

}  // namespace hypervec
