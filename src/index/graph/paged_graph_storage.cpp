/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/paged_graph_storage.h>
#include <utils/log/assert.h>

#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace hypervec {
namespace {

void ValidateNodeCount(size_t node_count) {
  constexpr size_t kCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(node_count <= kCapacity,
                            "PagedGraphStorage: node count exceeds GraphId");
}

}  // namespace

PagedGraphStorage::PagedGraphStorage(std::shared_ptr<const PageCache> cache,
                                     size_t node_count, size_t max_degree)
    : cache_(std::move(cache)),
      node_count_(node_count),
      max_degree_(max_degree),
      record_size_(0),
      records_per_page_(0),
      storage_size_(0) {
  HYPERVEC_THROW_IF_NOT_MSG(cache_ != nullptr,
                            "PagedGraphStorage: cache must not be null");
  ValidateNodeCount(node_count_);
  HYPERVEC_THROW_IF_NOT_MSG(max_degree_ > 0,
                            "PagedGraphStorage: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      max_degree_ <= (std::numeric_limits<uint32_t>::max)(),
      "PagedGraphStorage: max_degree exceeds degree representation");

  const size_t neighbor_bytes = mul_no_overflow(
      max_degree_, sizeof(GraphId), "PagedGraphStorage record size");
  record_size_ = add_no_overflow(sizeof(uint32_t), neighbor_bytes,
                                 "PagedGraphStorage record size");
  HYPERVEC_THROW_IF_NOT_MSG(
      record_size_ <= cache_->PageSize(),
      "PagedGraphStorage: one graph record must fit in a page");
  records_per_page_ = cache_->PageSize() / record_size_;

  const size_t page_count = node_count_ / records_per_page_ +
                            (node_count_ % records_per_page_ != 0 ? 1U : 0U);
  HYPERVEC_THROW_IF_NOT_MSG(
      page_count == 0 ||
          cache_->PageSize() <=
              (std::numeric_limits<uint64_t>::max)() / page_count,
      "PagedGraphStorage: storage size overflows");
  storage_size_ = static_cast<uint64_t>(page_count) *
                  static_cast<uint64_t>(cache_->PageSize());
  HYPERVEC_THROW_IF_NOT_MSG(
      cache_->SourceSize() == storage_size_,
      "PagedGraphStorage: data size does not match the declared graph");
}

size_t PagedGraphStorage::NodeCount() const noexcept { return node_count_; }

size_t PagedGraphStorage::MaxDegree() const noexcept { return max_degree_; }

std::pair<uint64_t, size_t> PagedGraphStorage::Locate(GraphId node) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      node >= 0 && static_cast<size_t>(node) < node_count_,
      "PagedGraphStorage::Neighbors: node is outside the graph");
  const size_t index = static_cast<size_t>(node);
  const uint64_t page_id = static_cast<uint64_t>(index / records_per_page_);
  const size_t offset = (index % records_per_page_) * record_size_;
  return {page_id, offset};
}

GraphNeighborList PagedGraphStorage::Neighbors(GraphId node) const {
  const auto location = Locate(node);
  const PageCache::PageHandle page = cache_->Get(location.first);
  HYPERVEC_THROW_IF_NOT_MSG(
      location.second <= page->data.size() &&
          record_size_ <= page->data.size() - location.second,
      "PagedGraphStorage: graph record exceeds its page");

  const uint8_t* record = page->data.data() + location.second;
  uint32_t degree = 0;
  std::memcpy(&degree, record, sizeof(degree));
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<size_t>(degree) <= max_degree_,
      "PagedGraphStorage: stored degree exceeds max_degree");

  const uint8_t* neighbor_bytes = record + sizeof(degree);
  HYPERVEC_THROW_IF_NOT_MSG(
      reinterpret_cast<uintptr_t>(neighbor_bytes) % alignof(GraphId) == 0,
      "PagedGraphStorage: neighbor storage is not aligned");
  const GraphId* neighbors = reinterpret_cast<const GraphId*>(neighbor_bytes);
  for (size_t index = 0; index < degree; ++index) {
    const GraphId neighbor = neighbors[index];
    HYPERVEC_THROW_IF_NOT_MSG(
        neighbor >= 0 && static_cast<size_t>(neighbor) < node_count_,
        "PagedGraphStorage: neighbor is outside the graph");
    HYPERVEC_THROW_IF_NOT_MSG(neighbor != node,
                              "PagedGraphStorage: self loop is not allowed");
    for (size_t previous = 0; previous < index; ++previous) {
      HYPERVEC_THROW_IF_NOT_MSG(
          neighbors[previous] != neighbor,
          "PagedGraphStorage: duplicate neighbors are not allowed");
    }
  }

  std::shared_ptr<const void> owner = page;
  return GraphNeighborList(GraphNeighborView(neighbors, degree),
                           std::move(owner));
}

void PagedGraphStorage::Prefetch(GraphId node) const noexcept {
  if (node < 0 || static_cast<size_t>(node) >= node_count_) {
    return;
  }
  const size_t index = static_cast<size_t>(node);
  cache_->Prefetch(static_cast<uint64_t>(index / records_per_page_));
}

}  // namespace hypervec
