/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/diskann/diskann_storage.h>
#include <persistence/io.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

void ValidateNodeCount(size_t node_count, const char* operation) {
  constexpr size_t kCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_FMT(node_count <= kCapacity,
                            "%s: node count exceeds GraphId capacity",
                            operation);
}

void WriteExact(IOWriter* writer, const uint8_t* data, size_t bytes) {
  size_t completed = 0;
  while (completed < bytes) {
    const size_t written = (*writer)(data + completed, 1, bytes - completed);
    HYPERVEC_THROW_IF_NOT_MSG(
        written > 0 && written <= bytes - completed,
        "WriteDiskAnnNodes: writer failed to make valid progress");
    completed += written;
  }
}

void ValidateGraph(const DiskAnnNodeLayout& layout, const GraphStorage& graph) {
  HYPERVEC_THROW_IF_NOT_MSG(
      graph.NodeCount() == layout.NodeCount(),
      "WriteDiskAnnNodes: graph node count does not match the layout");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph.MaxDegree() <= layout.MaxDegree(),
      "WriteDiskAnnNodes: graph max degree exceeds the layout");
  for (size_t node = 0; node < graph.NodeCount(); ++node) {
    const GraphId graph_node = static_cast<GraphId>(node);
    const GraphNeighborList neighbors = graph.Neighbors(graph_node);
    HYPERVEC_THROW_IF_NOT_MSG(
        neighbors.size() <= layout.MaxDegree(),
        "WriteDiskAnnNodes: neighbor count exceeds the layout");
    std::unordered_set<GraphId> seen;
    seen.reserve(neighbors.size());
    for (GraphId neighbor : neighbors) {
      HYPERVEC_THROW_IF_NOT_MSG(
          neighbor >= 0 && static_cast<size_t>(neighbor) < graph.NodeCount(),
          "WriteDiskAnnNodes: neighbor is outside the graph");
      HYPERVEC_THROW_IF_NOT_MSG(
          neighbor != graph_node,
          "WriteDiskAnnNodes: self loops are not allowed");
      HYPERVEC_THROW_IF_NOT_MSG(
          seen.insert(neighbor).second,
          "WriteDiskAnnNodes: duplicate neighbors are not allowed");
    }
  }
}

}  // namespace

DiskAnnNodeLayout::DiskAnnNodeLayout(size_t node_count, size_t dimension,
                                     size_t max_degree, size_t page_size)
    : node_count_(node_count),
      dimension_(dimension),
      max_degree_(max_degree),
      page_size_(page_size),
      vector_bytes_(0),
      record_size_(0),
      records_per_page_(0),
      page_count_(0),
      storage_size_(0) {
  ValidateNodeCount(node_count_, "DiskAnnNodeLayout");
  HYPERVEC_THROW_IF_NOT_MSG(dimension_ > 0,
                            "DiskAnnNodeLayout: dimension must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(max_degree_ > 0,
                            "DiskAnnNodeLayout: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(page_size_ > 0,
                            "DiskAnnNodeLayout: page_size must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      max_degree_ <= (std::numeric_limits<uint32_t>::max)(),
      "DiskAnnNodeLayout: max_degree exceeds degree representation");

  vector_bytes_ = mul_no_overflow(dimension_, sizeof(float),
                                  "DiskAnnNodeLayout vector bytes");
  mul_no_overflow(node_count_, dimension_,
                  "DiskAnnNodeLayout vector element count");
  const size_t graph_bytes =
      add_no_overflow(sizeof(uint32_t),
                      mul_no_overflow(max_degree_, sizeof(GraphId),
                                      "DiskAnnNodeLayout neighbor bytes"),
                      "DiskAnnNodeLayout graph bytes");
  record_size_ = add_no_overflow(vector_bytes_, graph_bytes,
                                 "DiskAnnNodeLayout record size");
  HYPERVEC_THROW_IF_NOT_MSG(
      record_size_ <= page_size_,
      "DiskAnnNodeLayout: one node record must fit in a page");
  records_per_page_ = page_size_ / record_size_;
  page_count_ = static_cast<uint64_t>(node_count_ / records_per_page_) +
                (node_count_ % records_per_page_ != 0 ? 1U : 0U);
  HYPERVEC_THROW_IF_NOT_MSG(
      page_count_ == 0 ||
          page_size_ <= (std::numeric_limits<uint64_t>::max)() / page_count_,
      "DiskAnnNodeLayout: storage size overflows");
  storage_size_ = page_count_ * static_cast<uint64_t>(page_size_);
}

DiskAnnNodeLocation DiskAnnNodeLayout::Locate(GraphId node) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      node >= 0 && static_cast<size_t>(node) < node_count_,
      "DiskAnnNodeLayout::Locate: node is outside the layout");
  const size_t index = static_cast<size_t>(node);
  return {static_cast<uint64_t>(index / records_per_page_),
          (index % records_per_page_) * record_size_};
}

PagedVectorStorage::PagedVectorStorage(std::shared_ptr<const PageCache> cache,
                                       DiskAnnNodeLayout layout)
    : cache_(std::move(cache)), layout_(std::move(layout)) {
  HYPERVEC_THROW_IF_NOT_MSG(cache_ != nullptr,
                            "PagedVectorStorage: cache must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      cache_->PageSize() == layout_.PageSize(),
      "PagedVectorStorage: cache page size does not match the layout");
  HYPERVEC_THROW_IF_NOT_MSG(
      cache_->SourceSize() == layout_.StorageSize(),
      "PagedVectorStorage: data size does not match the layout");
}

DiskAnnVectorView PagedVectorStorage::Vector(GraphId node) const {
  const DiskAnnNodeLocation location = layout_.Locate(node);
  const PageCache::PageHandle page = cache_->Get(location.page_id);
  HYPERVEC_THROW_IF_NOT_MSG(
      location.offset <= page->data.size() &&
          layout_.VectorBytes() <= page->data.size() - location.offset,
      "PagedVectorStorage: vector record exceeds its page");
  const uint8_t* bytes = page->data.data() + location.offset;
  HYPERVEC_THROW_IF_NOT_MSG(
      reinterpret_cast<uintptr_t>(bytes) % alignof(float) == 0,
      "PagedVectorStorage: vector storage is not aligned");
  const float* vector = reinterpret_cast<const float*>(bytes);
  std::shared_ptr<const void> owner = page;
  return DiskAnnVectorView(std::span<const float>(vector, layout_.Dimension()),
                           std::move(owner));
}

void PagedVectorStorage::Prefetch(GraphId node) const noexcept {
  if (node < 0 || static_cast<size_t>(node) >= layout_.NodeCount()) {
    return;
  }
  const size_t index = static_cast<size_t>(node);
  cache_->Prefetch(static_cast<uint64_t>(index / layout_.RecordsPerPage()));
}

void WriteDiskAnnNodes(const DiskAnnNodeLayout& layout, const float* vectors,
                       const GraphStorage& graph, IOWriter* writer) {
  HYPERVEC_THROW_IF_NOT_MSG(writer != nullptr,
                            "WriteDiskAnnNodes: writer must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      layout.NodeCount() == 0 || vectors != nullptr,
      "WriteDiskAnnNodes: vectors must not be null for a non-empty layout");
  ValidateGraph(layout, graph);

  std::vector<uint8_t> page(layout.PageSize(), 0);
  size_t node = 0;
  for (uint64_t page_id = 0; page_id < layout.PageCount(); ++page_id) {
    std::fill(page.begin(), page.end(), uint8_t{0});
    for (size_t slot = 0;
         slot < layout.RecordsPerPage() && node < layout.NodeCount();
         ++slot, ++node) {
      uint8_t* record = page.data() + slot * layout.RecordSize();
      const float* source = vectors + node * layout.Dimension();
      std::memcpy(record, source, layout.VectorBytes());

      const GraphNeighborList neighbors =
          graph.Neighbors(static_cast<GraphId>(node));
      const uint32_t degree = static_cast<uint32_t>(neighbors.size());
      uint8_t* graph_record = record + layout.DegreeOffset();
      std::memcpy(graph_record, &degree, sizeof(degree));
      if (!neighbors.empty()) {
        std::memcpy(graph_record + sizeof(degree), neighbors.data(),
                    neighbors.size() * sizeof(GraphId));
      }
    }
    WriteExact(writer, page.data(), page.size());
  }
}

}  // namespace hypervec
