/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_storage.h>
#include <utils/log/assert.h>
#include <utils/structures/prefetch.h>

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

void ValidateNodeCount(size_t node_count, const char* operation) {
  constexpr size_t kMaxNodeCount =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_FMT(node_count <= kMaxNodeCount,
                            "%s: node count exceeds GraphId capacity",
                            operation);
}

size_t ValidateNode(GraphId node, size_t node_count, const char* operation) {
  HYPERVEC_THROW_IF_NOT_FMT(node >= 0 && static_cast<size_t>(node) < node_count,
                            "%s: graph node is outside [0, node_count)",
                            operation);
  return static_cast<size_t>(node);
}

std::vector<GraphId> ValidateNeighborList(GraphId node,
                                          GraphNeighborView neighbors,
                                          size_t node_count, size_t max_degree,
                                          const char* operation) {
  HYPERVEC_THROW_IF_NOT_FMT(neighbors.size() <= max_degree,
                            "%s: neighbor count exceeds max_degree", operation);
  std::vector<GraphId> validated;
  validated.reserve(neighbors.size());
  std::unordered_set<GraphId> seen;
  for (GraphId neighbor : neighbors) {
    ValidateNode(neighbor, node_count, operation);
    HYPERVEC_THROW_IF_NOT_FMT(neighbor != node,
                              "%s: self loops are not allowed", operation);
    HYPERVEC_THROW_IF_NOT_FMT(seen.insert(neighbor).second,
                              "%s: duplicate neighbors are not allowed",
                              operation);
    validated.push_back(neighbor);
  }
  return validated;
}

size_t CheckedStorageSize(size_t node_count, size_t max_degree,
                          const char* operation) {
  if (node_count == 0) {
    return 0;
  }
  HYPERVEC_THROW_IF_NOT_FMT(
      max_degree <= (std::numeric_limits<size_t>::max)() / node_count,
      "%s: graph allocation size overflow", operation);
  return node_count * max_degree;
}

}  // namespace

void GraphStorage::Prefetch(GraphId /* node */) const noexcept {}

MutableBoundedGraph::MutableBoundedGraph(size_t max_degree)
    : max_degree_(max_degree) {
  HYPERVEC_THROW_IF_NOT_MSG(max_degree > 0,
                            "MutableBoundedGraph: max_degree must be positive");
}

MutableBoundedGraph::MutableBoundedGraph(size_t node_count, size_t max_degree)
    : MutableBoundedGraph(max_degree) {
  Resize(node_count);
}

size_t MutableBoundedGraph::NodeCount() const noexcept {
  return adjacency_.size();
}

size_t MutableBoundedGraph::MaxDegree() const noexcept { return max_degree_; }

GraphNeighborView MutableBoundedGraph::Neighbors(GraphId node) const {
  const size_t index =
      ValidateNode(node, NodeCount(), "MutableBoundedGraph::Neighbors");
  return adjacency_[index];
}

void MutableBoundedGraph::Prefetch(GraphId node) const noexcept {
  if (node < 0 || static_cast<size_t>(node) >= adjacency_.size()) {
    return;
  }
  const auto& neighbors = adjacency_[static_cast<size_t>(node)];
  if (!neighbors.empty()) {
    prefetch_L2(neighbors.data());
  }
}

void MutableBoundedGraph::Resize(size_t node_count) {
  ValidateNodeCount(node_count, "MutableBoundedGraph::Resize");
  if (node_count < adjacency_.size()) {
    for (size_t node = 0; node < node_count; ++node) {
      for (GraphId neighbor : adjacency_[node]) {
        HYPERVEC_THROW_IF_NOT_MSG(
            static_cast<size_t>(neighbor) < node_count,
            "MutableBoundedGraph::Resize: retained node references a removed "
            "node");
      }
    }
  }
  adjacency_.resize(node_count);
}

void MutableBoundedGraph::SetNeighbors(GraphId node,
                                       GraphNeighborView neighbors) {
  const size_t index =
      ValidateNode(node, NodeCount(), "MutableBoundedGraph::SetNeighbors");
  adjacency_[index] =
      ValidateNeighborList(node, neighbors, NodeCount(), max_degree_,
                           "MutableBoundedGraph::SetNeighbors");
}

bool MutableBoundedGraph::AddNeighbor(GraphId node, GraphId neighbor) {
  const size_t index =
      ValidateNode(node, NodeCount(), "MutableBoundedGraph::AddNeighbor");
  ValidateNode(neighbor, NodeCount(), "MutableBoundedGraph::AddNeighbor");
  HYPERVEC_THROW_IF_NOT_MSG(node != neighbor,
                            "MutableBoundedGraph::AddNeighbor: self loop");
  auto& neighbors = adjacency_[index];
  if (std::find(neighbors.begin(), neighbors.end(), neighbor) !=
      neighbors.end()) {
    return false;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      neighbors.size() < max_degree_,
      "MutableBoundedGraph::AddNeighbor: max_degree exceeded");
  neighbors.push_back(neighbor);
  return true;
}

bool MutableBoundedGraph::RemoveNeighbor(GraphId node, GraphId neighbor) {
  const size_t index =
      ValidateNode(node, NodeCount(), "MutableBoundedGraph::RemoveNeighbor");
  auto& neighbors = adjacency_[index];
  const auto position = std::find(neighbors.begin(), neighbors.end(), neighbor);
  if (position == neighbors.end()) {
    return false;
  }
  neighbors.erase(position);
  return true;
}

FixedDegreeGraph::FixedDegreeGraph(size_t max_degree)
    : max_degree_(max_degree) {
  HYPERVEC_THROW_IF_NOT_MSG(max_degree > 0,
                            "FixedDegreeGraph: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      max_degree <= (std::numeric_limits<uint32_t>::max)(),
      "FixedDegreeGraph: max_degree exceeds degree storage capacity");
}

FixedDegreeGraph::FixedDegreeGraph(size_t node_count, size_t max_degree)
    : FixedDegreeGraph(max_degree) {
  Resize(node_count);
}

size_t FixedDegreeGraph::NodeCount() const noexcept { return degrees_.size(); }

size_t FixedDegreeGraph::MaxDegree() const noexcept { return max_degree_; }

GraphNeighborView FixedDegreeGraph::Neighbors(GraphId node) const {
  const size_t index =
      ValidateNode(node, NodeCount(), "FixedDegreeGraph::Neighbors");
  return GraphNeighborView(data_.data() + index * max_degree_, degrees_[index]);
}

void FixedDegreeGraph::Prefetch(GraphId node) const noexcept {
  if (node < 0 || static_cast<size_t>(node) >= NodeCount()) {
    return;
  }
  prefetch_L2(data_.data() + static_cast<size_t>(node) * max_degree_);
}

void FixedDegreeGraph::Resize(size_t node_count) {
  ValidateNodeCount(node_count, "FixedDegreeGraph::Resize");
  if (node_count < NodeCount()) {
    for (size_t node = 0; node < node_count; ++node) {
      for (GraphId neighbor : Neighbors(static_cast<GraphId>(node))) {
        HYPERVEC_THROW_IF_NOT_MSG(
            static_cast<size_t>(neighbor) < node_count,
            "FixedDegreeGraph::Resize: retained node references a removed "
            "node");
      }
    }
  }
  const size_t new_storage_size =
      CheckedStorageSize(node_count, max_degree_, "FixedDegreeGraph::Resize");
  std::vector<GraphId> replacement(new_storage_size, kInvalidGraphId);
  std::vector<uint32_t> replacement_degrees(node_count, 0);
  const size_t retained_nodes = std::min(node_count, NodeCount());
  for (size_t node = 0; node < retained_nodes; ++node) {
    const size_t degree = degrees_[node];
    std::copy_n(data_.data() + node * max_degree_, degree,
                replacement.data() + node * max_degree_);
    replacement_degrees[node] = degrees_[node];
  }
  data_ = std::move(replacement);
  degrees_ = std::move(replacement_degrees);
}

void FixedDegreeGraph::SetNeighbors(GraphId node, GraphNeighborView neighbors) {
  const size_t index =
      ValidateNode(node, NodeCount(), "FixedDegreeGraph::SetNeighbors");
  const std::vector<GraphId> validated =
      ValidateNeighborList(node, neighbors, NodeCount(), max_degree_,
                           "FixedDegreeGraph::SetNeighbors");
  GraphId* destination = data_.data() + index * max_degree_;
  std::fill_n(destination, max_degree_, kInvalidGraphId);
  std::copy(validated.begin(), validated.end(), destination);
  degrees_[index] = static_cast<uint32_t>(validated.size());
}

CsrGraph::CsrGraph(std::vector<size_t> offsets, std::vector<GraphId> edges)
    : offsets_(std::move(offsets)), edges_(std::move(edges)) {
  ValidateAndSetMaxDegree();
}

CsrGraph::CsrGraph(const GraphStorage& graph) {
  ValidateNodeCount(graph.NodeCount(), "CsrGraph");
  offsets_.reserve(graph.NodeCount() + 1);
  offsets_.push_back(0);
  for (size_t node = 0; node < graph.NodeCount(); ++node) {
    const auto neighbors = graph.Neighbors(static_cast<GraphId>(node));
    HYPERVEC_THROW_IF_NOT_MSG(
        edges_.size() <=
            (std::numeric_limits<size_t>::max)() - neighbors.size(),
        "CsrGraph: edge count overflow");
    edges_.insert(edges_.end(), neighbors.begin(), neighbors.end());
    offsets_.push_back(edges_.size());
  }
  ValidateAndSetMaxDegree();
}

size_t CsrGraph::NodeCount() const noexcept {
  return offsets_.empty() ? 0 : offsets_.size() - 1;
}

size_t CsrGraph::MaxDegree() const noexcept { return max_degree_; }

GraphNeighborView CsrGraph::Neighbors(GraphId node) const {
  const size_t index = ValidateNode(node, NodeCount(), "CsrGraph::Neighbors");
  const size_t count = offsets_[index + 1] - offsets_[index];
  const GraphId* data = count == 0 ? nullptr : edges_.data() + offsets_[index];
  return GraphNeighborView(data, count);
}

void CsrGraph::Prefetch(GraphId node) const noexcept {
  if (node < 0 || static_cast<size_t>(node) >= NodeCount()) {
    return;
  }
  const size_t index = static_cast<size_t>(node);
  if (offsets_[index] < offsets_[index + 1]) {
    prefetch_L2(edges_.data() + offsets_[index]);
  }
}

void CsrGraph::ValidateAndSetMaxDegree() {
  HYPERVEC_THROW_IF_NOT_MSG(!offsets_.empty(),
                            "CsrGraph: offsets must contain an initial zero");
  HYPERVEC_THROW_IF_NOT_MSG(offsets_.front() == 0,
                            "CsrGraph: first offset must be zero");
  ValidateNodeCount(NodeCount(), "CsrGraph");
  HYPERVEC_THROW_IF_NOT_MSG(offsets_.back() == edges_.size(),
                            "CsrGraph: final offset must equal edge count");

  max_degree_ = 0;
  for (size_t node = 0; node < NodeCount(); ++node) {
    HYPERVEC_THROW_IF_NOT_MSG(offsets_[node] <= offsets_[node + 1],
                              "CsrGraph: offsets must be nondecreasing");
    HYPERVEC_THROW_IF_NOT_MSG(offsets_[node + 1] <= edges_.size(),
                              "CsrGraph: offset exceeds edge storage");
    const size_t count = offsets_[node + 1] - offsets_[node];
    const GraphId* data = count == 0 ? nullptr : edges_.data() + offsets_[node];
    const GraphNeighborView neighbors(data, count);
    ValidateNeighborList(static_cast<GraphId>(node), neighbors, NodeCount(),
                         neighbors.size(), "CsrGraph");
    max_degree_ = std::max(max_degree_, neighbors.size());
  }
}

}  // namespace hypervec
