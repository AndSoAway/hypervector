/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/hnsw/hnsw.h>
#include <index/hnsw/hnsw_graph_adapter.h>
#include <utils/log/assert.h>
#include <utils/structures/prefetch.h>

namespace hypervec {

HnswGraphLayerView::HnswGraphLayerView(const HNSW& hnsw, int layer)
    : hnsw_(hnsw), layer_(layer) {
  HYPERVEC_THROW_IF_NOT_MSG(layer >= 0,
                            "HnswGraphLayerView: layer must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<size_t>(layer + 1) < hnsw_.cum_nneighbor_per_level.size(),
      "HnswGraphLayerView: layer is outside the HNSW layout");
}

size_t HnswGraphLayerView::NodeCount() const noexcept {
  return hnsw_.levels.size();
}

size_t HnswGraphLayerView::MaxDegree() const noexcept {
  const size_t level = static_cast<size_t>(layer_);
  return static_cast<size_t>(hnsw_.cum_nneighbor_per_level[level + 1] -
                             hnsw_.cum_nneighbor_per_level[level]);
}

GraphNeighborView HnswGraphLayerView::Neighbors(GraphId node) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      node >= 0 && static_cast<size_t>(node) < NodeCount(),
      "HnswGraphLayerView::Neighbors: node is outside the graph");
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw_.offsets.size() == NodeCount() + 1,
      "HnswGraphLayerView::Neighbors: invalid HNSW offsets");
  if (hnsw_.levels[static_cast<size_t>(node)] <= layer_) {
    return {};
  }

  size_t begin = 0;
  size_t end = 0;
  hnsw_.NeighborRange(node, layer_, &begin, &end);
  HYPERVEC_THROW_IF_NOT_MSG(
      begin <= end && end <= hnsw_.neighbors.size(),
      "HnswGraphLayerView::Neighbors: neighbor range is invalid");
  size_t count = 0;
  while (begin + count < end && hnsw_.neighbors[begin + count] >= 0) {
    ++count;
  }
  return GraphNeighborView(hnsw_.neighbors.data() + begin, count);
}

void HnswGraphLayerView::Prefetch(GraphId node) const noexcept {
  if (node < 0 || static_cast<size_t>(node) >= NodeCount() ||
      hnsw_.levels[static_cast<size_t>(node)] <= layer_ ||
      hnsw_.offsets.size() != NodeCount() + 1) {
    return;
  }
  const size_t begin =
      hnsw_.offsets[static_cast<size_t>(node)] +
      static_cast<size_t>(hnsw_.cum_nneighbor_per_level[layer_]);
  if (begin < hnsw_.neighbors.size()) {
    prefetch_L2(hnsw_.neighbors.data() + begin);
  }
}

}  // namespace hypervec
