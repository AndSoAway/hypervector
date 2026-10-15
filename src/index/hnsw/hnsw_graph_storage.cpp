/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/hnsw/hnsw_graph_storage.h>
#include <utils/log/assert.h>
#include <utils/structures/prefetch.h>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace hypervec {
namespace {

void ValidateHeader(const HNSW& hnsw, int layer) {
  static_assert(std::is_same_v<HNSW::storage_idx_t, GraphId>);
  HYPERVEC_THROW_IF_NOT_MSG(layer >= 0,
                            "HNSWGraphStorage: layer must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<size_t>(layer) + 1 < hnsw.cum_nneighbor_per_level.size(),
      "HNSWGraphStorage: layer is outside the capacity table");
  HYPERVEC_THROW_IF_NOT_MSG(hnsw.offsets.size() == hnsw.levels.size() + 1 &&
                                !hnsw.offsets.empty() &&
                                hnsw.offsets.front() == 0 &&
                                hnsw.offsets.back() == hnsw.neighbors.size(),
                            "HNSWGraphStorage: graph arrays are inconsistent");
}

void ValidateNodeLayout(const HNSW& hnsw, size_t node) {
  HYPERVEC_THROW_IF_NOT_MSG(hnsw.levels[node] > 0 &&
                                static_cast<size_t>(hnsw.levels[node]) <
                                    hnsw.cum_nneighbor_per_level.size() &&
                                hnsw.offsets[node] <= hnsw.offsets[node + 1],
                            "HNSWGraphStorage: node layout is invalid");
}

}  // namespace

HNSWGraphStorage::HNSWGraphStorage(const HNSW& hnsw, int layer,
                                   HNSWGraphValidation validation)
    : hnsw_(hnsw), layer_(layer) {
  HYPERVEC_THROW_IF_NOT_MSG(validation == HNSWGraphValidation::kFull ||
                                validation == HNSWGraphValidation::kOnAccess,
                            "HNSWGraphStorage: unknown validation policy");
  ValidateHeader(hnsw_, layer_);
  if (validation == HNSWGraphValidation::kFull) {
    for (size_t node = 0; node < hnsw_.levels.size(); ++node) {
      ValidateNodeLayout(hnsw_, node);
    }
  }
}

size_t HNSWGraphStorage::NodeCount() const noexcept {
  return hnsw_.levels.size();
}

size_t HNSWGraphStorage::MaxDegree() const noexcept {
  const size_t layer = static_cast<size_t>(layer_);
  if (layer_ < 0 || layer + 1 >= hnsw_.cum_nneighbor_per_level.size()) {
    return 0;
  }
  const int begin = hnsw_.cum_nneighbor_per_level[layer];
  const int end = hnsw_.cum_nneighbor_per_level[layer + 1];
  return begin >= 0 && end >= begin ? static_cast<size_t>(end - begin) : 0;
}

GraphNeighborList HNSWGraphStorage::Neighbors(GraphId node) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      node >= 0 && static_cast<size_t>(node) < NodeCount(),
      "HNSWGraphStorage::Neighbors: node is outside the graph");
  const size_t index = static_cast<size_t>(node);
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw_.offsets.size() == NodeCount() + 1 && !hnsw_.offsets.empty() &&
          hnsw_.offsets.front() == 0 &&
          hnsw_.offsets.back() == hnsw_.neighbors.size(),
      "HNSWGraphStorage::Neighbors: graph layout changed incompatibly");
  ValidateNodeLayout(hnsw_, index);
  if (hnsw_.levels[index] <= layer_) {
    return {};
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<size_t>(layer_) + 1 < hnsw_.cum_nneighbor_per_level.size(),
      "HNSWGraphStorage::Neighbors: graph layout changed incompatibly");
  const size_t layer_begin = static_cast<size_t>(
      hnsw_.cum_nneighbor_per_level[static_cast<size_t>(layer_)]);
  const size_t layer_end = static_cast<size_t>(
      hnsw_.cum_nneighbor_per_level[static_cast<size_t>(layer_) + 1]);
  HYPERVEC_THROW_IF_NOT_MSG(
      layer_end >= layer_begin &&
          hnsw_.offsets[index] <=
              (std::numeric_limits<size_t>::max)() - layer_end,
      "HNSWGraphStorage::Neighbors: layer offset overflows");
  const size_t begin = hnsw_.offsets[index] + layer_begin;
  const size_t end = hnsw_.offsets[index] + layer_end;
  HYPERVEC_THROW_IF_NOT_MSG(
      end <= hnsw_.offsets[index + 1] && end <= hnsw_.neighbors.size(),
      "HNSWGraphStorage::Neighbors: neighbor span is invalid");

  size_t count = 0;
  bool reached_padding = false;
  for (size_t position = begin; position < end; ++position) {
    const GraphId neighbor = hnsw_.neighbors[position];
    if (neighbor == kInvalidGraphId) {
      reached_padding = true;
      continue;
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        !reached_padding && neighbor >= 0 &&
            static_cast<size_t>(neighbor) < NodeCount() && neighbor != node &&
            hnsw_.levels[static_cast<size_t>(neighbor)] > layer_,
        "HNSWGraphStorage::Neighbors: neighbor entry is invalid");
    ++count;
  }
  const GraphId* first = count == 0 ? nullptr : hnsw_.neighbors.data() + begin;
  return GraphNeighborList(GraphNeighborView(first, count));
}

void HNSWGraphStorage::Prefetch(GraphId node) const noexcept {
  if (node < 0 || static_cast<size_t>(node) >= NodeCount()) {
    return;
  }
  const size_t index = static_cast<size_t>(node);
  const size_t layer = static_cast<size_t>(layer_);
  if (layer_ < 0 || hnsw_.levels[index] <= layer_ ||
      layer + 1 >= hnsw_.cum_nneighbor_per_level.size() ||
      hnsw_.offsets.size() != NodeCount() + 1) {
    return;
  }
  const int relative = hnsw_.cum_nneighbor_per_level[layer];
  if (relative < 0 ||
      hnsw_.offsets[index] > (std::numeric_limits<size_t>::max)() -
                                 static_cast<size_t>(relative)) {
    return;
  }
  const size_t begin = hnsw_.offsets[index] + static_cast<size_t>(relative);
  if (begin < hnsw_.offsets[index + 1] && begin < hnsw_.neighbors.size()) {
    prefetch_L2(hnsw_.neighbors.data() + begin);
  }
}

}  // namespace hypervec
