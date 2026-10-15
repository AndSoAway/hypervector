/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>
#include <index/hnsw/hnsw.h>

#include <cstddef>

namespace hypervec {

enum class HNSWGraphValidation {
  /** Validate every node when the adapter is constructed. */
  kFull,
  /** Validate constant-size metadata eagerly and each node when accessed. */
  kOnAccess,
};

/** Zero-copy read-only view of one HNSW layer as common graph storage.
 *
 * Node identifiers stay dense across the complete HNSW index. Nodes that do
 * not exist at the selected upper layer expose an empty neighbor list. The
 * referenced HNSW object must outlive this adapter.
 */
class HNSWGraphStorage final : public GraphStorage {
 public:
  explicit HNSWGraphStorage(
      const HNSW& hnsw, int layer = 0,
      HNSWGraphValidation validation = HNSWGraphValidation::kFull);

  size_t NodeCount() const noexcept override;
  size_t MaxDegree() const noexcept override;
  GraphNeighborList Neighbors(GraphId node) const override;
  void Prefetch(GraphId node) const noexcept override;

  int Layer() const noexcept { return layer_; }

 private:
  const HNSW& hnsw_;
  int layer_;
};

}  // namespace hypervec
