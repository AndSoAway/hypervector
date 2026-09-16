/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>

namespace hypervec {

struct HNSW;

/** Zero-copy read-only view of one HNSW graph layer.
 *
 * G3 uses layer zero in production search. The layer parameter remains
 * available for legacy helper compatibility and focused tests.
 */
class HnswGraphLayerView final : public GraphStorage {
 public:
  explicit HnswGraphLayerView(const HNSW& hnsw, int layer = 0);

  size_t NodeCount() const noexcept override;
  size_t MaxDegree() const noexcept override;
  GraphNeighborView Neighbors(GraphId node) const override;
  void Prefetch(GraphId node) const noexcept override;

 private:
  const HNSW& hnsw_;
  int layer_;
};

}  // namespace hypervec
