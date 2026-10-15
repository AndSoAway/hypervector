/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_searcher.h>
#include <index/graph/graph_storage.h>
#include <index/graph/neighbor_pruner.h>
#include <utils/distances/distance_computer.h>

#include <cstddef>
#include <cstdint>

namespace hypervec {

struct VamanaBuildOptions {
  size_t max_degree = 32;
  size_t search_width = 64;
  size_t candidate_pool_size = 200;
  float alpha = 1.2F;
  size_t build_passes = 2;
  uint64_t random_seed = 0x9E3779B97F4A7C15ULL;
};

struct VamanaBuildStats {
  size_t passes_completed = 0;
  size_t nodes_processed = 0;
  size_t candidate_distance_computations = 0;
  size_t reciprocal_edges_added = 0;
  size_t reciprocal_edges_repruned = 0;
  size_t reciprocal_edges_rejected = 0;
  size_t connectivity_distance_computations = 0;
  size_t connectivity_edges_added = 0;
  size_t connectivity_edges_replaced = 0;
  GraphSearchStats search;
  GraphPruneStats pruning;

  void Reset() noexcept;
  void Combine(const VamanaBuildStats& other) noexcept;
};

/** Build an in-memory Vamana graph by greedy search and robust pruning.
 *
 * The caller supplies the navigation point. Each pass visits nodes in a
 * deterministic shuffled order, searches the graph built so far, robustly
 * prunes outgoing candidates, and proposes reciprocal edges. When multiple
 * passes are requested, the first uses alpha=1 and later passes use the
 * configured alpha. A single pass uses the configured alpha directly. A
 * final bounded repair preserves a directed path from the navigation point to
 * every node, including datasets with duplicate vectors.
 *
 * DistanceComputer must provide symmetric_dis(), with smaller values meaning
 * closer neighbors.
 */
class VamanaBuilder {
 public:
  explicit VamanaBuilder(VamanaBuildOptions options = {});

  MutableBoundedGraph Build(DistanceComputer& distance, size_t node_count,
                            GraphId navigation_point,
                            VamanaBuildStats* stats = nullptr) const;

  const VamanaBuildOptions& Options() const noexcept { return options_; }

 private:
  VamanaBuildOptions options_;
};

}  // namespace hypervec
