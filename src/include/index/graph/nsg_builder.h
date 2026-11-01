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

namespace hypervec {

struct NSGBuildOptions {
  size_t max_degree = 32;
  size_t search_width = 40;
  size_t candidate_pool_size = 200;
  bool check_relative_distance = true;
};

struct NSGBuildStats {
  size_t pruned_nodes = 0;
  size_t candidate_distance_computations = 0;
  size_t reciprocal_edges_added = 0;
  size_t reciprocal_edges_rejected = 0;
  size_t connectivity_distance_computations = 0;
  size_t connectivity_edges_added = 0;
  GraphSearchStats search;
  GraphPruneStats pruning;

  void Reset() noexcept;
  void Combine(const NSGBuildStats& other) noexcept;
};

/** Convert a candidate k-nearest-neighbor graph into a sparse NSG graph.
 *
 * The caller selects the navigation point. For every node, the builder joins
 * graph-search results with its original candidate neighbors, applies the
 * relative-neighborhood occlusion rule, inserts reciprocal proposals, and
 * repairs directed reachability from the navigation point.
 *
 * Connectivity repair may add one edge beyond max_degree. Consequently the
 * returned graph capacity is min(node_count - 1, max_degree + 1).
 * DistanceComputer must provide symmetric_dis(), with smaller values meaning
 * closer neighbors.
 */
class NSGBuilder {
 public:
  explicit NSGBuilder(NSGBuildOptions options = {});

  MutableBoundedGraph Build(const GraphStorage& candidate_graph,
                            DistanceComputer& distance,
                            GraphId navigation_point,
                            NSGBuildStats* stats = nullptr) const;

  const NSGBuildOptions& Options() const noexcept { return options_; }

 private:
  NSGBuildOptions options_;
  HnswHeuristicPruner pruner_;
};

}  // namespace hypervec
