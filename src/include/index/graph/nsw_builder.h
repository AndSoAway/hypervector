/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>
#include <index/graph/visited_table.h>

#include <cstddef>

namespace hypervec {

struct DistanceComputer;

struct NSWBuildOptions {
  size_t graph_degree = 32;
  int ef_construction = 40;
};

/** Component-level counters suitable for construction diagnostics and Eval. */
struct NSWBuildStats {
  size_t nodes_inserted = 0;
  size_t graph_searches = 0;
  size_t distance_computations = 0;
  size_t reciprocal_updates = 0;

  void Reset() noexcept { *this = {}; }
  void Combine(const NSWBuildStats& other) noexcept;
};

/** Incremental single-layer NSW graph builder.
 *
 * The caller must resize the graph before insertion, set the distance
 * computer's query to the new vector, and insert nodes in increasing ID
 * order. Existing nodes are discovered with best-first graph search. The new
 * node keeps its nearest graph_degree candidates; reciprocal lists are
 * bounded by graph.MaxDegree().
 */
class NSWIncrementalBuilder {
 public:
  NSWIncrementalBuilder(MutableBoundedGraph& graph, NSWBuildOptions options);

  void AddNode(GraphId node, GraphId entry_point, DistanceComputer& distance,
               NSWBuildStats* stats = nullptr);

 private:
  MutableBoundedGraph& graph_;
  NSWBuildOptions options_;
  VisitedTable visited_;
};

}  // namespace hypervec
