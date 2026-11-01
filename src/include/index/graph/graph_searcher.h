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
#include <utils/distances/distance_computer.h>

#include <cstddef>
#include <span>
#include <vector>

namespace hypervec {

struct IDSelector;

struct GraphSearchOptions {
  size_t ef_search = 16;
  bool check_relative_distance = true;
  const IDSelector* selector = nullptr;
};

struct GraphSearchStats {
  size_t queries = 0;
  size_t exhausted_queries = 0;
  size_t distance_computations = 0;
  size_t visited_nodes = 0;
  size_t expanded_nodes = 0;
  size_t traversed_edges = 0;

  void Reset() noexcept;
  void Combine(const GraphSearchStats& other) noexcept;
};

struct GraphSearchResult {
  GraphId id;
  float distance;
};

/** Best-first search over one graph layer.
 *
 * DistanceComputer must be configured with the current query before Search.
 * Selectors affect returned results only: rejected nodes remain available as
 * navigation intermediates. Results are ordered by distance, then node ID.
 */
class GraphSearcher {
 public:
  explicit GraphSearcher(const GraphStorage& graph) : graph_(graph) {}

  std::vector<GraphSearchResult> Search(
      DistanceComputer& distance, std::span<const GraphId> entry_points,
      const GraphSearchOptions& options, VisitedTable* visited,
      GraphSearchStats* stats = nullptr) const;

 private:
  const GraphStorage& graph_;
};

}  // namespace hypervec
