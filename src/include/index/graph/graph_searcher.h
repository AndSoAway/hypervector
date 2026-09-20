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

enum class GraphSearchFrontierPolicy {
  /** Selected results also bound graph navigation. */
  kResultBound,
  /** A separate unfiltered nearest-neighbor set bounds navigation. */
  kNavigationBound,
};

struct GraphSearchOptions {
  size_t ef_search = 16;
  bool check_relative_distance = true;
  const IDSelector* selector = nullptr;
  GraphSearchFrontierPolicy frontier_policy =
      GraphSearchFrontierPolicy::kResultBound;
  /** Maximum expanded nodes per search, or zero for no explicit limit. */
  size_t max_expansions = 0;
  /** Maximum pending candidates, or zero for an unbounded candidate queue. */
  size_t max_candidates = 0;
};

struct GraphSearchStats {
  size_t queries = 0;
  size_t exhausted_queries = 0;
  size_t distance_computations = 0;
  size_t visited_nodes = 0;
  size_t expanded_nodes = 0;
  size_t traversed_edges = 0;
  size_t peak_candidates = 0;

  void Reset() noexcept;
  void Combine(const GraphSearchStats& other) noexcept;
};

struct GraphSearchResult {
  GraphId id;
  float distance;
};

/** An entry point whose distance to the current query is already known. */
struct GraphSearchSeed {
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

  /** Search from entry points with caller-computed distances.
   *
   * Seed distances are not included in distance_computations because this
   * call does not compute them. If an ID occurs more than once, the first
   * seed is used.
   */
  std::vector<GraphSearchResult> Search(
      DistanceComputer& distance, std::span<const GraphSearchSeed> seeds,
      const GraphSearchOptions& options, VisitedTable* visited,
      GraphSearchStats* stats = nullptr) const;

 private:
  std::vector<GraphSearchResult> SearchPrepared(
      DistanceComputer& distance, std::span<const GraphSearchSeed> seeds,
      size_t seed_distance_computations, const GraphSearchOptions& options,
      VisitedTable* visited, GraphSearchStats* stats) const;

  const GraphStorage& graph_;
};

}  // namespace hypervec
