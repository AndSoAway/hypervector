/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hypervec {

struct DistanceComputer;
struct IDSelector;
template <typename T, typename TI>
struct ResultHandlerUnordered;
struct VisitedTable;
using ResultHandler = ResultHandlerUnordered<float, int64_t>;

/** Bounded best-first candidate queue shared by graph indexes. */
class GraphCandidateQueue {
 public:
  explicit GraphCandidateQueue(int capacity);

  void push(GraphId id, float distance);
  float max() const;
  int size() const;
  void clear();
  int PopMin(float* minimum_distance = nullptr);
  int CountBelow(float threshold) const;

  // Kept public while HNSW migrates its existing diagnostic/test helpers.
  int n;
  int k;
  int nvalid;
  std::vector<GraphId> ids;
  std::vector<float> dis;
};

/** Search controls independent of a concrete graph index. */
struct GraphSearchOptions {
  int ef_search = 16;
  bool check_relative_distance = true;
  const IDSelector* selector = nullptr;
};

/** Component-level counters suitable for Eval statistics. */
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

/** Best-first traversal over any GraphStorage implementation.
 *
 * Filtering controls result admission only: filtered nodes remain valid
 * navigation intermediates. Initial candidate distances are supplied by the
 * caller and therefore are not counted in distance_computations. The graph
 * must satisfy GraphStorage's structural invariants; use ValidateGraph at
 * construction or load boundaries rather than validating every edge here.
 */
class GraphSearcher {
 public:
  explicit GraphSearcher(const GraphStorage& graph) : graph_(graph) {}

  int Search(DistanceComputer& distance, ResultHandler& results,
             GraphCandidateQueue& candidates, VisitedTable& visited,
             const GraphSearchOptions& options, GraphSearchStats* stats,
             int initial_result_count = 0) const;

 private:
  const GraphStorage& graph_;
};

}  // namespace hypervec
