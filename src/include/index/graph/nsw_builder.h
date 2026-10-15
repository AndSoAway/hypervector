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
#include <mutex>
#include <span>
#include <vector>

namespace hypervec {

struct NSWBuildOptions {
  size_t ef_construction = 64;
  bool check_relative_distance = true;
  bool fill_to_max_degree = true;
};

struct NSWBuildStats {
  size_t inserted_nodes = 0;
  size_t reciprocal_updates = 0;
  size_t pruned_neighbor_lists = 0;
  size_t reciprocal_distance_computations = 0;
  GraphSearchStats search;
  GraphPruneStats pruning;

  void Reset() noexcept;
  void Combine(const NSWBuildStats& other) noexcept;
};

struct NSWInsertionResult {
  GraphId node_id = kInvalidGraphId;
  GraphId entry_point = kInvalidGraphId;
};

/** Incrementally constructs a single-layer navigable small-world graph.
 *
 * Stored vectors and graph nodes must use the same contiguous IDs. Before
 * adding a non-first node N, DistanceComputer must already be able to access
 * vector N and must have that vector configured as its current query.
 *
 * The builder searches existing nodes, selects outgoing neighbors with the
 * HNSW diversity heuristic, and updates reciprocal edges. Existing neighbor
 * lists are pruned only when they are already full.
 */
class NSWIncrementalBuilder {
 public:
  explicit NSWIncrementalBuilder(NSWBuildOptions options = {});

  /** Append one node and return its ID and the entry point for the next add.
   *
   * An empty graph requires kInvalidGraphId. A non-empty graph requires a
   * valid existing entry point. Validation and distance evaluation finish
   * before graph mutation begins.
   */
  NSWInsertionResult AddNode(DistanceComputer& distance,
                             MutableGraphStorage& graph, GraphId entry_point,
                             NSWBuildStats* stats = nullptr) const;

  /** Read-only proposal against a stable graph snapshot. Each worker owns its
   * distance computer and visited table; graph writes must wait for all
   * proposals in the batch to finish. */
  std::vector<GraphId> Propose(DistanceComputer& distance,
                               const GraphStorage& graph, GraphId entry_point,
                               VisitedTable* visited,
                               NSWBuildStats* stats) const;

  /** Install a proposal and its reciprocal links, in insertion order. */
  void Commit(DistanceComputer& distance, MutableGraphStorage& graph,
              GraphId node, const std::vector<GraphId>& neighbors,
              NSWBuildStats* stats) const;

  /** Concurrent commit for proposals which target only earlier batches.
   * Workers write disjoint new-node lists; each older reciprocal list is
   * protected independently. Caller pre-sizes graph and lock array. */
  void CommitConcurrent(DistanceComputer& distance, MutableGraphStorage& graph,
                        GraphId node, const std::vector<GraphId>& neighbors,
                        std::span<std::mutex> node_locks,
                        NSWBuildStats* stats) const;

  const NSWBuildOptions& Options() const noexcept { return options_; }

 private:
  NSWBuildOptions options_;
  HnswHeuristicPruner pruner_;
};

}  // namespace hypervec
