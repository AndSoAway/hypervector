/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <index/graph/neighbor_pruner.h>
#include <index/graph/nsw_builder.h>
#include <index/graph/visited_table.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hypervec {
namespace {

bool CandidateLess(const NeighborCandidate& lhs, const NeighborCandidate& rhs) {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

std::vector<NeighborCandidate> CollectCandidates(
    GraphId node, GraphId entry_point, const GraphStorage& graph,
    int ef_construction, DistanceComputer& distance, VisitedTable& visited,
    NSWBuildStats* stats) {
  const int result_count = std::min(ef_construction, static_cast<int>(node));
  std::vector<float> distances(static_cast<size_t>(result_count));
  std::vector<idx_t> labels(static_cast<size_t>(result_count));
  using Handler = HeapBlockResultHandler<CMax<float, idx_t>>;
  Handler block(1, distances.data(), labels.data(), result_count);
  Handler::SingleResultHandler results(block);
  results.begin(0);

  GraphCandidateQueue candidates(ef_construction);
  candidates.push(entry_point, distance(entry_point));
  GraphSearchOptions search_options;
  search_options.ef_search = ef_construction;
  GraphSearchStats search_stats;
  GraphSearcher(graph).Search(distance, results, candidates, visited,
                              search_options, &search_stats);
  results.end();
  visited.advance();

  std::vector<NeighborCandidate> collected;
  collected.reserve(static_cast<size_t>(result_count));
  for (int i = 0; i < result_count; ++i) {
    if (labels[static_cast<size_t>(i)] >= 0) {
      collected.push_back({static_cast<GraphId>(labels[static_cast<size_t>(i)]),
                           distances[static_cast<size_t>(i)]});
    }
  }
  std::sort(collected.begin(), collected.end(), CandidateLess);
  collected.erase(std::unique(collected.begin(), collected.end(),
                              [](const NeighborCandidate& lhs,
                                 const NeighborCandidate& rhs) {
                                return lhs.id == rhs.id;
                              }),
                  collected.end());

  if (stats != nullptr) {
    ++stats->graph_searches;
    // The initial entry-point distance is supplied by this component; graph
    // search reports only distances computed while traversing edges.
    stats->distance_computations +=
        search_stats.distance_computations + size_t{1};
  }
  return collected;
}

void AddReciprocalEdge(GraphId existing, GraphId added,
                       MutableBoundedGraph& graph, DistanceComputer& distance,
                       NSWBuildStats* stats) {
  std::vector<NeighborCandidate> candidates;
  const GraphNeighborView old_neighbors = graph.Neighbors(existing);
  candidates.reserve(old_neighbors.size() + 1);
  for (GraphId neighbor : old_neighbors) {
    candidates.push_back(
        {neighbor, distance.symmetric_dis(existing, neighbor)});
  }
  const GraphId protected_successor = existing + 1;
  const bool has_added = std::any_of(
      candidates.begin(), candidates.end(),
      [added](const auto& candidate) { return candidate.id == added; });
  if (!has_added) {
    candidates.push_back({added, distance.symmetric_dis(existing, added)});
  }
  const bool has_successor =
      std::any_of(candidates.begin(), candidates.end(),
                  [protected_successor](const auto& candidate) {
                    return candidate.id == protected_successor;
                  });
  if (!has_successor && protected_successor <= added) {
    candidates.push_back(
        {protected_successor,
         distance.symmetric_dis(existing, protected_successor)});
  }
  std::sort(candidates.begin(), candidates.end(), CandidateLess);

  NearestNeighborPruner pruner;
  const std::vector<NeighborCandidate> selected =
      pruner.Prune(candidates, graph.MaxDegree(), distance);
  std::vector<GraphId> neighbors;
  neighbors.reserve(selected.size());
  bool contains_successor = false;
  for (const NeighborCandidate& candidate : selected) {
    neighbors.push_back(candidate.id);
    contains_successor =
        contains_successor || candidate.id == protected_successor;
  }

  // Preserve the insertion-order backbone. Ordinary NSW edges remain nearest
  // neighbor links, while 0 -> 1 -> ... guarantees that bounded reciprocal
  // pruning cannot disconnect a node from the fixed entry point later.
  if (!contains_successor) {
    neighbors.back() = protected_successor;
  }
  graph.SetNeighbors(existing, neighbors);
  if (stats != nullptr) {
    stats->distance_computations += candidates.size();
    ++stats->reciprocal_updates;
  }
}

}  // namespace

void NSWBuildStats::Combine(const NSWBuildStats& other) noexcept {
  nodes_inserted += other.nodes_inserted;
  graph_searches += other.graph_searches;
  distance_computations += other.distance_computations;
  reciprocal_updates += other.reciprocal_updates;
}

NSWIncrementalBuilder::NSWIncrementalBuilder(MutableBoundedGraph& graph,
                                             NSWBuildOptions options)
    : graph_(graph), options_(options), visited_(graph.NodeCount()) {
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.graph_degree > 0,
      "NSWIncrementalBuilder: graph_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.graph_degree <= graph_.MaxDegree(),
      "NSWIncrementalBuilder: graph_degree exceeds graph max_degree");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.ef_construction > 0,
      "NSWIncrementalBuilder: ef_construction must be positive");
}

void NSWIncrementalBuilder::AddNode(GraphId node, GraphId entry_point,
                                    DistanceComputer& distance,
                                    NSWBuildStats* stats) {
  HYPERVEC_THROW_IF_NOT_MSG(
      node > 0 && static_cast<size_t>(node) < graph_.NodeCount(),
      "NSWIncrementalBuilder::AddNode: node must be an allocated non-root ID");
  HYPERVEC_THROW_IF_NOT_MSG(
      entry_point >= 0 && entry_point < node,
      "NSWIncrementalBuilder::AddNode: entry point must be an existing node");

  NSWBuildStats local_stats;
  std::vector<NeighborCandidate> candidates =
      CollectCandidates(node, entry_point, graph_, options_.ef_construction,
                        distance, visited_, &local_stats);
  const GraphId predecessor = node - 1;
  const bool has_predecessor =
      std::any_of(candidates.begin(), candidates.end(),
                  [predecessor](const auto& candidate) {
                    return candidate.id == predecessor;
                  });
  if (!has_predecessor) {
    candidates.push_back({predecessor, distance(predecessor)});
    ++local_stats.distance_computations;
    std::sort(candidates.begin(), candidates.end(), CandidateLess);
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      !candidates.empty(),
      "NSWIncrementalBuilder::AddNode: graph search found no candidates");

  NearestNeighborPruner pruner;
  std::vector<NeighborCandidate> selected =
      pruner.Prune(candidates, options_.graph_degree, distance);
  const bool selected_predecessor = std::any_of(
      selected.begin(), selected.end(), [predecessor](const auto& candidate) {
        return candidate.id == predecessor;
      });
  if (!selected_predecessor) {
    selected.back() = {predecessor, distance(predecessor)};
    ++local_stats.distance_computations;
  }
  std::vector<GraphId> neighbors;
  neighbors.reserve(selected.size());
  for (const NeighborCandidate& candidate : selected) {
    neighbors.push_back(candidate.id);
  }
  graph_.SetNeighbors(node, neighbors);

  for (size_t i = 0; i < selected.size(); ++i) {
    AddReciprocalEdge(selected[i].id, node, graph_, distance, &local_stats);
  }
  ++local_stats.nodes_inserted;
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
}

}  // namespace hypervec
