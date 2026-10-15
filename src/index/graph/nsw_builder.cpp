/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/nsw_builder.h>
#include <index/graph/visited_table.h>
#include <utils/log/assert.h>

#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

struct PendingNeighborUpdate {
  GraphId node;
  std::vector<GraphId> neighbors;
};

std::vector<GraphId> CandidateIds(
    const std::vector<NeighborCandidate>& candidates) {
  std::vector<GraphId> ids;
  ids.reserve(candidates.size());
  for (const NeighborCandidate& candidate : candidates) {
    ids.push_back(candidate.id);
  }
  return ids;
}

}  // namespace

void NSWBuildStats::Reset() noexcept { *this = {}; }

void NSWBuildStats::Combine(const NSWBuildStats& other) noexcept {
  inserted_nodes += other.inserted_nodes;
  reciprocal_updates += other.reciprocal_updates;
  pruned_neighbor_lists += other.pruned_neighbor_lists;
  reciprocal_distance_computations += other.reciprocal_distance_computations;
  search.Combine(other.search);
  pruning.Combine(other.pruning);
}

NSWIncrementalBuilder::NSWIncrementalBuilder(NSWBuildOptions options)
    : options_(options), pruner_(options.fill_to_max_degree) {
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.ef_construction > 0,
      "NSWIncrementalBuilder: ef_construction must be positive");
}

NSWInsertionResult NSWIncrementalBuilder::AddNode(DistanceComputer& distance,
                                                  MutableGraphStorage& graph,
                                                  GraphId entry_point,
                                                  NSWBuildStats* stats) const {
  const size_t old_node_count = graph.NodeCount();
  HYPERVEC_THROW_IF_NOT_MSG(
      old_node_count <=
          static_cast<size_t>((std::numeric_limits<GraphId>::max)()),
      "NSWIncrementalBuilder: graph has no remaining GraphId capacity");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph.MaxDegree() > 0,
      "NSWIncrementalBuilder: graph max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.ef_construction >= graph.MaxDegree(),
      "NSWIncrementalBuilder: ef_construction must cover max_degree");

  const GraphId new_node = static_cast<GraphId>(old_node_count);
  NSWBuildStats local_stats;
  if (old_node_count == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        entry_point == kInvalidGraphId,
        "NSWIncrementalBuilder: an empty graph requires an invalid entry "
        "point");
    graph.Resize(1);
    local_stats.inserted_nodes = 1;
    if (stats != nullptr) {
      stats->Combine(local_stats);
    }
    return {new_node, new_node};
  }

  HYPERVEC_THROW_IF_NOT_MSG(
      entry_point >= 0 && static_cast<size_t>(entry_point) < old_node_count,
      "NSWIncrementalBuilder: entry point is outside the graph");

  VisitedTable visited(old_node_count, false);
  const GraphSearcher searcher(graph);
  const std::array<GraphId, 1> entry_points = {entry_point};
  const std::vector<GraphSearchResult> search_results = searcher.Search(
      distance, entry_points,
      GraphSearchOptions{options_.ef_construction,
                         options_.check_relative_distance, nullptr},
      &visited, &local_stats.search);

  std::vector<NeighborCandidate> candidates;
  candidates.reserve(search_results.size());
  for (const GraphSearchResult& result : search_results) {
    candidates.push_back({result.id, result.distance});
  }
  const std::vector<NeighborCandidate> selected = pruner_.Prune(
      candidates, graph.MaxDegree(), distance, &local_stats.pruning);
  HYPERVEC_THROW_IF_NOT_MSG(
      !selected.empty(),
      "NSWIncrementalBuilder: search returned no usable neighbor");

  std::vector<PendingNeighborUpdate> reciprocal_updates;
  reciprocal_updates.reserve(selected.size());
  for (const NeighborCandidate& neighbor : selected) {
    const GraphNeighborList current = graph.Neighbors(neighbor.id);
    std::vector<GraphId> updated(current.begin(), current.end());
    if (updated.size() < graph.MaxDegree()) {
      updated.push_back(new_node);
    } else {
      std::vector<NeighborCandidate> reciprocal_candidates;
      reciprocal_candidates.reserve(updated.size() + 1);
      for (GraphId existing : updated) {
        reciprocal_candidates.push_back(
            {existing, distance.symmetric_dis(neighbor.id, existing)});
        ++local_stats.reciprocal_distance_computations;
      }
      reciprocal_candidates.push_back(
          {new_node, distance.symmetric_dis(neighbor.id, new_node)});
      ++local_stats.reciprocal_distance_computations;
      updated =
          CandidateIds(pruner_.Prune(reciprocal_candidates, graph.MaxDegree(),
                                     distance, &local_stats.pruning));
      ++local_stats.pruned_neighbor_lists;
    }
    reciprocal_updates.push_back({neighbor.id, std::move(updated)});
  }

  const std::vector<GraphId> outgoing = CandidateIds(selected);
  graph.Resize(old_node_count + 1);
  graph.SetNeighbors(new_node, outgoing);
  for (const PendingNeighborUpdate& update : reciprocal_updates) {
    graph.SetNeighbors(update.node, update.neighbors);
  }

  local_stats.inserted_nodes = 1;
  local_stats.reciprocal_updates = reciprocal_updates.size();
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return {new_node, entry_point};
}

}  // namespace hypervec
