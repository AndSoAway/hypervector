/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/neighbor_pruner.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <vector>

namespace hypervec {

void GraphPruneStats::Combine(const GraphPruneStats& other) noexcept {
  candidates_examined += other.candidates_examined;
  distance_computations += other.distance_computations;
  accepted += other.accepted;
  rejected += other.rejected;
}

std::vector<NeighborCandidate> NearestNeighborPruner::Prune(
    std::span<const NeighborCandidate> candidates, size_t max_degree,
    DistanceComputer& /* distance */, GraphPruneStats* stats) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      max_degree > 0, "NearestNeighborPruner: max_degree must be positive");

  GraphPruneStats local_stats;
  local_stats.candidates_examined = candidates.size();
  local_stats.accepted = std::min(candidates.size(), max_degree);
  local_stats.rejected = candidates.size() - local_stats.accepted;
  for (const NeighborCandidate& candidate : candidates) {
    HYPERVEC_THROW_IF_NOT_MSG(
        candidate.id >= 0,
        "NearestNeighborPruner: candidate id must be non-negative");
  }

  std::vector<NeighborCandidate> selected;
  selected.reserve(local_stats.accepted);
  selected.insert(selected.end(), candidates.begin(),
                  candidates.begin() + local_stats.accepted);
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return selected;
}

std::vector<NeighborCandidate> HnswHeuristicPruner::Prune(
    std::span<const NeighborCandidate> candidates, size_t max_degree,
    DistanceComputer& distance, GraphPruneStats* stats) const {
  HYPERVEC_THROW_IF_NOT_MSG(max_degree > 0,
                            "HnswHeuristicPruner: max_degree must be positive");

  GraphPruneStats local_stats;
  std::vector<NeighborCandidate> selected;
  std::vector<NeighborCandidate> rejected;
  selected.reserve(max_degree);
  if (fill_to_max_degree_) {
    rejected.reserve(candidates.size());
  }

  for (const NeighborCandidate& candidate : candidates) {
    HYPERVEC_THROW_IF_NOT_MSG(
        candidate.id >= 0,
        "HnswHeuristicPruner: candidate id must be non-negative");
    ++local_stats.candidates_examined;
    bool keep = true;
    for (const NeighborCandidate& existing : selected) {
      ++local_stats.distance_computations;
      if (distance.symmetric_dis(existing.id, candidate.id) <
          candidate.distance) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push_back(candidate);
      ++local_stats.accepted;
      if (selected.size() == max_degree) {
        break;
      }
    } else {
      ++local_stats.rejected;
      if (fill_to_max_degree_) {
        rejected.push_back(candidate);
      }
    }
  }

  for (size_t i = 0; fill_to_max_degree_ && selected.size() < max_degree &&
                     i < rejected.size();
       ++i) {
    selected.push_back(rejected[i]);
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return selected;
}

}  // namespace hypervec
