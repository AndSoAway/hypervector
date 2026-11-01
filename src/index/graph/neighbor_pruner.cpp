/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/neighbor_pruner.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace hypervec {
namespace {

bool CandidateOrder(const NeighborCandidate& lhs,
                    const NeighborCandidate& rhs) noexcept {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

}  // namespace

void GraphPruneStats::Reset() noexcept { *this = {}; }

void GraphPruneStats::Combine(const GraphPruneStats& other) noexcept {
  candidates_examined += other.candidates_examined;
  distance_computations += other.distance_computations;
  accepted += other.accepted;
  rejected += other.rejected;
  filled += other.filled;
}

std::vector<NeighborCandidate> HnswHeuristicPruner::Prune(
    std::span<const NeighborCandidate> candidates, size_t max_neighbors,
    DistanceComputer& distance, GraphPruneStats* stats) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      max_neighbors > 0, "HnswHeuristicPruner: max_neighbors must be positive");

  std::vector<NeighborCandidate> ordered(candidates.begin(), candidates.end());
  std::unordered_set<GraphId> ids;
  ids.reserve(ordered.size());
  for (const NeighborCandidate& candidate : ordered) {
    HYPERVEC_THROW_IF_NOT_MSG(
        candidate.id >= 0,
        "HnswHeuristicPruner: candidate ID must be non-negative");
    HYPERVEC_THROW_IF_NOT_MSG(
        !std::isnan(candidate.distance),
        "HnswHeuristicPruner: candidate distance must not be NaN");
    HYPERVEC_THROW_IF_NOT_MSG(ids.insert(candidate.id).second,
                              "HnswHeuristicPruner: duplicate candidate ID");
  }
  std::sort(ordered.begin(), ordered.end(), CandidateOrder);

  GraphPruneStats local_stats;
  std::vector<NeighborCandidate> selected;
  std::vector<NeighborCandidate> rejected;
  selected.reserve(std::min(max_neighbors, ordered.size()));
  if (fill_to_capacity_) {
    rejected.reserve(ordered.size());
  }

  for (const NeighborCandidate& candidate : ordered) {
    ++local_stats.candidates_examined;
    bool diverse = true;
    for (const NeighborCandidate& existing : selected) {
      ++local_stats.distance_computations;
      if (distance.symmetric_dis(existing.id, candidate.id) <
          candidate.distance) {
        diverse = false;
        break;
      }
    }
    if (diverse) {
      selected.push_back(candidate);
      ++local_stats.accepted;
      if (selected.size() == max_neighbors) {
        break;
      }
    } else {
      ++local_stats.rejected;
      if (fill_to_capacity_) {
        rejected.push_back(candidate);
      }
    }
  }

  if (fill_to_capacity_) {
    for (const NeighborCandidate& candidate : rejected) {
      if (selected.size() == max_neighbors) {
        break;
      }
      selected.push_back(candidate);
      ++local_stats.filled;
    }
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return selected;
}

}  // namespace hypervec
