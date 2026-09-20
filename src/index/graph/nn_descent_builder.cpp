/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/neighbor_pruner.h>
#include <index/graph/nn_descent_builder.h>
#include <index/graph/visited_table.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

using NeighborList = std::vector<NeighborCandidate>;

bool CandidateOrder(const NeighborCandidate& lhs,
                    const NeighborCandidate& rhs) noexcept {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

void ValidateDistance(float distance) {
  HYPERVEC_THROW_IF_NOT_MSG(
      !std::isnan(distance),
      "NNDescentBuilder: distance computation returned NaN");
}

void AddCandidate(GraphId candidate, size_t node_count, VisitedTable* seen,
                  std::vector<GraphId>* candidates) {
  HYPERVEC_THROW_IF_NOT_MSG(
      candidate >= 0 && static_cast<size_t>(candidate) < node_count,
      "NNDescentBuilder: candidate ID is outside the graph");
  if (seen->set(static_cast<size_t>(candidate))) {
    candidates->push_back(candidate);
  }
}

std::vector<GraphId> NeighborIds(const NeighborList& neighbors) {
  std::vector<GraphId> ids;
  ids.reserve(neighbors.size());
  for (const NeighborCandidate& neighbor : neighbors) {
    ids.push_back(neighbor.id);
  }
  return ids;
}

size_t CountNewNeighbors(const NeighborList& previous,
                         const NeighborList& replacement) {
  size_t updates = 0;
  for (const NeighborCandidate& candidate : replacement) {
    const auto found =
        std::find_if(previous.begin(), previous.end(),
                     [&](const auto& old) { return old.id == candidate.id; });
    if (found == previous.end()) {
      ++updates;
    }
  }
  return updates;
}

size_t SampleBounded(std::mt19937_64* random, size_t bound) {
  const uint64_t unsigned_bound = static_cast<uint64_t>(bound);
  const uint64_t rejection_threshold =
      (uint64_t{0} - unsigned_bound) % unsigned_bound;
  uint64_t sample = 0;
  do {
    sample = (*random)();
  } while (sample < rejection_threshold);
  return static_cast<size_t>(sample % unsigned_bound);
}

}  // namespace

void NNDescentStats::Reset() noexcept { *this = {}; }

void NNDescentStats::Combine(const NNDescentStats& other) noexcept {
  iterations += other.iterations;
  initial_distance_computations += other.initial_distance_computations;
  refinement_distance_computations += other.refinement_distance_computations;
  neighbor_updates += other.neighbor_updates;
  converged = converged || other.converged;
}

NNDescentBuilder::NNDescentBuilder(NNDescentOptions options)
    : options_(options) {
  HYPERVEC_THROW_IF_NOT_MSG(options_.max_degree > 0,
                            "NNDescentBuilder: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.max_iterations > 0,
      "NNDescentBuilder: max_iterations must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(options_.convergence_threshold) &&
          options_.convergence_threshold >= 0.0 &&
          options_.convergence_threshold <= 1.0,
      "NNDescentBuilder: convergence_threshold must be in [0, 1]");
}

MutableBoundedGraph NNDescentBuilder::Build(DistanceComputer& distance,
                                            size_t node_count,
                                            NNDescentStats* stats) const {
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(
      node_count <= kGraphCapacity,
      "NNDescentBuilder: node count exceeds GraphId capacity");

  MutableBoundedGraph graph(node_count, options_.max_degree);
  NNDescentStats local_stats;
  if (node_count <= 1) {
    local_stats.converged = true;
    if (stats != nullptr) {
      stats->Combine(local_stats);
    }
    return graph;
  }

  const size_t degree = std::min(options_.max_degree, node_count - 1);
  std::vector<NeighborList> neighborhoods(node_count);
  std::mt19937_64 random(options_.random_seed);
  VisitedTable sampled(node_count, false);

  for (size_t node = 0; node < node_count; ++node) {
    std::vector<GraphId> initial;
    initial.reserve(degree);
    sampled.set(node);
    if (degree == node_count - 1) {
      for (size_t candidate = 0; candidate < node_count; ++candidate) {
        if (candidate != node) {
          initial.push_back(static_cast<GraphId>(candidate));
        }
      }
    } else {
      while (initial.size() < degree) {
        const size_t candidate = SampleBounded(&random, node_count);
        if (sampled.set(candidate)) {
          initial.push_back(static_cast<GraphId>(candidate));
        }
      }
    }

    NeighborList& neighbors = neighborhoods[node];
    neighbors.reserve(degree);
    for (GraphId candidate : initial) {
      const float candidate_distance =
          distance.symmetric_dis(static_cast<GraphId>(node), candidate);
      ValidateDistance(candidate_distance);
      neighbors.push_back({candidate, candidate_distance});
      ++local_stats.initial_distance_computations;
    }
    std::sort(neighbors.begin(), neighbors.end(), CandidateOrder);
    sampled.advance();
  }

  const size_t edge_slots =
      mul_no_overflow(node_count, degree, "NNDescentBuilder edge slots");
  VisitedTable seen(node_count, false);
  for (size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
    std::vector<std::vector<GraphId>> reverse(node_count);
    for (size_t node = 0; node < node_count; ++node) {
      for (const NeighborCandidate& neighbor : neighborhoods[node]) {
        reverse[static_cast<size_t>(neighbor.id)].push_back(
            static_cast<GraphId>(node));
      }
    }

    std::vector<NeighborList> refined(node_count);
    size_t iteration_updates = 0;
    for (size_t node = 0; node < node_count; ++node) {
      std::vector<GraphId> candidates;
      candidates.reserve(degree);
      seen.set(node);
      for (const NeighborCandidate& neighbor : neighborhoods[node]) {
        AddCandidate(neighbor.id, node_count, &seen, &candidates);
      }
      for (GraphId incoming : reverse[node]) {
        AddCandidate(incoming, node_count, &seen, &candidates);
      }

      const std::vector<GraphId> first_hop = candidates;
      for (GraphId adjacent : first_hop) {
        for (const NeighborCandidate& neighbor :
             neighborhoods[static_cast<size_t>(adjacent)]) {
          AddCandidate(neighbor.id, node_count, &seen, &candidates);
        }
        for (GraphId incoming : reverse[static_cast<size_t>(adjacent)]) {
          AddCandidate(incoming, node_count, &seen, &candidates);
        }
      }

      NeighborList evaluated;
      evaluated.reserve(candidates.size());
      for (GraphId candidate : candidates) {
        const float candidate_distance =
            distance.symmetric_dis(static_cast<GraphId>(node), candidate);
        ValidateDistance(candidate_distance);
        evaluated.push_back({candidate, candidate_distance});
        ++local_stats.refinement_distance_computations;
      }
      if (evaluated.size() > degree) {
        std::partial_sort(evaluated.begin(), evaluated.begin() + degree,
                          evaluated.end(), CandidateOrder);
        evaluated.resize(degree);
      } else {
        std::sort(evaluated.begin(), evaluated.end(), CandidateOrder);
      }
      NeighborList& replacement = refined[node];
      replacement.reserve(degree);
      replacement.assign(evaluated.begin(), evaluated.end());
      iteration_updates += CountNewNeighbors(neighborhoods[node], replacement);
      seen.advance();
    }

    neighborhoods = std::move(refined);
    ++local_stats.iterations;
    local_stats.neighbor_updates += iteration_updates;
    if (static_cast<double>(iteration_updates) <=
        options_.convergence_threshold * static_cast<double>(edge_slots)) {
      local_stats.converged = true;
      break;
    }
  }

  for (size_t node = 0; node < node_count; ++node) {
    const std::vector<GraphId> ids = NeighborIds(neighborhoods[node]);
    graph.SetNeighbors(static_cast<GraphId>(node), ids);
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return graph;
}

}  // namespace hypervec
