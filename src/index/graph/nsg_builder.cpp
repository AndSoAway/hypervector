/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_validation.h>
#include <index/graph/nsg_builder.h>
#include <index/graph/visited_table.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
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

float ValidateDistance(float distance) {
  HYPERVEC_THROW_IF_NOT_MSG(!std::isnan(distance),
                            "NSGBuilder: distance computation returned NaN");
  return distance;
}

class NodeQueryDistanceComputer final : public DistanceComputer {
 public:
  NodeQueryDistanceComputer(DistanceComputer& distance, GraphId query)
      : distance_(distance), query_(query) {}

  void SetQuery(const float* /*query*/) override {}

  float operator()(idx_t index) override {
    return ValidateDistance(distance_.symmetric_dis(query_, index));
  }

  float symmetric_dis(idx_t lhs, idx_t rhs) override {
    return ValidateDistance(distance_.symmetric_dis(lhs, rhs));
  }

 private:
  DistanceComputer& distance_;
  GraphId query_;
};

bool Contains(const NeighborList& neighbors, GraphId candidate) {
  return std::any_of(
      neighbors.begin(), neighbors.end(),
      [&](const auto& neighbor) { return neighbor.id == candidate; });
}

void AddCandidate(const NeighborCandidate& candidate, VisitedTable* pooled,
                  NeighborList* candidates) {
  if (pooled->set(static_cast<size_t>(candidate.id))) {
    candidates->push_back(candidate);
  }
}

std::vector<bool> FindReachable(const std::vector<NeighborList>& neighborhoods,
                                size_t node_count, GraphId entry_point,
                                size_t* reachable_count) {
  std::vector<bool> reachable(node_count, false);
  std::queue<GraphId> pending;
  reachable[static_cast<size_t>(entry_point)] = true;
  pending.push(entry_point);
  *reachable_count = 0;
  while (!pending.empty()) {
    const GraphId node = pending.front();
    pending.pop();
    ++*reachable_count;
    for (const NeighborCandidate& neighbor :
         neighborhoods[static_cast<size_t>(node)]) {
      if (!reachable[static_cast<size_t>(neighbor.id)]) {
        reachable[static_cast<size_t>(neighbor.id)] = true;
        pending.push(neighbor.id);
      }
    }
  }
  return reachable;
}

std::vector<GraphId> NeighborIds(const NeighborList& neighbors) {
  std::vector<GraphId> result;
  result.reserve(neighbors.size());
  for (const NeighborCandidate& neighbor : neighbors) {
    result.push_back(neighbor.id);
  }
  return result;
}

}  // namespace

void NSGBuildStats::Reset() noexcept { *this = {}; }

void NSGBuildStats::Combine(const NSGBuildStats& other) noexcept {
  pruned_nodes += other.pruned_nodes;
  candidate_distance_computations += other.candidate_distance_computations;
  reciprocal_edges_added += other.reciprocal_edges_added;
  reciprocal_edges_rejected += other.reciprocal_edges_rejected;
  connectivity_distance_computations +=
      other.connectivity_distance_computations;
  connectivity_edges_added += other.connectivity_edges_added;
  search.Combine(other.search);
  pruning.Combine(other.pruning);
}

NSGBuilder::NSGBuilder(NSGBuildOptions options)
    : options_(options), pruner_(false) {
  HYPERVEC_THROW_IF_NOT_MSG(options_.max_degree > 0,
                            "NSGBuilder: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(options_.search_width > 0,
                            "NSGBuilder: search_width must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.candidate_pool_size >= options_.max_degree,
      "NSGBuilder: candidate_pool_size must cover max_degree");
}

MutableBoundedGraph NSGBuilder::Build(const GraphStorage& candidate_graph,
                                      DistanceComputer& distance,
                                      GraphId navigation_point,
                                      NSGBuildStats* stats) const {
  const size_t node_count = candidate_graph.NodeCount();
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(node_count <= kGraphCapacity,
                            "NSGBuilder: node count exceeds GraphId capacity");

  if (node_count == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        navigation_point == kInvalidGraphId,
        "NSGBuilder: an empty graph requires an invalid navigation point");
    return MutableBoundedGraph(1);
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      navigation_point >= 0 &&
          static_cast<size_t>(navigation_point) < node_count,
      "NSGBuilder: navigation point is outside the graph");
  const GraphValidationReport candidate_report = ValidateGraph(candidate_graph);
  HYPERVEC_THROW_IF_NOT_MSG(
      candidate_report.IsStructurallyValid(),
      "NSGBuilder: candidate graph is structurally invalid");

  const size_t pruned_degree = std::min(options_.max_degree, node_count - 1);
  const size_t graph_capacity =
      node_count == 1 ? 1 : std::min(node_count - 1, pruned_degree + 1);
  MutableBoundedGraph graph(node_count, graph_capacity);
  NSGBuildStats local_stats;
  if (node_count == 1) {
    if (stats != nullptr) {
      stats->Combine(local_stats);
    }
    return graph;
  }

  std::vector<NeighborList> neighborhoods(node_count);
  const GraphSearcher searcher(candidate_graph);
  VisitedTable searched(node_count, false);
  VisitedTable pooled(node_count, false);
  const std::array<GraphId, 1> entry_points = {navigation_point};
  for (size_t node = 0; node < node_count; ++node) {
    NodeQueryDistanceComputer query_distance(distance,
                                             static_cast<GraphId>(node));
    const std::vector<GraphSearchResult> search_results = searcher.Search(
        query_distance, entry_points,
        GraphSearchOptions{options_.search_width,
                           options_.check_relative_distance, nullptr},
        &searched, &local_stats.search);

    const GraphNeighborView original_neighbors =
        candidate_graph.Neighbors(static_cast<GraphId>(node));
    NeighborList candidates;
    candidates.reserve(add_no_overflow(search_results.size(),
                                       original_neighbors.size(),
                                       "NSGBuilder candidate pool"));
    pooled.set(node);
    for (const GraphSearchResult& result : search_results) {
      AddCandidate({result.id, ValidateDistance(result.distance)}, &pooled,
                   &candidates);
    }
    for (GraphId neighbor : original_neighbors) {
      if (pooled.set(static_cast<size_t>(neighbor))) {
        candidates.push_back({neighbor, query_distance(neighbor)});
        ++local_stats.candidate_distance_computations;
      }
    }
    std::sort(candidates.begin(), candidates.end(), CandidateOrder);
    if (candidates.size() > options_.candidate_pool_size) {
      candidates.resize(options_.candidate_pool_size);
    }
    neighborhoods[node] = pruner_.Prune(candidates, pruned_degree,
                                        query_distance, &local_stats.pruning);
    ++local_stats.pruned_nodes;
    pooled.advance();
  }

  const std::vector<NeighborList> directed = neighborhoods;
  for (size_t node = 0; node < node_count; ++node) {
    for (const NeighborCandidate& neighbor : directed[node]) {
      NeighborList& reciprocal =
          neighborhoods[static_cast<size_t>(neighbor.id)];
      const GraphId source = static_cast<GraphId>(node);
      if (Contains(reciprocal, source)) {
        continue;
      }
      const NeighborCandidate proposal{source, neighbor.distance};
      if (reciprocal.size() < pruned_degree) {
        reciprocal.push_back(proposal);
        ++local_stats.reciprocal_edges_added;
        continue;
      }

      NeighborList candidates = reciprocal;
      candidates.push_back(proposal);
      NodeQueryDistanceComputer query_distance(distance, neighbor.id);
      reciprocal = pruner_.Prune(candidates, pruned_degree, query_distance,
                                 &local_stats.pruning);
      if (Contains(reciprocal, source)) {
        ++local_stats.reciprocal_edges_added;
      } else {
        ++local_stats.reciprocal_edges_rejected;
      }
    }
  }

  size_t reachable_count = 0;
  std::vector<bool> reachable = FindReachable(
      neighborhoods, node_count, navigation_point, &reachable_count);
  while (reachable_count < node_count) {
    const auto target_position =
        std::find(reachable.begin(), reachable.end(), false);
    const GraphId target =
        static_cast<GraphId>(target_position - reachable.begin());
    GraphId best_source = kInvalidGraphId;
    float best_distance = (std::numeric_limits<float>::infinity)();
    for (size_t source = 0; source < node_count; ++source) {
      if (!reachable[source] ||
          neighborhoods[source].size() >= graph_capacity) {
        continue;
      }
      const float source_distance = ValidateDistance(
          distance.symmetric_dis(static_cast<GraphId>(source), target));
      ++local_stats.connectivity_distance_computations;
      if (best_source == kInvalidGraphId || source_distance < best_distance ||
          (source_distance == best_distance &&
           static_cast<GraphId>(source) < best_source)) {
        best_source = static_cast<GraphId>(source);
        best_distance = source_distance;
      }
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        best_source != kInvalidGraphId,
        "NSGBuilder: no capacity remains for connectivity repair");
    neighborhoods[static_cast<size_t>(best_source)].push_back(
        {target, best_distance});
    ++local_stats.connectivity_edges_added;
    reachable = FindReachable(neighborhoods, node_count, navigation_point,
                              &reachable_count);
  }

  for (size_t node = 0; node < node_count; ++node) {
    std::sort(neighborhoods[node].begin(), neighborhoods[node].end(),
              CandidateOrder);
    const std::vector<GraphId> ids = NeighborIds(neighborhoods[node]);
    graph.SetNeighbors(static_cast<GraphId>(node), ids);
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return graph;
}

}  // namespace hypervec
