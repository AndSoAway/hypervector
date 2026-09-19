/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/vamana_builder.h>
#include <index/graph/visited_table.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
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

float ValidateDistance(float distance) {
  HYPERVEC_THROW_IF_NOT_MSG(!std::isnan(distance),
                            "VamanaBuilder: distance computation returned NaN");
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

bool Contains(std::span<const GraphId> neighbors, GraphId candidate) {
  return std::find(neighbors.begin(), neighbors.end(), candidate) !=
         neighbors.end();
}

std::vector<GraphId> NeighborIds(const NeighborList& neighbors) {
  std::vector<GraphId> result;
  result.reserve(neighbors.size());
  for (const NeighborCandidate& neighbor : neighbors) {
    result.push_back(neighbor.id);
  }
  return result;
}

NeighborList CollectCandidates(
    const MutableBoundedGraph& graph, GraphId source,
    std::span<const GraphSearchResult> search_results,
    NodeQueryDistanceComputer& query_distance, VisitedTable* pooled,
    size_t candidate_pool_size, VamanaBuildStats* stats) {
  NeighborList candidates;
  const GraphNeighborView previous = graph.Neighbors(source);
  candidates.reserve(add_no_overflow(search_results.size(), previous.size(),
                                     "VamanaBuilder candidate pool"));
  pooled->set(static_cast<size_t>(source));
  for (const GraphSearchResult& result : search_results) {
    if (pooled->set(static_cast<size_t>(result.id))) {
      candidates.push_back({result.id, ValidateDistance(result.distance)});
    }
  }
  for (GraphId neighbor : previous) {
    if (pooled->set(static_cast<size_t>(neighbor))) {
      candidates.push_back({neighbor, query_distance(neighbor)});
      ++stats->candidate_distance_computations;
    }
  }
  std::sort(candidates.begin(), candidates.end(), CandidateOrder);
  if (candidates.size() > candidate_pool_size) {
    candidates.resize(candidate_pool_size);
  }
  pooled->advance();
  return candidates;
}

void InsertReciprocal(MutableBoundedGraph* graph, GraphId source,
                      GraphId neighbor, DistanceComputer& distance,
                      const VamanaRobustPruner& pruner, size_t max_degree,
                      VamanaBuildStats* stats) {
  const GraphNeighborView current = graph->Neighbors(neighbor);
  if (Contains(current, source)) {
    return;
  }
  if (current.size() < max_degree) {
    graph->AddNeighbor(neighbor, source);
    ++stats->reciprocal_edges_added;
    return;
  }

  NeighborList candidates;
  candidates.reserve(current.size() + 1);
  for (GraphId candidate : current) {
    candidates.push_back({candidate, ValidateDistance(distance.symmetric_dis(
                                         neighbor, candidate))});
    ++stats->candidate_distance_computations;
  }
  candidates.push_back(
      {source, ValidateDistance(distance.symmetric_dis(neighbor, source))});
  ++stats->candidate_distance_computations;
  NodeQueryDistanceComputer query_distance(distance, neighbor);
  const NeighborList selected =
      pruner.Prune(candidates, max_degree, query_distance, &stats->pruning);
  const std::vector<GraphId> ids = NeighborIds(selected);
  graph->SetNeighbors(neighbor, ids);
  ++stats->reciprocal_edges_repruned;
  if (Contains(graph->Neighbors(neighbor), source)) {
    ++stats->reciprocal_edges_added;
  } else {
    ++stats->reciprocal_edges_rejected;
  }
}

struct Reachability {
  std::vector<bool> reachable;
  std::vector<GraphId> parent;
  size_t count = 0;
};

Reachability FindReachable(const GraphStorage& graph, GraphId entry_point) {
  Reachability result;
  result.reachable.assign(graph.NodeCount(), false);
  result.parent.assign(graph.NodeCount(), kInvalidGraphId);
  std::queue<GraphId> pending;
  result.reachable[static_cast<size_t>(entry_point)] = true;
  pending.push(entry_point);
  while (!pending.empty()) {
    const GraphId source = pending.front();
    pending.pop();
    ++result.count;
    for (GraphId neighbor : graph.Neighbors(source)) {
      if (!result.reachable[static_cast<size_t>(neighbor)]) {
        result.reachable[static_cast<size_t>(neighbor)] = true;
        result.parent[static_cast<size_t>(neighbor)] = source;
        pending.push(neighbor);
      }
    }
  }
  return result;
}

bool HasRepairCapacity(const MutableBoundedGraph& graph, GraphId source,
                       size_t max_degree, std::span<const GraphId> parent) {
  const GraphNeighborView neighbors = graph.Neighbors(source);
  if (neighbors.size() < max_degree) {
    return true;
  }
  return std::any_of(neighbors.begin(), neighbors.end(), [&](GraphId neighbor) {
    return parent[static_cast<size_t>(neighbor)] != source;
  });
}

GraphId SelectReplacement(const MutableBoundedGraph& graph, GraphId source,
                          std::span<const GraphId> parent,
                          DistanceComputer& distance, VamanaBuildStats* stats) {
  GraphId replacement = kInvalidGraphId;
  float farthest = -(std::numeric_limits<float>::infinity)();
  for (GraphId neighbor : graph.Neighbors(source)) {
    if (parent[static_cast<size_t>(neighbor)] == source) {
      continue;
    }
    const float candidate_distance =
        ValidateDistance(distance.symmetric_dis(source, neighbor));
    ++stats->connectivity_distance_computations;
    if (replacement == kInvalidGraphId || candidate_distance > farthest ||
        (candidate_distance == farthest && neighbor > replacement)) {
      replacement = neighbor;
      farthest = candidate_distance;
    }
  }
  return replacement;
}

void RepairConnectivity(MutableBoundedGraph* graph, GraphId entry_point,
                        size_t max_degree, DistanceComputer& distance,
                        VamanaBuildStats* stats) {
  Reachability reachability = FindReachable(*graph, entry_point);
  while (reachability.count < graph->NodeCount()) {
    const auto target_position = std::find(reachability.reachable.begin(),
                                           reachability.reachable.end(), false);
    const GraphId target =
        static_cast<GraphId>(target_position - reachability.reachable.begin());
    GraphId best_source = kInvalidGraphId;
    float best_distance = (std::numeric_limits<float>::infinity)();
    const auto consider_source = [&](GraphId source) {
      if (!reachability.reachable[static_cast<size_t>(source)] ||
          !HasRepairCapacity(*graph, source, max_degree, reachability.parent)) {
        return;
      }
      const float candidate_distance =
          ValidateDistance(distance.symmetric_dis(source, target));
      ++stats->connectivity_distance_computations;
      if (best_source == kInvalidGraphId ||
          candidate_distance < best_distance ||
          (candidate_distance == best_distance && source < best_source)) {
        best_source = source;
        best_distance = candidate_distance;
      }
    };
    for (GraphId source : graph->Neighbors(target)) {
      consider_source(source);
    }
    if (best_source == kInvalidGraphId) {
      for (size_t source = 0; source < graph->NodeCount(); ++source) {
        if (!reachability.reachable[source]) {
          continue;
        }
        consider_source(static_cast<GraphId>(source));
      }
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        best_source != kInvalidGraphId,
        "VamanaBuilder: no bounded edge is available for connectivity repair");

    if (graph->Neighbors(best_source).size() == max_degree) {
      const GraphId replacement = SelectReplacement(
          *graph, best_source, reachability.parent, distance, stats);
      HYPERVEC_THROW_IF_NOT_MSG(
          replacement != kInvalidGraphId,
          "VamanaBuilder: connectivity repair would remove a tree edge");
      HYPERVEC_THROW_IF_NOT_MSG(
          graph->RemoveNeighbor(best_source, replacement),
          "VamanaBuilder: connectivity replacement edge disappeared");
      ++stats->connectivity_edges_replaced;
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        graph->AddNeighbor(best_source, target),
        "VamanaBuilder: connectivity edge already exists");
    ++stats->connectivity_edges_added;
    reachability = FindReachable(*graph, entry_point);
  }
}

}  // namespace

void VamanaBuildStats::Reset() noexcept { *this = {}; }

void VamanaBuildStats::Combine(const VamanaBuildStats& other) noexcept {
  passes_completed += other.passes_completed;
  nodes_processed += other.nodes_processed;
  candidate_distance_computations += other.candidate_distance_computations;
  reciprocal_edges_added += other.reciprocal_edges_added;
  reciprocal_edges_repruned += other.reciprocal_edges_repruned;
  reciprocal_edges_rejected += other.reciprocal_edges_rejected;
  connectivity_distance_computations +=
      other.connectivity_distance_computations;
  connectivity_edges_added += other.connectivity_edges_added;
  connectivity_edges_replaced += other.connectivity_edges_replaced;
  search.Combine(other.search);
  pruning.Combine(other.pruning);
}

VamanaBuilder::VamanaBuilder(VamanaBuildOptions options) : options_(options) {
  HYPERVEC_THROW_IF_NOT_MSG(options_.max_degree > 0,
                            "VamanaBuilder: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.search_width >= options_.max_degree,
      "VamanaBuilder: search_width must cover max_degree");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.candidate_pool_size >= options_.search_width,
      "VamanaBuilder: candidate_pool_size must cover search_width");
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(options_.alpha) && options_.alpha >= 1.0F,
      "VamanaBuilder: alpha must be finite and at least one");
  HYPERVEC_THROW_IF_NOT_MSG(options_.build_passes > 0,
                            "VamanaBuilder: build_passes must be positive");
}

MutableBoundedGraph VamanaBuilder::Build(DistanceComputer& distance,
                                         size_t node_count,
                                         GraphId navigation_point,
                                         VamanaBuildStats* stats) const {
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(
      node_count <= kGraphCapacity,
      "VamanaBuilder: node count exceeds GraphId capacity");
  if (node_count == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        navigation_point == kInvalidGraphId,
        "VamanaBuilder: an empty graph requires an invalid navigation point");
    return MutableBoundedGraph(options_.max_degree);
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      navigation_point >= 0 &&
          static_cast<size_t>(navigation_point) < node_count,
      "VamanaBuilder: navigation point is outside the graph");

  MutableBoundedGraph graph(node_count, options_.max_degree);
  VamanaBuildStats local_stats;
  if (node_count == 1) {
    if (stats != nullptr) {
      stats->Combine(local_stats);
    }
    return graph;
  }

  const size_t degree = std::min(options_.max_degree, node_count - 1);
  const GraphSearcher searcher(graph);
  VisitedTable searched(node_count, false);
  VisitedTable pooled(node_count, false);
  std::vector<GraphId> order(node_count);
  for (size_t node = 0; node < node_count; ++node) {
    order[node] = static_cast<GraphId>(node);
  }
  std::mt19937_64 random(options_.random_seed);
  const std::array<GraphId, 1> entry_points = {navigation_point};

  for (size_t pass = 0; pass < options_.build_passes; ++pass) {
    std::shuffle(order.begin(), order.end(), random);
    const bool bootstrap_pass = options_.build_passes > 1 && pass == 0;
    const VamanaRobustPruner pruner(bootstrap_pass ? 1.0F : options_.alpha);
    for (GraphId node : order) {
      NodeQueryDistanceComputer query_distance(distance, node);
      const std::vector<GraphSearchResult> search_results = searcher.Search(
          query_distance, entry_points,
          GraphSearchOptions{options_.search_width, false, nullptr}, &searched,
          &local_stats.search);
      const NeighborList candidates = CollectCandidates(
          graph, node, search_results, query_distance, &pooled,
          options_.candidate_pool_size, &local_stats);
      const NeighborList selected = pruner.Prune(
          candidates, degree, query_distance, &local_stats.pruning);
      const std::vector<GraphId> ids = NeighborIds(selected);
      graph.SetNeighbors(node, ids);
      for (const NeighborCandidate& neighbor : selected) {
        InsertReciprocal(&graph, node, neighbor.id, distance, pruner, degree,
                         &local_stats);
      }
      ++local_stats.nodes_processed;
    }
    ++local_stats.passes_completed;
  }
  RepairConnectivity(&graph, navigation_point, degree, distance, &local_stats);

  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return graph;
}

}  // namespace hypervec
