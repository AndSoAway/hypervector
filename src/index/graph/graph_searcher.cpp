/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <utils/log/assert.h>
#include <utils/selector/id_selector.h>

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

struct CloserFirst {
  bool operator()(const GraphSearchResult& lhs,
                  const GraphSearchResult& rhs) const noexcept {
    if (lhs.distance != rhs.distance) {
      return lhs.distance > rhs.distance;
    }
    return lhs.id > rhs.id;
  }
};

struct FartherFirst {
  bool operator()(const GraphSearchResult& lhs,
                  const GraphSearchResult& rhs) const noexcept {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.id < rhs.id;
  }
};

bool ResultOrder(const GraphSearchResult& lhs,
                 const GraphSearchResult& rhs) noexcept {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

}  // namespace

void GraphSearchStats::Reset() noexcept { *this = {}; }

void GraphSearchStats::Combine(const GraphSearchStats& other) noexcept {
  queries += other.queries;
  exhausted_queries += other.exhausted_queries;
  distance_computations += other.distance_computations;
  visited_nodes += other.visited_nodes;
  expanded_nodes += other.expanded_nodes;
  traversed_edges += other.traversed_edges;
}

std::vector<GraphSearchResult> GraphSearcher::Search(
    DistanceComputer& distance, std::span<const GraphId> entry_points,
    const GraphSearchOptions& options, VisitedTable* visited,
    GraphSearchStats* stats) const {
  HYPERVEC_THROW_IF_NOT_MSG(options.ef_search > 0,
                            "GraphSearcher: ef_search must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(visited != nullptr,
                            "GraphSearcher: visited table must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      visited->Size() == graph_.NodeCount(),
      "GraphSearcher: visited table size does not match graph");
  for (GraphId entry : entry_points) {
    HYPERVEC_THROW_IF_NOT_MSG(
        entry >= 0 && static_cast<size_t>(entry) < graph_.NodeCount(),
        "GraphSearcher: entry point is outside the graph");
  }

  GraphSearchStats local_stats;
  local_stats.queries = 1;
  visited->advance();

  std::priority_queue<GraphSearchResult, std::vector<GraphSearchResult>,
                      CloserFirst>
      candidates;
  std::priority_queue<GraphSearchResult, std::vector<GraphSearchResult>,
                      FartherFirst>
      results;

  const auto add_result = [&](const GraphSearchResult& candidate) {
    if (options.selector != nullptr &&
        !options.selector->IsMember(candidate.id)) {
      return;
    }
    results.push(candidate);
    if (results.size() > options.ef_search) {
      results.pop();
    }
  };

  for (GraphId entry : entry_points) {
    if (!visited->set(static_cast<size_t>(entry))) {
      continue;
    }
    const GraphSearchResult seed{entry, distance(entry)};
    ++local_stats.distance_computations;
    ++local_stats.visited_nodes;
    candidates.push(seed);
    add_result(seed);
    graph_.Prefetch(entry);
  }

  bool stopped_early = false;
  while (!candidates.empty()) {
    const GraphSearchResult current = candidates.top();
    if (options.check_relative_distance &&
        results.size() >= options.ef_search &&
        current.distance > results.top().distance) {
      stopped_early = true;
      break;
    }
    candidates.pop();

    const GraphNeighborList neighbors = graph_.Neighbors(current.id);
    ++local_stats.expanded_nodes;
    local_stats.traversed_edges += neighbors.size();
    for (GraphId neighbor : neighbors) {
      visited->prefetch(static_cast<size_t>(neighbor));
    }
    for (GraphId neighbor : neighbors) {
      if (!visited->set(static_cast<size_t>(neighbor))) {
        continue;
      }
      const GraphSearchResult candidate{neighbor, distance(neighbor)};
      ++local_stats.distance_computations;
      ++local_stats.visited_nodes;

      const bool within_frontier = results.size() < options.ef_search ||
                                   candidate.distance <= results.top().distance;
      if (!within_frontier) {
        continue;
      }
      candidates.push(candidate);
      add_result(candidate);
      graph_.Prefetch(neighbor);
    }
  }

  if (!stopped_early && candidates.empty()) {
    local_stats.exhausted_queries = 1;
  }

  std::vector<GraphSearchResult> ordered;
  ordered.reserve(results.size());
  while (!results.empty()) {
    ordered.push_back(results.top());
    results.pop();
  }
  std::sort(ordered.begin(), ordered.end(), ResultOrder);
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return ordered;
}

}  // namespace hypervec
