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
#include <array>
#include <queue>
#include <unordered_set>
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

void ValidateSearchRequest(const GraphStorage& graph,
                           const GraphSearchOptions& options,
                           const VisitedTable* visited) {
  HYPERVEC_THROW_IF_NOT_MSG(options.ef_search > 0,
                            "GraphSearcher: ef_search must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(visited != nullptr,
                            "GraphSearcher: visited table must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      visited->Size() == graph.NodeCount(),
      "GraphSearcher: visited table size does not match graph");
}

void ValidateEntry(const GraphStorage& graph, GraphId entry) {
  HYPERVEC_THROW_IF_NOT_MSG(
      entry >= 0 && static_cast<size_t>(entry) < graph.NodeCount(),
      "GraphSearcher: entry point is outside the graph");
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
  ValidateSearchRequest(graph_, options, visited);
  for (GraphId entry : entry_points) {
    ValidateEntry(graph_, entry);
  }

  std::vector<GraphSearchSeed> seeds;
  seeds.reserve(entry_points.size());
  std::unordered_set<GraphId> unique_entries;
  unique_entries.reserve(entry_points.size());
  for (GraphId entry : entry_points) {
    if (unique_entries.insert(entry).second) {
      seeds.push_back({entry, distance(entry)});
    }
  }
  return SearchPrepared(distance, seeds, seeds.size(), options, visited, stats);
}

std::vector<GraphSearchResult> GraphSearcher::Search(
    DistanceComputer& distance, std::span<const GraphSearchSeed> seeds,
    const GraphSearchOptions& options, VisitedTable* visited,
    GraphSearchStats* stats) const {
  ValidateSearchRequest(graph_, options, visited);
  for (const GraphSearchSeed& seed : seeds) {
    ValidateEntry(graph_, seed.id);
  }

  std::vector<GraphSearchSeed> unique_seeds;
  unique_seeds.reserve(seeds.size());
  std::unordered_set<GraphId> unique_entries;
  unique_entries.reserve(seeds.size());
  for (const GraphSearchSeed& seed : seeds) {
    if (unique_entries.insert(seed.id).second) {
      unique_seeds.push_back(seed);
    }
  }
  return SearchPrepared(distance, unique_seeds, 0, options, visited, stats);
}

std::vector<GraphSearchResult> GraphSearcher::SearchPrepared(
    DistanceComputer& distance, std::span<const GraphSearchSeed> seeds,
    size_t seed_distance_computations, const GraphSearchOptions& options,
    VisitedTable* visited, GraphSearchStats* stats) const {
  GraphSearchStats local_stats;
  local_stats.queries = 1;
  local_stats.distance_computations = seed_distance_computations;
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

  const auto add_candidate = [&](GraphId id, float candidate_distance) {
    const GraphSearchResult candidate{id, candidate_distance};
    const bool within_frontier = results.size() < options.ef_search ||
                                 candidate.distance <= results.top().distance;
    if (!within_frontier) {
      return;
    }
    candidates.push(candidate);
    add_result(candidate);
    graph_.Prefetch(id);
  };

  for (const GraphSearchSeed& seed : seeds) {
    visited->set(static_cast<size_t>(seed.id));
    ++local_stats.visited_nodes;
    const GraphSearchResult candidate{seed.id, seed.distance};
    candidates.push(candidate);
    add_result(candidate);
    graph_.Prefetch(seed.id);
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
    std::array<GraphId, 4> distance_batch;
    size_t batch_size = 0;
    for (GraphId neighbor : neighbors) {
      if (!visited->set(static_cast<size_t>(neighbor))) {
        continue;
      }
      distance_batch[batch_size++] = neighbor;
      if (batch_size == distance_batch.size()) {
        std::array<float, 4> distances;
        distance.distances_batch_4(distance_batch[0], distance_batch[1],
                                   distance_batch[2], distance_batch[3],
                                   distances[0], distances[1], distances[2],
                                   distances[3]);
        local_stats.distance_computations += distance_batch.size();
        local_stats.visited_nodes += distance_batch.size();
        for (size_t index = 0; index < distance_batch.size(); ++index) {
          add_candidate(distance_batch[index], distances[index]);
        }
        batch_size = 0;
      }
    }
    for (size_t index = 0; index < batch_size; ++index) {
      add_candidate(distance_batch[index], distance(distance_batch[index]));
      ++local_stats.distance_computations;
      ++local_stats.visited_nodes;
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
