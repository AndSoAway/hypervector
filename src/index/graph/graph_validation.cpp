/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_validation.h>

#include <algorithm>
#include <queue>
#include <unordered_set>
#include <vector>

namespace hypervec {

bool GraphValidationReport::IsStructurallyValid() const noexcept {
  return out_of_range_edges == 0 && self_loops == 0 && duplicate_edges == 0 &&
         degree_violations == 0;
}

double GraphValidationReport::ReachableRatio() const noexcept {
  if (node_count == 0) {
    return 1.0;
  }
  return static_cast<double>(reachable_nodes) / static_cast<double>(node_count);
}

GraphValidationReport ValidateGraph(const GraphStorage& graph,
                                    GraphId entry_point) {
  GraphValidationReport report;
  report.node_count = graph.NodeCount();
  std::vector<std::vector<GraphId>> valid_edges(report.node_count);
  std::vector<std::vector<GraphId>> undirected(report.node_count);

  for (size_t node = 0; node < report.node_count; ++node) {
    const GraphNeighborView neighbors =
        graph.Neighbors(static_cast<GraphId>(node));
    report.edge_count += neighbors.size();
    report.max_observed_degree =
        std::max(report.max_observed_degree, neighbors.size());
    if (neighbors.size() > graph.MaxDegree()) {
      ++report.degree_violations;
    }

    std::unordered_set<GraphId> seen;
    for (GraphId neighbor : neighbors) {
      if (neighbor < 0 || static_cast<size_t>(neighbor) >= report.node_count) {
        ++report.out_of_range_edges;
        continue;
      }
      if (neighbor == static_cast<GraphId>(node)) {
        ++report.self_loops;
        continue;
      }
      if (!seen.insert(neighbor).second) {
        ++report.duplicate_edges;
        continue;
      }
      valid_edges[node].push_back(neighbor);
      undirected[node].push_back(neighbor);
      undirected[static_cast<size_t>(neighbor)].push_back(
          static_cast<GraphId>(node));
    }
  }

  std::vector<bool> visited(report.node_count, false);
  std::queue<GraphId> pending;
  for (size_t start = 0; start < report.node_count; ++start) {
    if (visited[start]) {
      continue;
    }
    ++report.weakly_connected_components;
    visited[start] = true;
    pending.push(static_cast<GraphId>(start));
    while (!pending.empty()) {
      const GraphId current = pending.front();
      pending.pop();
      for (GraphId neighbor : undirected[static_cast<size_t>(current)]) {
        if (!visited[static_cast<size_t>(neighbor)]) {
          visited[static_cast<size_t>(neighbor)] = true;
          pending.push(neighbor);
        }
      }
    }
  }

  if (entry_point >= 0 &&
      static_cast<size_t>(entry_point) < report.node_count) {
    std::fill(visited.begin(), visited.end(), false);
    visited[static_cast<size_t>(entry_point)] = true;
    pending.push(entry_point);
    while (!pending.empty()) {
      const GraphId current = pending.front();
      pending.pop();
      ++report.reachable_nodes;
      for (GraphId neighbor : valid_edges[static_cast<size_t>(current)]) {
        if (!visited[static_cast<size_t>(neighbor)]) {
          visited[static_cast<size_t>(neighbor)] = true;
          pending.push(neighbor);
        }
      }
    }
  }
  return report;
}

}  // namespace hypervec
