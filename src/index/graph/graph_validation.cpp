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
  return self_loops == 0 && duplicate_edges == 0 && out_of_range_edges == 0 &&
         degree_violations == 0 && invalid_entry_points == 0;
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
  std::vector<std::vector<GraphId>> valid_outgoing(report.node_count);
  std::vector<std::vector<GraphId>> undirected(report.node_count);

  for (size_t node = 0; node < report.node_count; ++node) {
    const GraphId graph_node = static_cast<GraphId>(node);
    const GraphNeighborList neighbors = graph.Neighbors(graph_node);
    report.edge_count += neighbors.size();
    if (neighbors.size() > graph.MaxDegree()) {
      ++report.degree_violations;
    }

    std::unordered_set<GraphId> seen;
    seen.reserve(neighbors.size());
    for (GraphId neighbor : neighbors) {
      if (neighbor < 0 || static_cast<size_t>(neighbor) >= report.node_count) {
        ++report.out_of_range_edges;
        continue;
      }
      if (!seen.insert(neighbor).second) {
        ++report.duplicate_edges;
        continue;
      }
      if (neighbor == graph_node) {
        ++report.self_loops;
        continue;
      }
      valid_outgoing[node].push_back(neighbor);
      undirected[node].push_back(neighbor);
      undirected[static_cast<size_t>(neighbor)].push_back(graph_node);
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
      const GraphId node = pending.front();
      pending.pop();
      for (GraphId neighbor : undirected[static_cast<size_t>(node)]) {
        if (!visited[static_cast<size_t>(neighbor)]) {
          visited[static_cast<size_t>(neighbor)] = true;
          pending.push(neighbor);
        }
      }
    }
  }

  if (entry_point == kInvalidGraphId) {
    return report;
  }
  if (entry_point < 0 ||
      static_cast<size_t>(entry_point) >= report.node_count) {
    report.invalid_entry_points = 1;
    return report;
  }

  std::fill(visited.begin(), visited.end(), false);
  visited[static_cast<size_t>(entry_point)] = true;
  pending.push(entry_point);
  while (!pending.empty()) {
    const GraphId node = pending.front();
    pending.pop();
    ++report.reachable_nodes;
    for (GraphId neighbor : valid_outgoing[static_cast<size_t>(node)]) {
      if (!visited[static_cast<size_t>(neighbor)]) {
        visited[static_cast<size_t>(neighbor)] = true;
        pending.push(neighbor);
      }
    }
  }
  return report;
}

}  // namespace hypervec
