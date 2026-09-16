/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>

#include <cstddef>

namespace hypervec {

struct GraphValidationReport {
  size_t node_count = 0;
  size_t edge_count = 0;
  size_t max_observed_degree = 0;
  size_t out_of_range_edges = 0;
  size_t self_loops = 0;
  size_t duplicate_edges = 0;
  size_t degree_violations = 0;
  size_t weakly_connected_components = 0;
  size_t reachable_nodes = 0;

  bool IsStructurallyValid() const noexcept;
  double ReachableRatio() const noexcept;
};

/** Inspect structural invariants, weak components, and directed reachability.
 *
 * A negative entry point disables the directed reachability pass.
 */
GraphValidationReport ValidateGraph(const GraphStorage& graph,
                                    GraphId entry_point = kInvalidGraphId);

}  // namespace hypervec
