/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/common/platform_macros.h>
#include <utils/structures/prefetch.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

namespace hypervec {

HYPERVEC_API extern size_t visited_table_hashset_threshold;

/** Fast reusable visited set shared by graph search algorithms. */
struct VisitedTable {
  std::vector<uint8_t> visited;
  std::unordered_set<size_t> visited_set;
  uint8_t visno;

  explicit VisitedTable(size_t size,
                        std::optional<bool> use_hashset = std::nullopt);

  bool set(size_t node) {
    if (visno == 0) {
      return visited_set.insert(node).second;
    }
    if (visited[node] == visno) {
      return false;
    }
    visited[node] = visno;
    return true;
  }

  bool get(size_t node) const {
    if (visno == 0) {
      return visited_set.count(node) != 0;
    }
    return visited[node] == visno;
  }

  void prefetch(size_t node) const {
    if (visno != 0) {
      prefetch_L2(&visited[node]);
    }
  }

  void advance();
};

}  // namespace hypervec
