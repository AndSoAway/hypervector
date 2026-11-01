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

/** Reusable visited-node state shared by graph search algorithms. */
class VisitedTable {
 public:
  /** Selects a vector or hash set automatically unless explicitly provided. */
  explicit VisitedTable(size_t size,
                        std::optional<bool> use_hashset = std::nullopt);

  size_t Size() const noexcept { return size_; }

  /** Mark a node and return true only on the first visit in this epoch. */
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

  /** Start a fresh epoch without clearing the vector on the common path. */
  void advance();

  // Kept public for compatibility with the existing HNSW implementation.
  std::vector<uint8_t> visited;
  std::unordered_set<size_t> visited_set;
  uint8_t visno;

 private:
  size_t size_;
};

}  // namespace hypervec
