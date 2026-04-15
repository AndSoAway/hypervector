/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#ifndef HYPERVEC_VISITED_TABLE_H
#define HYPERVEC_VISITED_TABLE_H

#include <core/platform_macros.h>
#include <stdint.h>
#include <utils/structures/prefetch.h>

#include <optional>
#include <unordered_set>
#include <vector>

namespace hypervec {

HYPERVEC_API extern size_t visited_table_hashset_threshold;

/// A fast, reusable Visited Set for graph Search algorithms.
struct VisitedTable {
  std::vector<uint8_t> visited;
  std::unordered_set<size_t> visited_set;
  uint8_t visno;  // 0 if using visited_set, 1..250 if using vector.

  // If use_hashset is nullopt, the use of a hashset will be determined by
  // size >= visited_table_hashset_threshold.
  explicit VisitedTable(size_t size,
                        std::optional<bool> use_hashset = std::nullopt);

  /// set flag #no to true, return whether this changed it.
  bool set(size_t no) {
    if (visno == 0) {
      return visited_set.insert(no).second;
    } else if (visited[no] == visno) {
      return false;
    } else {
      visited[no] = visno;
      return true;
    }
  }

  /// get flag #no
  bool get(size_t no) const {
    if (visno == 0) {
      return visited_set.count(no) != 0;
    } else {
      return visited[no] == visno;
    }
  }

  void prefetch(size_t no) const {
    if (visno != 0) {
      prefetch_L2(&visited[no]);
    }
  }

  /// Reset all flags to false
  void advance();
};

}  // namespace hypervec

#endif
