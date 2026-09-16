/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/visited_table.h>

#include <cstring>

namespace hypervec {

size_t visited_table_hashset_threshold = 500000;

VisitedTable::VisitedTable(size_t size, std::optional<bool> use_hashset)
    : visno(use_hashset.value_or(size >= visited_table_hashset_threshold) ? 0
                                                                          : 1) {
  if (visno != 0) {
    visited.resize(size, 0);
  }
}

void VisitedTable::advance() {
  if (visno == 0) {
    visited_set.clear();
  } else if (visno < 254) {
    ++visno;
  } else {
    std::memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
    visno = 1;
  }
}

}  // namespace hypervec
