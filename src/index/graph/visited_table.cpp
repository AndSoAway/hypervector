/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/visited_table.h>

#include <algorithm>

namespace hypervec {

// Vector access is faster, while a hash set avoids initializing a very large
// array for searches that visit only a small fraction of its nodes.
size_t visited_table_hashset_threshold = 500000;

VisitedTable::VisitedTable(size_t size, std::optional<bool> use_hashset)
    : visno(use_hashset.value_or(size >= visited_table_hashset_threshold) ? 0
                                                                          : 1),
      size_(size) {
  if (visno != 0) {
    visited.resize(size, 0);
  }
}

void VisitedTable::advance() {
  if (visno == 0) {
    visited_set.clear();
  } else if (visno < 254) {
    // 254 rather than 255 because HNSW may use visno + 1 temporarily.
    ++visno;
  } else {
    std::fill(visited.begin(), visited.end(), uint8_t{0});
    visno = 1;
  }
}

}  // namespace hypervec
