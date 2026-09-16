/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/engine.h>

#include <array>
#include <iostream>
#include <memory>

int main() {
  const char* config = R"({
    "type": "hnsw",
    "dimension": 4,
    "metric": "l2",
    "storage": {"type": "flat"},
    "build": {"graph_degree": 8, "ef_construction": 40},
    "defaults": {"ef_search": 16},
    "id_mode": "external"
  })";
  std::unique_ptr<hypervec::Index> index = hypervec::CreateIndex(config);

  const std::array<float, 16> vectors = {0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                                         0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F,
                                         3.0F, 3.0F, 3.0F, 3.0F};
  const std::array<hypervec::idx_t, 4> ids = {101, 205, 309, 413};
  index->Build({4, 4, vectors.data(), ids.data()});

  const std::array<float, 4> query = {0.8F, 0.1F, 0.0F, 0.0F};
  std::array<float, 2> distances{};
  std::array<hypervec::idx_t, 2> labels{};
  index->Search(1, query.data(), 2, distances.data(), labels.data());

  std::cout << "created: " << index->TypeName() << '\n';
  for (size_t i = 0; i < labels.size(); ++i) {
    std::cout << labels[i] << " " << distances[i] << '\n';
  }
  return 0;
}
