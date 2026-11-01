/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/index_factory.h>

#include <array>
#include <cmath>
#include <memory>

int main() {
  constexpr int kDimension = 2;
  const std::array<float, 4> database = {0.0F, 0.0F, 10.0F, 10.0F};
  const std::array<float, 2> query = {1.0F, 1.0F};

  hypervec::IndexConfig config("flat", kDimension);
  std::unique_ptr<hypervec::Index> index = hypervec::CreateIndex(config);
  index->Add(2, database.data());

  std::array<float, 1> distances = {};
  std::array<hypervec::idx_t, 1> labels = {};
  index->Search(1, query.data(), 1, distances.data(), labels.data());

  if (labels[0] != 0 || std::abs(distances[0] - 2.0F) > 1e-6F) {
    return 1;
  }
  return 0;
}
