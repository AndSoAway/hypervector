/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/lvq/lvq.h>
#include <utils/distances/distance_computer.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hypervec {

struct LVQDistanceComputer : DistanceComputer {
  const LocalVectorQuantizer& lvq;
  const uint8_t* codes;
  size_t code_size;
  std::vector<float> dis_table;
  std::vector<float> decode_buf_a;
  std::vector<float> decode_buf_b;

  LVQDistanceComputer(const LocalVectorQuantizer& lvq, const uint8_t* codes,
                      size_t code_size);

  void SetQuery(const float* x) override;
  float operator()(idx_t i) override;
  float symmetric_dis(idx_t i, idx_t j) override;
};

}  // namespace hypervec
