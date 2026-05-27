/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/lvq/lvq_distance_computer.h>

#include <utils/log/assert.h>

namespace hypervec {

LVQDistanceComputer::LVQDistanceComputer(const LocalVectorQuantizer& lvq,
                                         const uint8_t* codes,
                                         size_t code_size)
  : lvq(lvq), codes(codes), code_size(code_size) {
  HYPERVEC_THROW_IF_NOT(lvq.is_trained);
  HYPERVEC_THROW_IF_NOT(code_size == lvq.code_size);
}

void LVQDistanceComputer::SetQuery(const float* x) {
  const size_t table_size =
    static_cast<size_t>(lvq.nlocal) * static_cast<size_t>(lvq.ksub);
  if (dis_table.size() != table_size) {
    dis_table.resize(table_size);
  }
  lvq.ComputeDistanceTable(x, dis_table.data());
}

float LVQDistanceComputer::operator()(idx_t i) {
  HYPERVEC_THROW_IF_NOT(!dis_table.empty());
  return lvq.ApplyDistanceTable(dis_table.data(),
                                codes + static_cast<size_t>(i) * code_size);
}

float LVQDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
  const size_t d = static_cast<size_t>(lvq.d);
  if (decode_buf_a.size() != d) {
    decode_buf_a.resize(d);
    decode_buf_b.resize(d);
  }
  lvq.Decode(codes + static_cast<size_t>(i) * code_size, decode_buf_a.data());
  lvq.Decode(codes + static_cast<size_t>(j) * code_size, decode_buf_b.data());
  float acc = 0.0f;
  for (size_t k = 0; k < d; k++) {
    const float diff = decode_buf_a[k] - decode_buf_b[k];
    acc += diff * diff;
  }
  return acc;
}

}  // namespace hypervec
