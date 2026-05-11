/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/pq/pq.h>
#include <utils/distances/distance_computer.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hypervec {

/** Random-access ADC distance computer over a flat array of PQ codes.
 *
 *  SetQuery(x) builds the per-query L2-squared lookup table (size M * ksub
 *  floats) once; subsequent operator()(i) reads code i from `codes` and sums
 *  ApplyDistanceTable. symmetric_dis(i, j) decodes both codes back to floats
 *  and computes a plain L2 — a fallback path; SDC tables are a future knob.
 *
 *  Lifetime: borrows `pq` and `codes` — caller (typically IndexPQ /
 *  IndexHNSWPQ) must keep them alive for the lifetime of this object.
 *  Constructed via IndexPQ::GetDistanceComputer(); not thread-safe (mirrors
 *  the DistanceComputer contract — instantiate one per thread).
 */
struct PQDistanceComputer : DistanceComputer {
  const ProductQuantizer& pq;
  const uint8_t* codes;     ///< borrowed; size n_total * code_size bytes
  size_t code_size;         ///< bytes per encoded vector

  /// Per-query ADC table, sized M * ksub on first SetQuery call.
  std::vector<float> dis_table;

  /// Decode buffers used by symmetric_dis. Sized to pq.d on first use.
  std::vector<float> decode_buf_a;
  std::vector<float> decode_buf_b;

  PQDistanceComputer(const ProductQuantizer& pq, const uint8_t* codes,
                     size_t code_size);

  void SetQuery(const float* x) override;

  float operator()(idx_t i) override;

  float symmetric_dis(idx_t i, idx_t j) override;
};

}  // namespace hypervec
