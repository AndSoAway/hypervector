/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hypervec {

/** LVQ-b: mean-center, then quantize each vector's components using its own
 *  range and `nbits` per component. The code contains two float32 bounds
 *  (lower and step) followed by bit-packed scalar values. L2 only.
 */
struct LocalVectorQuantizer {
  idx_t d = 0;
  int nbits = 0;
  size_t code_size = 0;
  bool is_trained = false;
  std::vector<float> mean;

  LocalVectorQuantizer() = default;
  LocalVectorQuantizer(idx_t d, int nbits);

  void SetDerivedValues();
  void Train(idx_t n, const float* x);
  void ComputeCode(const float* x, uint8_t* code) const;
  void ComputeCodes(idx_t n, const float* x, uint8_t* codes) const;
  void Decode(const uint8_t* code, float* x) const;
  void DecodeBatch(idx_t n, const uint8_t* codes, float* x) const;
  // For LVQ the query buffer is a mean-centered query, rather than a PQ table.
  void ComputeDistanceTable(const float* x, float* query_buffer) const;
  float ApplyDistanceTable(const float* query_buffer,
                           const uint8_t* code) const;
  using DistanceKernel = float (*)(const LocalVectorQuantizer&, const float*,
                                   const uint8_t*);
  // Select once per scanner/search, not once per stored vector. The returned
  // function handles this quantizer's bit width and current CPU/OS support.
  DistanceKernel GetDistanceKernel() const;
  void SearchL2(idx_t nx, const float* x, idx_t ncodes, const uint8_t* codes,
                idx_t k, float* distances, idx_t* labels) const;
};

}  // namespace hypervec
