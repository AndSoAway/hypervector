/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstdint>
#include <vector>

namespace hypervec {

#define HYPERVEC_LVQ_DEFAULT_SEED 1234
#define HYPERVEC_LVQ_DEFAULT_NITER 25
#define HYPERVEC_LVQ_DEFAULT_NREDO 1
#define HYPERVEC_LVQ_MAX_NBITS 16

struct LVQParameters {
  int niter = HYPERVEC_LVQ_DEFAULT_NITER;
  int seed = HYPERVEC_LVQ_DEFAULT_SEED;
  int nredo = HYPERVEC_LVQ_DEFAULT_NREDO;
  bool verbose = false;
};

/** Local Vector Quantization codec.
 *
 *  The codec first assigns each vector to one of `nlocal` local centroids.
 *  Each local cell owns a residual codebook with `ksub = 1 << nbits`
 *  codewords. A code stores (local_id, residual_code_id), and reconstructs as
 *  local_centroid[local_id] + residual_codebook[local_id, residual_code_id].
 *
 *  T1 scope: kMetricL2 only.
 */
struct LocalVectorQuantizer {
  idx_t d = 0;
  idx_t nlocal = 0;
  int nbits = 0;
  int local_nbits = 0;
  idx_t ksub = 0;
  size_t code_size = 0;
  bool is_trained = false;

  std::vector<float> local_centroids;
  std::vector<float> residual_codebooks;

  LocalVectorQuantizer() = default;
  LocalVectorQuantizer(idx_t d, idx_t nlocal, int nbits);

  const float* GetLocalCentroid(idx_t local_id) const {
    return local_centroids.data() + local_id * d;
  }
  float* GetLocalCentroid(idx_t local_id) {
    return local_centroids.data() + local_id * d;
  }

  const float* GetResidualCodeword(idx_t local_id, idx_t code_id) const {
    return residual_codebooks.data() + (local_id * ksub + code_id) * d;
  }
  float* GetResidualCodeword(idx_t local_id, idx_t code_id) {
    return residual_codebooks.data() + (local_id * ksub + code_id) * d;
  }

  void SetDerivedValues();
  void Train(idx_t n, const float* x, const LVQParameters& params = {});
  void ComputeCode(const float* x, uint8_t* code) const;
  void ComputeCodes(idx_t n, const float* x, uint8_t* codes) const;
  void Decode(const uint8_t* code, float* x) const;
  void DecodeBatch(idx_t n, const uint8_t* codes, float* x) const;
  void ComputeDistanceTable(const float* x, float* dis_table) const;
  float ApplyDistanceTable(const float* dis_table, const uint8_t* code) const;
  void SearchL2(idx_t nx, const float* x, idx_t ncodes, const uint8_t* codes,
                idx_t k, float* distances, idx_t* labels) const;
  void DecodeCode(const uint8_t* code, idx_t* local_id,
                  idx_t* code_id) const;
};

}  // namespace hypervec
