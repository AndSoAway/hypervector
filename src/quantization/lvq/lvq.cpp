/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/lvq/lvq.h>

#include <utils/algo/kmeans/kmeans.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>
#include <utils/log/exception.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

int BitsForCount(idx_t n) {
  HYPERVEC_THROW_IF_NOT(n > 0);
  int bits = 0;
  idx_t value = n - 1;
  while (value > 0) {
    bits++;
    value >>= 1;
  }
  return std::max(bits, 1);
}

idx_t Nearest(const float* x, const float* y, idx_t n, idx_t d,
              float* scratch) {
  return static_cast<idx_t>(fvec_L2sqr_ny_nearest(
    scratch, x, y, static_cast<size_t>(d), static_cast<size_t>(n)));
}

void DecodePair(const LocalVectorQuantizer& lvq, const uint8_t* code,
                idx_t* local_id, idx_t* code_id) {
  PQDecoderGeneric all(code, lvq.local_nbits + lvq.nbits);
  const uint64_t combined = all.decode();
  const uint64_t local_mask =
    (static_cast<uint64_t>(1) << lvq.local_nbits) - 1;
  *local_id = static_cast<idx_t>(combined & local_mask);
  *code_id = static_cast<idx_t>(combined >> lvq.local_nbits);
}

void EncodePair(const LocalVectorQuantizer& lvq, idx_t local_id,
                idx_t code_id, uint8_t* code) {
  std::memset(code, 0, lvq.code_size);
  PQEncoderGeneric enc(code, lvq.local_nbits + lvq.nbits);
  const uint64_t combined =
    static_cast<uint64_t>(local_id) |
    (static_cast<uint64_t>(code_id) << lvq.local_nbits);
  enc.encode(combined);
}

}  // namespace

LocalVectorQuantizer::LocalVectorQuantizer(idx_t d, idx_t nlocal, int nbits)
  : d(d), nlocal(nlocal), nbits(nbits) {
  SetDerivedValues();
}

void LocalVectorQuantizer::SetDerivedValues() {
  HYPERVEC_THROW_IF_NOT_FMT(d > 0, "LocalVectorQuantizer: d must be > 0, got %ld",
                            static_cast<long>(d));
  HYPERVEC_THROW_IF_NOT_FMT(
    nlocal > 0, "LocalVectorQuantizer: nlocal must be > 0, got %ld",
    static_cast<long>(nlocal));
  HYPERVEC_THROW_IF_NOT_FMT(
    nbits >= 1 && nbits <= HYPERVEC_LVQ_MAX_NBITS,
    "LocalVectorQuantizer: nbits (%d) must be in [1, %d]", nbits,
    HYPERVEC_LVQ_MAX_NBITS);
  local_nbits = BitsForCount(nlocal);
  HYPERVEC_THROW_IF_NOT_MSG(
    local_nbits + nbits < 64,
    "LocalVectorQuantizer: combined local/code bit width must be < 64");
  ksub = static_cast<idx_t>(1) << nbits;
  code_size = (static_cast<size_t>(local_nbits + nbits) + 7) / 8;
  local_centroids.resize(static_cast<size_t>(nlocal) * d);
  residual_codebooks.resize(static_cast<size_t>(nlocal) * ksub * d);
}

void LocalVectorQuantizer::Train(idx_t n, const float* x,
                                 const LVQParameters& params) {
  HYPERVEC_THROW_IF_NOT_FMT(
    n >= nlocal,
    "LocalVectorQuantizer::Train: need at least nlocal=%ld vectors, got %ld",
    static_cast<long>(nlocal), static_cast<long>(n));
  HYPERVEC_THROW_IF_NOT_MSG(params.niter > 0,
                            "LocalVectorQuantizer::Train: niter must be > 0");
  HYPERVEC_THROW_IF_NOT_MSG(params.nredo > 0,
                            "LocalVectorQuantizer::Train: nredo must be > 0");

  KMeansParameters kp;
  kp.niter = params.niter;
  kp.seed = params.seed;
  kp.nredo = params.nredo;
  kp.verbose = params.verbose;
  kp.metric = kMetricL2;
  RunKMeans(n, x, d, nlocal, local_centroids.data(), kp);

  std::vector<float> assign_dis(static_cast<size_t>(n));
  std::vector<idx_t> assign_ids(static_cast<size_t>(n));
  float_maxheap_array_t res = {static_cast<size_t>(n), 1, assign_ids.data(),
                               assign_dis.data()};
  knn_L2sqr(x, local_centroids.data(), static_cast<size_t>(d),
            static_cast<size_t>(n), static_cast<size_t>(nlocal), &res);

  std::vector<std::vector<float>> residuals(static_cast<size_t>(nlocal));
  for (idx_t i = 0; i < n; i++) {
    const idx_t local_id = assign_ids[static_cast<size_t>(i)];
    auto& bucket = residuals[static_cast<size_t>(local_id)];
    const float* centroid = GetLocalCentroid(local_id);
    const float* xi = x + i * d;
    for (idx_t j = 0; j < d; j++) {
      bucket.push_back(xi[j] - centroid[j]);
    }
  }

  std::vector<std::pair<int, std::exception_ptr>> exceptions;
#pragma omp parallel for if (nlocal > 1)
  for (idx_t local_id = 0; local_id < nlocal; local_id++) {
    try {
      auto& bucket = residuals[static_cast<size_t>(local_id)];
      const idx_t bucket_n = static_cast<idx_t>(bucket.size() / d);
      float* out = GetResidualCodeword(local_id, 0);
      if (bucket_n >= ksub) {
        KMeansParameters local_kp = kp;
        local_kp.seed = params.seed + 1009 + static_cast<int>(local_id);
        RunKMeans(bucket_n, bucket.data(), d, ksub, out, local_kp);
      } else {
        std::fill(out, out + static_cast<size_t>(ksub) * d, 0.0f);
        if (bucket_n > 0) {
          for (idx_t k = 0; k < ksub; k++) {
            std::memcpy(out + k * d, bucket.data() + (k % bucket_n) * d,
                        static_cast<size_t>(d) * sizeof(float));
          }
        }
      }
    } catch (...) {
#pragma omp critical
      exceptions.emplace_back(static_cast<int>(local_id),
                              std::current_exception());
    }
  }
  handleExceptions(exceptions);
  is_trained = true;
}

void LocalVectorQuantizer::ComputeCode(const float* x, uint8_t* code) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  std::vector<float> scratch(static_cast<size_t>(std::max(nlocal, ksub)));
  const idx_t local_id =
    Nearest(x, local_centroids.data(), nlocal, d, scratch.data());
  std::vector<float> residual(static_cast<size_t>(d));
  const float* centroid = GetLocalCentroid(local_id);
  for (idx_t j = 0; j < d; j++) {
    residual[static_cast<size_t>(j)] = x[j] - centroid[j];
  }
  const idx_t code_id = Nearest(residual.data(), GetResidualCodeword(local_id, 0),
                                ksub, d, scratch.data());
  EncodePair(*this, local_id, code_id, code);
}

void LocalVectorQuantizer::ComputeCodes(idx_t n, const float* x,
                                        uint8_t* codes) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
#pragma omp parallel for if (n > 1)
  for (idx_t i = 0; i < n; i++) {
    ComputeCode(x + i * d, codes + static_cast<size_t>(i) * code_size);
  }
}

void LocalVectorQuantizer::DecodeCode(const uint8_t* code, idx_t* local_id,
                                      idx_t* code_id) const {
  DecodePair(*this, code, local_id, code_id);
  HYPERVEC_THROW_IF_NOT(*local_id >= 0 && *local_id < nlocal);
  HYPERVEC_THROW_IF_NOT(*code_id >= 0 && *code_id < ksub);
}

void LocalVectorQuantizer::Decode(const uint8_t* code, float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  idx_t local_id;
  idx_t code_id;
  DecodeCode(code, &local_id, &code_id);
  const float* centroid = GetLocalCentroid(local_id);
  const float* residual = GetResidualCodeword(local_id, code_id);
  for (idx_t j = 0; j < d; j++) {
    x[j] = centroid[j] + residual[j];
  }
}

void LocalVectorQuantizer::DecodeBatch(idx_t n, const uint8_t* codes,
                                       float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
#pragma omp parallel for if (n > 1)
  for (idx_t i = 0; i < n; i++) {
    Decode(codes + static_cast<size_t>(i) * code_size, x + i * d);
  }
}

void LocalVectorQuantizer::ComputeDistanceTable(const float* x,
                                                float* dis_table) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  std::vector<float> decoded(static_cast<size_t>(d));
  for (idx_t local_id = 0; local_id < nlocal; local_id++) {
    const float* centroid = GetLocalCentroid(local_id);
    for (idx_t code_id = 0; code_id < ksub; code_id++) {
      const float* residual = GetResidualCodeword(local_id, code_id);
      for (idx_t j = 0; j < d; j++) {
        decoded[static_cast<size_t>(j)] = centroid[j] + residual[j];
      }
      dis_table[local_id * ksub + code_id] =
        fvec_L2sqr(x, decoded.data(), static_cast<size_t>(d));
    }
  }
}

float LocalVectorQuantizer::ApplyDistanceTable(const float* dis_table,
                                               const uint8_t* code) const {
  idx_t local_id;
  idx_t code_id;
  DecodeCode(code, &local_id, &code_id);
  return dis_table[local_id * ksub + code_id];
}

void LocalVectorQuantizer::SearchL2(idx_t nx, const float* x, idx_t ncodes,
                                    const uint8_t* codes, idx_t k,
                                    float* distances, idx_t* labels) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT(k > 0);
  const size_t table_sz = static_cast<size_t>(nlocal) * ksub;

#pragma omp parallel
  {
    std::vector<float> dis_table(table_sz);
#pragma omp for
    for (idx_t qi = 0; qi < nx; qi++) {
      ComputeDistanceTable(x + qi * d, dis_table.data());
      float* heap_dis = distances + qi * k;
      idx_t* heap_ids = labels + qi * k;
      heap_heapify<CMax<float, idx_t>>(k, heap_dis, heap_ids);

      float threshold = heap_dis[0];
      for (idx_t j = 0; j < ncodes; j++) {
        const float dis =
          ApplyDistanceTable(dis_table.data(),
                             codes + static_cast<size_t>(j) * code_size);
        if (CMax<float, idx_t>::cmp(threshold, dis)) {
          heap_replace_top<CMax<float, idx_t>>(k, heap_dis, heap_ids, dis, j);
          threshold = heap_dis[0];
        }
      }
      heap_reorder<CMax<float, idx_t>>(k, heap_dis, heap_ids);
    }
  }
}

}  // namespace hypervec
