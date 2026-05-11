/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/pq/pq.h>

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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hypervec {

// ===========================================================================
// Anonymous helpers
// ===========================================================================

namespace {

/// Encode one vector into `code` using the encoder type chosen by the caller.
/// `scratch` must have room for ksub floats and is reused for each
/// fvec_L2sqr_ny_nearest call across the M subquantizers.
template <typename Encoder>
void EncodeOneT(const ProductQuantizer& pq, const float* x, Encoder& enc,
                float* scratch) {
  for (idx_t m = 0; m < pq.M; m++) {
    const size_t k = fvec_L2sqr_ny_nearest(
      scratch, x + m * pq.dsub, pq.GetCentroids(m, 0),
      static_cast<size_t>(pq.dsub), static_cast<size_t>(pq.ksub));
    enc.encode(static_cast<uint64_t>(k));
  }
}

/// Decode one packed code back into a d-dim float vector by concatenating
/// looked-up subcentroids.
template <typename Decoder>
void DecodeOneT(const ProductQuantizer& pq, Decoder& dec, float* x) {
  for (idx_t m = 0; m < pq.M; m++) {
    const idx_t k = static_cast<idx_t>(dec.decode());
    std::memcpy(x + m * pq.dsub, pq.GetCentroids(m, k),
                static_cast<size_t>(pq.dsub) * sizeof(float));
  }
}

/// Dispatch on nbits to one of the three encoder paths and encode a single
/// vector. Branching once outside the M loop keeps the inner loop tight.
void ComputeCodeImpl(const ProductQuantizer& pq, const float* x, uint8_t* code,
                     float* scratch) {
  if (pq.nbits == 8) {
    PQEncoder8 enc(code);
    EncodeOneT(pq, x, enc, scratch);
  } else if (pq.nbits == 16) {
    PQEncoder16 enc(code);
    EncodeOneT(pq, x, enc, scratch);
  } else {
    PQEncoderGeneric enc(code, pq.nbits);
    EncodeOneT(pq, x, enc, scratch);
  }
}

/// Dispatch on nbits and decode a single packed code.
void DecodeImpl(const ProductQuantizer& pq, const uint8_t* code, float* x) {
  if (pq.nbits == 8) {
    PQDecoder8 dec(code);
    DecodeOneT(pq, dec, x);
  } else if (pq.nbits == 16) {
    PQDecoder16 dec(code);
    DecodeOneT(pq, dec, x);
  } else {
    PQDecoderGeneric dec(code, pq.nbits);
    DecodeOneT(pq, dec, x);
  }
}

}  // namespace

// ===========================================================================
// Construction & validation
// ===========================================================================

ProductQuantizer::ProductQuantizer(idx_t d, idx_t M, int nbits)
  : d(d), M(M), nbits(nbits) {
  SetDerivedValues();
}

void ProductQuantizer::SetDerivedValues() {
  HYPERVEC_THROW_IF_NOT_FMT(M > 0, "ProductQuantizer: M must be > 0, got %ld",
                            static_cast<long>(M));
  HYPERVEC_THROW_IF_NOT_FMT(d > 0, "ProductQuantizer: d must be > 0, got %ld",
                            static_cast<long>(d));
  HYPERVEC_THROW_IF_NOT_FMT(
    d % M == 0,
    "ProductQuantizer: d (%ld) must be divisible by M (%ld)",
    static_cast<long>(d), static_cast<long>(M));
  HYPERVEC_THROW_IF_NOT_FMT(
    nbits >= 1 && nbits <= HYPERVEC_PQ_MAX_NBITS,
    "ProductQuantizer: nbits (%d) must be in [1, %d]", nbits,
    HYPERVEC_PQ_MAX_NBITS);

  dsub = d / M;
  ksub = static_cast<idx_t>(1) << nbits;
  code_size =
    (static_cast<size_t>(M) * static_cast<size_t>(nbits) + 7) / 8;
  centroids.resize(static_cast<size_t>(M) * ksub * dsub);
  // Note: is_trained is intentionally NOT reset here. Constructor sets it
  // false; Train() sets it true; deserialization sets it true after the
  // centroid table is loaded.
}

// ===========================================================================
// Training
// ===========================================================================

void ProductQuantizer::Train(idx_t n, const float* x,
                             const PQParameters& params) {
  HYPERVEC_THROW_IF_NOT_FMT(
    n >= ksub,
    "ProductQuantizer::Train: need at least ksub=%ld training vectors, got %ld "
    "(consider lowering nbits or providing more data)",
    static_cast<long>(ksub), static_cast<long>(n));
  HYPERVEC_THROW_IF_NOT_FMT(params.niter > 0,
                            "ProductQuantizer::Train: niter must be > 0, got %d",
                            params.niter);
  HYPERVEC_THROW_IF_NOT_FMT(params.nredo > 0,
                            "ProductQuantizer::Train: nredo must be > 0, got %d",
                            params.nredo);

  KMeansParameters kp;
  kp.niter = params.niter;
  kp.nredo = params.nredo;
  kp.verbose = params.verbose;
  kp.metric = kMetricL2;  // T1 scope: L2 only

  // Subquantizers are independent — train them in parallel. Each thread
  // needs its own slice buffer (size n * dsub) to hold the m-th subvector
  // slice contiguously.
  std::vector<std::pair<int, std::exception_ptr>> exceptions;

#pragma omp parallel
  {
    std::vector<float> xslice(static_cast<size_t>(n) * dsub);

#pragma omp for
    for (idx_t m = 0; m < M; m++) {
      try {
        // Gather subvector m of every training vector into a contiguous
        // (n, dsub) matrix.
        for (idx_t i = 0; i < n; i++) {
          std::memcpy(xslice.data() + i * dsub, x + i * d + m * dsub,
                      static_cast<size_t>(dsub) * sizeof(float));
        }

        // Independent seed per subquantizer keeps the training reproducible
        // while preventing the same starting permutation across all m.
        kp.seed = params.seed + static_cast<int>(m);

        if (params.verbose) {
#ifdef _OPENMP
          // Verbose stderr from multiple threads gets interleaved; serialise.
#pragma omp critical
#endif
          {
            std::fprintf(stderr, "PQ training subquantizer %ld/%ld (dsub=%ld, "
                                 "ksub=%ld, n=%ld)\n",
                         static_cast<long>(m + 1), static_cast<long>(M),
                         static_cast<long>(dsub), static_cast<long>(ksub),
                         static_cast<long>(n));
          }
        }

        RunKMeans(n, xslice.data(), dsub, ksub, GetCentroids(m, 0), kp);
      } catch (...) {
#pragma omp critical
        exceptions.emplace_back(static_cast<int>(m), std::current_exception());
      }
    }
  }

  handleExceptions(exceptions);
  is_trained = true;
}

// ===========================================================================
// Encode / Decode
// ===========================================================================

void ProductQuantizer::ComputeCode(const float* x, uint8_t* code) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  std::vector<float> scratch(static_cast<size_t>(ksub));
  ComputeCodeImpl(*this, x, code, scratch.data());
}

void ProductQuantizer::ComputeCodes(idx_t n, const float* x,
                                    uint8_t* codes) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }

  std::vector<std::pair<int, std::exception_ptr>> exceptions;

#pragma omp parallel
  {
    std::vector<float> scratch(static_cast<size_t>(ksub));

#pragma omp for
    for (idx_t i = 0; i < n; i++) {
      try {
        ComputeCodeImpl(*this, x + i * d, codes + i * code_size, scratch.data());
      } catch (...) {
#pragma omp critical
        exceptions.emplace_back(static_cast<int>(i), std::current_exception());
      }
    }
  }

  handleExceptions(exceptions);
}

void ProductQuantizer::Decode(const uint8_t* code, float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  DecodeImpl(*this, code, x);
}

void ProductQuantizer::DecodeBatch(idx_t n, const uint8_t* codes,
                                   float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
#pragma omp parallel for if (n > 1)
  for (idx_t i = 0; i < n; i++) {
    DecodeImpl(*this, codes + i * code_size, x + i * d);
  }
}

// ===========================================================================
// ADC distance tables
// ===========================================================================

void ProductQuantizer::ComputeDistanceTable(const float* x,
                                            float* dis_table) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  // Per subquantizer m: dis_table[m*ksub..(m+1)*ksub) holds the L2² distances
  // from x's m-th subvector to all ksub centroids of subquantizer m.
  for (idx_t m = 0; m < M; m++) {
    fvec_L2sqr_ny(dis_table + m * ksub, x + m * dsub, GetCentroids(m, 0),
                  static_cast<size_t>(dsub), static_cast<size_t>(ksub));
  }
}

void ProductQuantizer::ComputeDistanceTables(idx_t nx, const float* x,
                                             float* dis_tables) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  const size_t table_sz = static_cast<size_t>(M) * ksub;
#pragma omp parallel for if (nx > 1)
  for (idx_t i = 0; i < nx; i++) {
    ComputeDistanceTable(x + i * d, dis_tables + i * table_sz);
  }
}

// ===========================================================================
// Brute-force PQ search (L2)
// ===========================================================================

float ProductQuantizer::ApplyDistanceTable(const float* dis_table,
                                           const uint8_t* code) const {
  float dis = 0.0f;
  if (nbits == 8) {
    for (idx_t m = 0; m < M; m++) {
      dis += dis_table[m * ksub + code[m]];
    }
  } else if (nbits == 16) {
    for (idx_t m = 0; m < M; m++) {
      const uint64_t k = static_cast<uint64_t>(code[2 * m]) |
                         (static_cast<uint64_t>(code[2 * m + 1]) << 8);
      dis += dis_table[m * ksub + static_cast<idx_t>(k)];
    }
  } else {
    PQDecoderGeneric dec(code, nbits);
    for (idx_t m = 0; m < M; m++) {
      const idx_t k = static_cast<idx_t>(dec.decode());
      dis += dis_table[m * ksub + k];
    }
  }
  return dis;
}

void ProductQuantizer::SearchL2(idx_t nx, const float* x, idx_t ncodes,
                                const uint8_t* codes, idx_t k,
                                float* distances, idx_t* labels) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT(k > 0);

  const size_t table_sz = static_cast<size_t>(M) * ksub;

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
          ApplyDistanceTable(dis_table.data(), codes + j * code_size);
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
