/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <utils/distances/metric_type.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace hypervec {

/// Project-wide default seed for k-means random initialization used by
/// subquantizer training. Mirrors HYPERVEC_KMEANS_DEFAULT_SEED so PQ and IVF
/// k-means draw from the same reproducible seed unless overridden.
#define HYPERVEC_PQ_DEFAULT_SEED 1234

/// Default Lloyd iteration count per subquantizer. Matches the IVF default
/// (HYPERVEC_KMEANS_DEFAULT_NITER) so subquantizer training has the same
/// convergence budget as the coarse quantizer.
#define HYPERVEC_PQ_DEFAULT_NITER 25

/// Default number of independent k-means restarts per subquantizer. PQ
/// training benefits from > 1 in practice (subquantizer search spaces are
/// small enough that nredo is cheap and reduces local-minimum risk), but
/// the conservative default of 1 matches the rest of the framework.
#define HYPERVEC_PQ_DEFAULT_NREDO 1

/// Hard upper bound on nbits. ksub = 1 << nbits, so nbits=16 caps each
/// subquantizer at 65 536 centroids — enough for all common configurations
/// while keeping the bit-packing register width well under 64 bits.
#define HYPERVEC_PQ_MAX_NBITS 16

/// Tunables for ProductQuantizer::Train. Defaults reproduce the most common
/// training configuration (L2 mean update, 25 Lloyd iterations, single
/// random restart).
struct PQParameters {
  /// Number of Lloyd iterations per subquantizer per redo.
  int niter = HYPERVEC_PQ_DEFAULT_NITER;

  /// RNG base seed. Subquantizer m uses seed = base + m so independent
  /// subquantizers have independent draws but the whole training is still
  /// reproducible from a single integer.
  int seed = HYPERVEC_PQ_DEFAULT_SEED;

  /// Number of independent random restarts per subquantizer; the run with
  /// the lowest objective is kept.
  int nredo = HYPERVEC_PQ_DEFAULT_NREDO;

  /// Print per-subquantizer progress to stderr.
  bool verbose = false;
};

/** Product Quantization codec.
 *
 *  Splits each input vector of dimension d into M contiguous subvectors of
 *  dimension dsub = d / M, then trains an independent k-means with
 *  ksub = 1 << nbits centroids on each subspace. A vector is encoded as the
 *  M centroid indices, packed bit-tight into ceil(M * nbits / 8) bytes.
 *
 *  Asymmetric Distance Computation (ADC): given a query x, build a lookup
 *  table dis_table[m, k] = || x_m - centroid_{m,k} ||² of size M * ksub
 *  floats once; the L2 distance from x to any code c is then the cheap sum
 *  Σ_m dis_table[m, c[m]]. This is the basis of IndexPQ::Search and the
 *  per-list scan in IndexIVFPQ::SearchPreassigned.
 *
 *  T1 scope: kMetricL2 only. IP / cosine support is deferred to a later
 *  phase (would require IP-spherical k-means and an inner-product table).
 *
 *  Memory layout of `centroids` (size M * ksub * dsub floats):
 *      centroids[(m * ksub + k) * dsub + j]  is component j of centroid k
 *                                            of subquantizer m.
 *  Use `GetCentroids(m, k)` to obtain a typed pointer.
 *
 *  This struct is forward-declared in `<persistence/index_io.h>` so that
 *  `read_ProductQuantizer` / `write_ProductQuantizer` can be exposed there.
 */
struct ProductQuantizer {
  /// Vector dimension. Must satisfy d % M == 0 (every subquantizer takes a
  /// dsub-sized contiguous slice).
  idx_t d = 0;

  /// Number of subquantizers (subvectors per vector).
  idx_t M = 0;

  /// Bits per subquantizer code. Range: [1, HYPERVEC_PQ_MAX_NBITS].
  int nbits = 0;

  /// Subvector dimension, derived from d and M (dsub = d / M).
  idx_t dsub = 0;

  /// Number of centroids per subquantizer, derived from nbits
  /// (ksub = 1 << nbits).
  idx_t ksub = 0;

  /// Bytes per encoded vector, derived as (M * nbits + 7) / 8.
  size_t code_size = 0;

  /// True after Train() has been called.
  bool is_trained = false;

  /// Per-subquantizer centroids, total size M * ksub * dsub floats.
  std::vector<float> centroids;

  ProductQuantizer() = default;

  /** Construct an untrained PQ with the given dimensions.
   *
   *  @param d      vector dimension; must be divisible by M
   *  @param M      number of subquantizers
   *  @param nbits  bits per subquantizer code (1..HYPERVEC_PQ_MAX_NBITS)
   *  @throws HypervecException on invalid (d, M, nbits) combination
   */
  ProductQuantizer(idx_t d, idx_t M, int nbits);

  /** Train every subquantizer on the given vectors.
   *
   *  Each subquantizer m runs an independent k-means on the slice
   *  x[:, m*dsub:(m+1)*dsub]. Subquantizers are trained in parallel
   *  via OpenMP since they share no state.
   *
   *  @param n       number of training vectors (must be >= ksub)
   *  @param x       training vectors, size n * d
   *  @param params  tunables; default reproduces baseline configuration
   *  @throws HypervecException if n < ksub or any subquantizer training fails
   */
  void Train(idx_t n, const float* x, const PQParameters& params = {});

  /// Convenience pointer accessor for centroid k of subquantizer m.
  const float* GetCentroids(idx_t m, idx_t k) const {
    return centroids.data() + (m * ksub + k) * dsub;
  }
  float* GetCentroids(idx_t m, idx_t k) {
    return centroids.data() + (m * ksub + k) * dsub;
  }

  /** Encode a single vector into `code` (size code_size bytes). */
  void ComputeCode(const float* x, uint8_t* code) const;

  /** Encode n vectors into `codes` (size n * code_size bytes). Parallelized
   *  via OpenMP across vectors. */
  void ComputeCodes(idx_t n, const float* x, uint8_t* codes) const;

  /** Decode a single code back to a d-dimensional float vector by
   *  concatenating the looked-up subcentroids. Lossy. */
  void Decode(const uint8_t* code, float* x) const;

  /** Decode n codes. */
  void DecodeBatch(idx_t n, const uint8_t* codes, float* x) const;

  /** Compute the L2-squared ADC lookup table for one query.
   *
   *  On output: dis_table[m * ksub + k] = || x_m - centroid_{m,k} ||²
   *  Size: M * ksub floats.
   */
  void ComputeDistanceTable(const float* x, float* dis_table) const;

  /** Batched ADC table compute over nx queries. Output layout:
   *  dis_tables[i * (M * ksub) + m * ksub + k] for query i. Parallelized
   *  via OpenMP across queries. */
  void ComputeDistanceTables(idx_t nx, const float* x,
                             float* dis_tables) const;

  /** Apply a precomputed distance table to one packed code:
   *      Σ_m dis_table[m * ksub + code[m]]
   *
   *  Specialised paths for nbits == 8 / 16 read codes as bytes / byte-pairs;
   *  otherwise PQDecoderGeneric unpacks bits on demand. Hot inner loop of
   *  every PQ-accelerated index variant. */
  float ApplyDistanceTable(const float* dis_table, const uint8_t* code) const;

  /** Brute-force PQ search under L2: for each of nx queries, scan all
   *  ncodes encoded vectors and keep the top-k smallest distances.
   *
   *  Distances accumulate from the per-query ADC table:
   *      dis(query_i, code_j) = Σ_m dis_table_i[m, code_j[m]]
   *
   *  Result heaps are initialised by this function (no heapify required
   *  from the caller). On output `distances` and `labels` are sorted
   *  ascending by distance (best first) per query.
   *
   *  @param nx        number of queries
   *  @param x         queries, size nx * d
   *  @param ncodes    number of database codes to scan
   *  @param codes     packed codes, size ncodes * code_size
   *  @param k         neighbours per query
   *  @param distances output distances, size nx * k
   *  @param labels    output ids (= row index in `codes`), size nx * k
   */
  void SearchL2(idx_t nx, const float* x, idx_t ncodes, const uint8_t* codes,
                idx_t k, float* distances, idx_t* labels) const;

  /** Recompute derived fields (dsub, ksub, code_size) from (d, M, nbits)
   *  and validate. Called by the constructor and by deserialization. */
  void SetDerivedValues();
};

// ===========================================================================
// Bit-tight code packing helpers (header-only, inlined into hot loops)
// ===========================================================================
//
// Three encoder/decoder pairs let callers branch once on nbits at the loop
// outside, instead of per-element. The byte-aligned cases (nbits == 8 / 16)
// shortcut to direct memory access; the generic path handles 1..16 bits via
// a 64-bit staging register.
//
// All three encoders advance an internal pointer; the destructor of
// PQEncoderGeneric flushes any partial trailing byte. Decoders advance
// likewise; no destructor needed.
//
// IMPORTANT for callers: write the unique-byte value into `code` first, then
// destruct (or let the encoder go out of scope) before reading `code`.

/** Generic packer for nbits in [1, 16]. Stages bits in a 64-bit register
 *  and emits whole bytes as they fill up. Worst case stage width is
 *  7 + 16 = 23 bits, well under 64. */
struct PQEncoderGeneric {
  uint8_t* code;     ///< current write position
  uint64_t reg;      ///< unflushed bits (low bits = next-to-emit)
  int n_buffered;    ///< number of bits currently held in reg (0..7)
  int nbits;         ///< bits per encoded value

  PQEncoderGeneric(uint8_t* code, int nbits)
    : code(code), reg(0), n_buffered(0), nbits(nbits) {}

  void encode(uint64_t x) {
    reg |= (x & ((static_cast<uint64_t>(1) << nbits) - 1))
           << n_buffered;
    n_buffered += nbits;
    while (n_buffered >= 8) {
      *code++ = static_cast<uint8_t>(reg & 0xFFu);
      reg >>= 8;
      n_buffered -= 8;
    }
  }

  ~PQEncoderGeneric() {
    if (n_buffered > 0) {
      *code = static_cast<uint8_t>(reg & 0xFFu);
    }
  }
};

/** Byte-aligned shortcut for nbits == 8. */
struct PQEncoder8 {
  uint8_t* code;
  explicit PQEncoder8(uint8_t* code) : code(code) {}
  void encode(uint64_t x) { *code++ = static_cast<uint8_t>(x); }
};

/** Two-byte shortcut for nbits == 16. Avoids unaligned uint16_t access by
 *  writing the low and high bytes explicitly. */
struct PQEncoder16 {
  uint8_t* code;
  explicit PQEncoder16(uint8_t* code) : code(code) {}
  void encode(uint64_t x) {
    code[0] = static_cast<uint8_t>(x & 0xFFu);
    code[1] = static_cast<uint8_t>((x >> 8) & 0xFFu);
    code += 2;
  }
};

/** Generic unpacker, inverse of PQEncoderGeneric. Pulls whole bytes into
 *  a 64-bit register on demand. */
struct PQDecoderGeneric {
  const uint8_t* code;
  uint64_t reg;
  int n_buffered;
  int nbits;
  uint64_t mask;

  PQDecoderGeneric(const uint8_t* code, int nbits)
    : code(code)
    , reg(0)
    , n_buffered(0)
    , nbits(nbits)
    , mask((static_cast<uint64_t>(1) << nbits) - 1) {}

  uint64_t decode() {
    while (n_buffered < nbits) {
      reg |= static_cast<uint64_t>(*code++) << n_buffered;
      n_buffered += 8;
    }
    const uint64_t v = reg & mask;
    reg >>= nbits;
    n_buffered -= nbits;
    return v;
  }
};

/** Byte-aligned shortcut for nbits == 8. */
struct PQDecoder8 {
  const uint8_t* code;
  explicit PQDecoder8(const uint8_t* code) : code(code) {}
  uint64_t decode() { return *code++; }
};

/** Two-byte shortcut for nbits == 16. */
struct PQDecoder16 {
  const uint8_t* code;
  explicit PQDecoder16(const uint8_t* code) : code(code) {}
  uint64_t decode() {
    const uint64_t v = static_cast<uint64_t>(code[0]) |
                       (static_cast<uint64_t>(code[1]) << 8);
    code += 2;
    return v;
  }
};

}  // namespace hypervec
