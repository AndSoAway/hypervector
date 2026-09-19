/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#ifndef INDEX_FLAT_H
#define INDEX_FLAT_H

#include <index/flat/index_flat_codes.h>

#include <vector>

namespace hypervec {

/** Index that stores the full vectors and performs exhaustive Search */
struct IndexFlat : IndexFlatCodes {
  explicit IndexFlat(idx_t d,  ///< dimensionality of the input vectors
                     MetricType metric = kMetricL2);

  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void RangeSearch(idx_t n, const float* x, float radius,
                    RangeSearchResult* result,
                    const SearchParameters* params = nullptr) const override;

  void Reconstruct(idx_t key, float* recons) const override;

  /** compute distance with a subset of vectors
   *
   * @param x       query vectors, size n * d
   * @param labels  indices of the vectors that should be compared
   *                for each query vector, size n * k
   * @param distances
   *                corresponding output distances, size n * k
   */
  void ComputeDistanceSubset(idx_t n, const float* x, idx_t k,
                               float* distances, const idx_t* labels) const;

  // get pointer to the floating point data
  float* GetXb() {
    return (float*)codes.data();
  }
  const float* GetXb() const {
    return (const float*)codes.data();
  }

  IndexFlat() {}

  FlatCodesDistanceComputer* GetFlatCodesDistanceComputer() const override;

  /* The standalone codec interface (just memcopies in this case) */
  void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;

  void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;
};

struct IndexFlatIP : IndexFlat {
  explicit IndexFlatIP(idx_t d) : IndexFlat(d, kMetricInnerProduct) {}
  IndexFlatIP() {}
};

struct IndexFlatL2 : IndexFlat {
  // Special cache for L2 norms.
  // If this cache is set, then GetDistanceComputer() returns
  // a special version that computes the distance using dot products
  // and l2 norms.
  std::vector<float> cached_l2norms;

  /**
   * @param d dimensionality of the input vectors
   */
  explicit IndexFlatL2(idx_t d) : IndexFlat(d, kMetricL2) {}
  IndexFlatL2() {}

  // override for l2 norms cache.
  FlatCodesDistanceComputer* GetFlatCodesDistanceComputer() const override;

  // compute L2 norms
  void SyncL2Norms();
  // clear L2 norms
  void ClearL2Norms();
};

/// optimized version for 1D "vectors".
struct IndexFlat1D : IndexFlatL2 {
  bool continuous_update = true;  ///< is the permutation updated continuously?

  std::vector<idx_t> perm;  ///< sorted database indices

  explicit IndexFlat1D(bool continuous_update = true);

  /// if not continuous_update, call this between the last Add and
  /// the first Search
  void UpdatePermutation();

  void Add(idx_t n, const float* x) override;

  void Reset() override;

  /// Warn: the distances returned are L1 not L2
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
};

}  // namespace hypervec

#endif
