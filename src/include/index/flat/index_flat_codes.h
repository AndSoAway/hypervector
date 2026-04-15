/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <core/distance_computer.h>
#include <core/index.h>
#include <core/maybe_owned_vector.h>

#include <vector>

namespace hypervec {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatCodes : Index {
  size_t code_size;

  /// encoded dataset, size n_total * code_size
  MaybeOwnedVector<uint8_t> codes;

  IndexFlatCodes();

  IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = kMetricL2);

  /// default Add uses SaEncode
  void Add(idx_t n, const float* x) override;

  void Reset() override;

  void ReconstructN(idx_t i0, idx_t ni, float* recons) const override;

  void Reconstruct(idx_t key, float* recons) const override;

  size_t SaCodeSize() const override;

  /** remove some ids. NB that because of the structure of the
   * index, the semantics of this operation are
   * different from the usual ones: the new ids are shifted */
  size_t RemoveIds(const IDSelector& sel) override;

  /** a FlatCodesDistanceComputer offers a distance_to_code method
   *
   * The default implementation explicitly decodes the vector with SaDecode.
   */
  virtual FlatCodesDistanceComputer* GetFlatCodesDistanceComputer() const;

  DistanceComputer* GetDistanceComputer() const override {
    return GetFlatCodesDistanceComputer();
  }

  /** Search implemented by decoding (most index types will have a faster
   * implementation) */
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void RangeSearch(idx_t n, const float* x, float radius,
                    RangeSearchResult* result,
                    const SearchParameters* params = nullptr) const override;

  virtual void Search1(const float* x, ResultHandler& handler,
                       SearchParameters* params = nullptr) const override;

  // returns a new instance of a CodePacker
  CodePacker* GetCodePacker() const;

  void CheckCompatibleForMerge(const Index& otherIndex) const override;

  virtual void MergeFrom(Index& otherIndex, idx_t add_id = 0) override;

  virtual void AddSaCodes(idx_t n, const uint8_t* x,
                            const idx_t* xids) override;

  // PermuteEntries. perm of size n_total maps new to old positions
  void PermuteEntries(const idx_t* perm);
};

}  // namespace hypervec
