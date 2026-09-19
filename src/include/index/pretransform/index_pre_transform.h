/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <transform/vector_transform.h>

#include <cstdint>
#include <memory>

namespace hypervec {

/** Owning index wrapper that applies one transform before an inner index.
 *
 * External vectors have transform->d_in dimensions. The inner index receives
 * transform->d_out dimensions. Search labels and distances are forwarded
 * unchanged; reconstructed vectors are reverse-transformed to external space.
 */
struct IndexPreTransform : Index {
  std::unique_ptr<VectorTransform> transform;
  std::unique_ptr<Index> index;

  IndexPreTransform(std::unique_ptr<VectorTransform> vector_transform,
                    std::unique_ptr<Index> index);

  IndexCapabilities GetCapabilities() const override;

  void Train(idx_t n, const float* x) override;
  void Train(idx_t n, const float* x, idx_t n_train_q,
             const float* xq_train) override;

  void Add(idx_t n, const float* x) override;
  void AddWithIds(idx_t n, const float* x, const idx_t* xids) override;

  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Search1(const float* x, ResultHandler& handler,
               SearchParameters* params = nullptr) const override;
  void RangeSearch(idx_t n, const float* x, float radius,
                   RangeSearchResult* result,
                   const SearchParameters* params = nullptr) const override;
  void SearchSubset(idx_t n, const float* x, idx_t k_base,
                    const idx_t* base_labels, idx_t k, float* distances,
                    idx_t* labels) const override;

  void Reset() override;
  size_t RemoveIds(const IDSelector& sel) override;
  void Reconstruct(idx_t key, float* recons) const override;

  size_t SaCodeSize() const override;
  void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;
  void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;
  void AddSaCodes(idx_t n, const uint8_t* codes, const idx_t* xids) override;

 private:
  void ValidateReady(const char* operation) const;
  void ValidateTraining(idx_t n, const float* x) const;
  void SyncState();
};

}  // namespace hypervec
