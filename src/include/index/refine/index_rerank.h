/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

namespace hypervec {

/** Read-only, opt-in exact reranking over an external row-major float matrix.
 *
 * Both the candidate index and exact vectors must outlive this non-owning
 * view; row i must correspond to index label i. No raw vectors are silently
 * embedded in the compressed index or its serialized format. The caller is
 * responsible for supplying the SAME original base on every reload.
 */
class IndexRerank final : public Index {
 public:
  IndexRerank(const Index& candidates, const float* exact_vectors,
              idx_t exact_count, idx_t candidate_count);

  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Add(idx_t n, const float* x) override;
  void Reset() override;

  idx_t CandidateCount() const noexcept { return candidate_count_; }

 private:
  const Index& candidates_;
  const float* exact_vectors_;
  idx_t candidate_count_;
};

}  // namespace hypervec
