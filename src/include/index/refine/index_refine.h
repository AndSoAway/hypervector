/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#ifndef HYPERVEC_INDEX_REFINE_INDEX_REFINE_H_
#define HYPERVEC_INDEX_REFINE_INDEX_REFINE_H_

#include <index/flat/index_flat.h>
#include <index/index.h>

namespace hypervec {

/** Per-query override of the over-fetch factor. */
struct RefineSearchParameters : SearchParameters {
  /// Number of candidates fetched from the base index is k * k_factor.
  float k_factor = 0.0f;  ///< <= 0 means "use the index default"
};

/** Re-ranking wrapper: a coarse/approximate base index (e.g. IVFPQ) supplies
 *  candidates, which are then re-scored with full-precision vectors held in a
 *  flat refine store and re-sorted to the true top-k.
 *
 *  Search over-fetches k * k_factor candidates from `base`, recomputes their
 *  exact distances against `refine` (an IndexFlat holding the raw vectors), and
 *  returns the best k. This recovers the recall that lossy quantizers (PQ, LVQ)
 *  give up, at the cost of storing full-precision vectors in `refine`.
 *
 *  Mirrors FAISS's IndexRefineFlat. Both children are owned iff own_fields.
 */
struct IndexRefineFlat : Index {
  Index* base = nullptr;        ///< coarse index supplying candidates
  IndexFlat* refine = nullptr;  ///< full-precision store for exact re-scoring
  bool own_fields = false;      ///< whether this index deletes base/refine
  float k_factor = 4.0f;        ///< default over-fetch multiplier

  IndexRefineFlat();

  /** Wrap `base`; builds an owned IndexFlatL2/IP refine store matching the
   *  base's dimension and metric. */
  explicit IndexRefineFlat(Index* base);

  /** Wrap `base` with a caller-supplied refine store. */
  IndexRefineFlat(Index* base, IndexFlat* refine);

  ~IndexRefineFlat() override;

  void Train(idx_t n, const float* x) override;
  void Add(idx_t n, const float* x) override;

  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void RangeSearch(idx_t n, const float* x, float radius,
                   RangeSearchResult* result,
                   const SearchParameters* params = nullptr) const override;

  void Reconstruct(idx_t key, float* recons) const override;

  void Reset() override;

  uint32_t fourcc() const override;
  void write_body(IOWriter* f) const override;
};

}  // namespace hypervec

#endif  // HYPERVEC_INDEX_REFINE_INDEX_REFINE_H_
