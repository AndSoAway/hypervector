/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */
#pragma once

#include <utils/structures/sparse_row.h>  // BM25Params, SparseRow

namespace hypervec {

/** BM25 contribution of a single matched term (pure function).
 *
 * Given a query term's IDF weight, the document's raw term frequency `tf`, and
 * the document length `doc_len`, returns that term's additive BM25 score:
 *
 *   idf * tf * (k1+1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
 *
 * `params.avgdl` is clamped to >= 1.0.  This is the single source of truth for
 * the BM25 term formula; BM25Score accumulates it over shared indices and the
 * inverted-index scorer calls it per posting.  Inline so all callers share one
 * definition with no per-call overhead.
 *
 * Aligned with knowhere sparse_utils.h::GetDocValueComputer / scorer.h.
 */
inline float bm25_term_score(float idf, float tf, float doc_len,
                             const BM25Params& params) {
  const float avgdl = params.avgdl > 1.0f ? params.avgdl : 1.0f;
  const float denom_norm =
      params.k1 * (1.0f - params.b + params.b * doc_len / avgdl);
  return idf * tf * (params.k1 + 1.0f) / (tf + denom_norm);
}

/** BM25 score of a whole document against a query (pure function).
 *
 * `query_idf` holds IDF weights (query side), `doc_tf` holds raw term
 * frequencies (document side); both are sparse vectors over term-id space,
 * sorted by index ascending.  `doc_len` is the document length (sum of its
 * term frequencies).  Returns
 *
 *   Σ_{t ∈ query∩doc}  idf(t) · bm25_term_score(...)
 *
 * computed via a two-pointer merge over the shared term ids.  Stateless and
 * independent of any Index type — callable from any retrieval path.
 *
 * Aligned with the Milvus/knowhere BM25 mode: query stores IDF, doc stores TF.
 */
float BM25Score(const SparseRow& query_idf, const SparseRow& doc_tf,
                const BM25Params& params, float doc_len);

}  // namespace hypervec
