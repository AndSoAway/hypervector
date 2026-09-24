/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/algo/bm25/bm25.h>

#include <cstddef>

namespace hypervec {

float BM25Score(const SparseRow& query_idf, const SparseRow& doc_tf,
                const BM25Params& params, float doc_len) {
  const size_t n_doc = doc_tf.nnz();
  const size_t n_q = query_idf.nnz();
  const SparseElement* doc = doc_tf.data();
  const SparseElement* q = query_idf.data();

  float acc = 0.0f;
  size_t i = 0;
  size_t j = 0;
  while (i < n_doc && j < n_q) {
    if (doc[i].index < q[j].index) {
      ++i;
    } else if (doc[i].index > q[j].index) {
      ++j;
    } else {
      // query value = IDF, doc value = TF.
      acc += bm25_term_score(q[j].value, doc[i].value, doc_len, params);
      ++i;
      ++j;
    }
  }
  return acc;
}

}  // namespace hypervec
