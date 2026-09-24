/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */
#pragma once

#include <utils/algo/sparse_vector/sparse_inverted_index.h>
#include <utils/structures/sparse_row.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hypervec {

/** BM25 inverted index — concrete implementation of SparseInvertedIndex.
 *
 * Stores documents as term→posting lists.  Each posting carries
 * (doc_id, tf, doc_len) as required by the T1 task spec.  Search uses a
 * Term-At-A-Time (TAAT) strategy: for each query term, walk its posting list
 * and accumulate each document's BM25 score via bm25_term_score (the single
 * BM25 formula defined in bm25.h — no duplication).
 *
 * No compression, no WAND/MaxScore pruning — sufficient for the course
 * verification criterion (correctness, not throughput).
 *
 * Data source consumes T5's sparse_vector representation (SparseRow): Add()
 * enumerates each document's term→weight entries to build the postings.
 *
 * Thread safety: read-only Search may run concurrently (under an external
 * read lock); Add is a write operation and must not overlap with Search.
 */
struct BM25InvertedIndex : SparseInvertedIndex {
  BM25InvertedIndex() = default;

  /// Add `n` TF SparseRows.  Doc ids are assigned sequentially from the
  /// current count (first Add call assigns ids 0..n-1, next call n..2n-1).
  void Add(idx_t n, const SparseRow* rows, idx_t dim) override;

  /// BM25 k-NN search (TAAT).  `params->bm25` supplies k1/b/avgdl.
  /// If params is null, defaults k1=1.2, b=0.75, avgdl=1.0.
  void Search(const SparseRow& query, idx_t k, float* distances,
              idx_t* labels,
              const SparseSearchParameters* params = nullptr) const override;

  /// Sum of TF values for document `row` (its BM25 document length).
  float RowSum(idx_t row) const override;

  /// Highest term id + 1 across all indexed documents.
  idx_t Dim() const override;

  /// Number of documents indexed.
  idx_t NumDocs() const { return static_cast<idx_t>(doc_lengths_.size()); }

  /// Document frequency of `term_id`: the number of documents containing it.
  /// Equals the length of the term's posting list, since each document
  /// contributes at most one posting per term (a TF SparseRow holds one entry
  /// per distinct term).  Returns 0 for an unseen term.
  idx_t DocFreq(uint32_t term_id) const;

  /// Average document length (mean of all document lengths).  0 if no docs.
  float Avgdl() const;

 private:
  /// One posting entry, carrying (doc_id, tf, doc_len) per the T1 spec.  The
  /// doc_len is denormalized into each posting so scoring needs no external
  /// lookup during the TAAT walk.
  struct Posting {
    idx_t doc_id;
    float tf;
    float doc_len;
  };

  // term_id → posting list.
  std::unordered_map<uint32_t, std::vector<Posting>> posting_;
  // doc_id → document length (sum of TF values); also exposed via RowSum.
  std::vector<float> doc_lengths_;
  // highest term id seen + 1
  idx_t dim_{0};
};

}  // namespace hypervec
