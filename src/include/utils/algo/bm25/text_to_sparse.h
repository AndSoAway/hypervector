/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */
#pragma once

#include <utils/structures/sparse_row.h>
#include <utils/structures/term_dictionary.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hypervec {

/** Converts pre-tokenized term lists to sparse TF/IDF vectors, tracks stats.
 *
 * Tokenization is the caller's responsibility (e.g. via CppjiebaAnalyzer).
 * This class only handles the numeric conversion and corpus statistics.
 *
 * Doc side  (insert path):  tokens → TF SparseRow + update DF / avgdl stats
 * Query side (search path):  tokens → IDF SparseRow  (read-only on stats)
 *
 * BM25 convention (aligned with Milvus/knowhere):
 *   doc   SparseRow values = raw term frequencies (TF)
 *   query SparseRow values = IDF weights  log(1 + (N−df+0.5)/(df+0.5))
 *
 * Thread safety: AddDocument is guarded by an internal mutex; QueryToSparse
 * is const and may be called concurrently once no AddDocument calls are
 * in flight.
 */
struct TextToSparse {
  TextToSparse() = default;

  /** Build a TF SparseRow from `tokens` and update corpus stats.
   *
   * `tokens` is the output of an Analyzer::Analyze() call (or any pre-split
   * term list).  Values = raw term frequency; indices = term ids assigned by
   * the internal TermDictionary (first-seen order).
   * Empty token list produces an empty SparseRow without updating stats.
   */
  SparseRow AddDocument(const std::vector<std::string>& tokens);

  /** Build an IDF SparseRow from `tokens`.
   *
   * Values = BM25 IDF weight: log(1 + (N−df(t)+0.5)/(df(t)+0.5))
   * where N = NumDocs(), df(t) = document frequency of term t.
   * Terms not seen during AddDocument are silently dropped (score 0).
   * Repeated tokens are deduplicated (each unique term appears once).
   * Empty token list produces an empty SparseRow.
   */
  SparseRow QueryToSparse(const std::vector<std::string>& tokens) const;

  /** Average document length (total tokens / num docs).  0 if no docs. */
  float Avgdl() const;

  /** Number of documents added so far. */
  uint64_t NumDocs() const;

  /** Read-only access to the term dictionary. */
  const TermDictionary& Dict() const { return dict_; }

 private:
  mutable std::mutex mu_;
  TermDictionary dict_;
  std::unordered_map<uint32_t, uint64_t> df_;  // term_id → document frequency
  uint64_t num_docs_{0};
  uint64_t total_tokens_{0};
};

}  // namespace hypervec
