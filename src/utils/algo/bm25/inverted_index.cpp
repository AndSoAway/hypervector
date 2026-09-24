/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/algo/bm25/inverted_index.h>

#include <utils/algo/bm25/bm25.h>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

namespace hypervec {

void BM25InvertedIndex::Add(idx_t n, const SparseRow* rows, idx_t /*dim*/) {
  const idx_t base = static_cast<idx_t>(doc_lengths_.size());
  doc_lengths_.reserve(doc_lengths_.size() + static_cast<size_t>(n));

  for (idx_t i = 0; i < n; ++i) {
    const SparseRow& row = rows[i];
    const idx_t doc_id = base + i;
    const float doc_len = row.row_sum();

    for (size_t j = 0; j < row.nnz(); ++j) {
      const uint32_t term_id = row.index_at(j);
      const float tf = row.value_at(j);
      posting_[term_id].push_back({doc_id, tf, doc_len});

      const idx_t next_dim = static_cast<idx_t>(term_id) + 1;
      if (next_dim > dim_) dim_ = next_dim;
    }
    doc_lengths_.push_back(doc_len);
  }
}

void BM25InvertedIndex::Search(const SparseRow& query, idx_t k,
                                float* distances, idx_t* labels,
                                const SparseSearchParameters* params) const {
  static const SparseSearchParameters kDefaultParams{};
  const BM25Params& bm25 = params ? params->bm25 : kDefaultParams.bm25;

  // TAAT: for each query term, walk its posting list and accumulate the
  // per-term BM25 contribution into each document's running score.
  std::unordered_map<idx_t, float> scores;
  for (size_t qi = 0; qi < query.nnz(); ++qi) {
    const uint32_t term_id = query.index_at(qi);
    const float idf = query.value_at(qi);

    const auto it = posting_.find(term_id);
    if (it == posting_.end()) continue;

    for (const Posting& p : it->second) {
      scores[p.doc_id] += bm25_term_score(idf, p.tf, p.doc_len, bm25);
    }
  }

  // Top-k by descending score.
  std::vector<std::pair<float, idx_t>> scored;
  scored.reserve(scores.size());
  for (const auto& kv : scores) scored.push_back({kv.second, kv.first});

  const size_t result_count = std::min(static_cast<size_t>(k), scored.size());
  std::partial_sort(scored.begin(),
                    scored.begin() + static_cast<ptrdiff_t>(result_count),
                    scored.end(),
                    [](const auto& a, const auto& b) {
                      return a.first > b.first;
                    });

  for (idx_t i = 0; i < k; ++i) {
    if (static_cast<size_t>(i) < result_count) {
      distances[i] = scored[static_cast<size_t>(i)].first;
      labels[i]    = scored[static_cast<size_t>(i)].second;
    } else {
      distances[i] = std::numeric_limits<float>::infinity();
      labels[i]    = -1;
    }
  }
}

float BM25InvertedIndex::RowSum(idx_t row) const {
  return doc_lengths_.at(static_cast<size_t>(row));
}

idx_t BM25InvertedIndex::Dim() const {
  return dim_;
}

idx_t BM25InvertedIndex::DocFreq(uint32_t term_id) const {
  const auto it = posting_.find(term_id);
  if (it == posting_.end()) return 0;
  return static_cast<idx_t>(it->second.size());
}

float BM25InvertedIndex::Avgdl() const {
  if (doc_lengths_.empty()) return 0.0f;
  float total = 0.0f;
  for (float len : doc_lengths_) total += len;
  return total / static_cast<float>(doc_lengths_.size());
}

}  // namespace hypervec
