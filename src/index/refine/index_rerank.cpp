/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/refine/index_rerank.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hypervec {

IndexRerank::IndexRerank(const Index& candidates, const float* exact_vectors,
                         idx_t exact_count, idx_t candidate_count)
    : Index(candidates.d, candidates.metric_type),
      candidates_(candidates),
      exact_vectors_(exact_vectors),
      candidate_count_(candidate_count) {
  HYPERVEC_THROW_IF_NOT_MSG(
      candidates.metric_type == kMetricL2 ||
          candidates.metric_type == kMetricInnerProduct,
      "IndexRerank: only L2 and inner product are supported");
  HYPERVEC_THROW_IF_NOT_MSG(
      candidates.d > 0 && exact_vectors != nullptr &&
          exact_count == candidates.n_total && candidate_count > 0,
      "IndexRerank: invalid exact base or candidate count");
  n_total = exact_count;
  is_trained = candidates.is_trained;
  metric_arg = candidates.metric_arg;
}

void IndexRerank::Search(idx_t n, const float* x, idx_t k, float* distances,
                         idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(n >= 0 && k > 0 && k <= candidate_count_,
                            "IndexRerank: require n >= 0 and 0 < k <= %" PRId64,
                            static_cast<int64_t>(candidate_count_));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || (x != nullptr && distances != nullptr && labels != nullptr),
      "IndexRerank: null search input or output");
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<size_t>(n) <= (std::numeric_limits<size_t>::max)() /
                                    static_cast<size_t>(candidate_count_),
      "IndexRerank: candidate buffer exceeds addressable memory");
  const size_t count = static_cast<size_t>(n) * candidate_count_;
  std::vector<float> candidate_distances(count);
  std::vector<idx_t> candidate_ids(count);
  candidates_.Search(n, x, candidate_count_, candidate_distances.data(),
                     candidate_ids.data(), params);
  const bool similarity = metric_type == kMetricInnerProduct;
  const float padding = similarity ? -(std::numeric_limits<float>::infinity)()
                                   : (std::numeric_limits<float>::infinity)();
  for (idx_t row = 0; row < n; ++row) {
    const float* query = x + static_cast<size_t>(row) * d;
    const size_t offset = static_cast<size_t>(row) * candidate_count_;
    const size_t output_offset = static_cast<size_t>(row) * k;
    std::vector<std::pair<float, idx_t>> scores;
    scores.reserve(candidate_count_);
    std::unordered_set<idx_t> seen;
    seen.reserve(candidate_count_);
    for (idx_t position = 0; position < candidate_count_; ++position) {
      const idx_t id = candidate_ids[offset + position];
      if (id < 0) {
        continue;
      }
      HYPERVEC_THROW_IF_NOT_MSG(id < n_total,
                                "IndexRerank: candidate ID exceeds base rows");
      if (!seen.insert(id).second) {
        continue;
      }
      const float* base = exact_vectors_ + static_cast<size_t>(id) * d;
      const float score = similarity ? fvec_inner_product(query, base, d)
                                     : fvec_L2sqr(query, base, d);
      scores.emplace_back(score, id);
    }
    const auto closer = [similarity](const auto& lhs, const auto& rhs) {
      return lhs.first == rhs.first
                 ? lhs.second < rhs.second
                 : (similarity ? lhs.first > rhs.first : lhs.first < rhs.first);
    };
    const size_t output_count = std::min(scores.size(), static_cast<size_t>(k));
    if (scores.size() > output_count) {
      std::nth_element(scores.begin(), scores.begin() + output_count,
                       scores.end(), closer);
    }
    std::sort(scores.begin(), scores.begin() + output_count, closer);
    for (size_t position = 0; position < static_cast<size_t>(k); ++position) {
      distances[output_offset + position] =
          position < output_count ? scores[position].first : padding;
      labels[output_offset + position] =
          position < output_count ? scores[position].second : idx_t{-1};
    }
  }
}

void IndexRerank::Add(idx_t, const float*) {
  HYPERVEC_THROW_MSG("IndexRerank: read-only view");
}

void IndexRerank::Reset() { HYPERVEC_THROW_MSG("IndexRerank: read-only view"); }

}  // namespace hypervec
