/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/algo/bm25/text_to_sparse.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace hypervec {

SparseRow TextToSparse::AddDocument(const std::vector<std::string>& tokens) {
  if (tokens.empty()) return SparseRow{};

  std::unordered_map<uint32_t, float> tf_map;
  tf_map.reserve(tokens.size());

  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& tok : tokens) {
    tf_map[dict_.get_or_add(tok)] += 1.0f;
  }

  for (const auto& kv : tf_map) {
    df_[kv.first] += 1;
  }
  ++num_docs_;
  total_tokens_ += static_cast<uint64_t>(tokens.size());

  std::vector<uint32_t> indices;
  std::vector<float> values;
  indices.reserve(tf_map.size());
  values.reserve(tf_map.size());
  for (const auto& kv : tf_map) {
    indices.push_back(kv.first);
    values.push_back(kv.second);
  }
  return SparseRow(std::move(indices), std::move(values));
}

SparseRow TextToSparse::QueryToSparse(
    const std::vector<std::string>& tokens) const {
  if (tokens.empty()) return SparseRow{};

  std::lock_guard<std::mutex> lock(mu_);
  const double N = static_cast<double>(num_docs_);

  std::unordered_map<uint32_t, float> idf_map;
  idf_map.reserve(tokens.size());

  for (const auto& tok : tokens) {
    if (!dict_.contains(tok)) continue;  // unseen term — scores 0, skip
    const uint32_t id = dict_.id_of(tok);
    if (idf_map.count(id)) continue;  // deduplicate repeated query tokens

    double df_val = 0.0;
    const auto it = df_.find(id);
    if (it != df_.end()) df_val = static_cast<double>(it->second);

    idf_map[id] = static_cast<float>(
        std::log(1.0 + (N - df_val + 0.5) / (df_val + 0.5)));
  }

  std::vector<uint32_t> indices;
  std::vector<float> values;
  indices.reserve(idf_map.size());
  values.reserve(idf_map.size());
  for (const auto& kv : idf_map) {
    indices.push_back(kv.first);
    values.push_back(kv.second);
  }
  return SparseRow(std::move(indices), std::move(values));
}

float TextToSparse::Avgdl() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (num_docs_ == 0) return 0.0f;
  return static_cast<float>(total_tokens_) / static_cast<float>(num_docs_);
}

uint64_t TextToSparse::NumDocs() const {
  std::lock_guard<std::mutex> lock(mu_);
  return num_docs_;
}

}  // namespace hypervec
