/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/pretransform/index_pre_transform.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {

IndexPreTransform::IndexPreTransform(
    std::unique_ptr<VectorTransform> vector_transform,
    std::unique_ptr<Index> index)
    : Index(), transform(std::move(vector_transform)), index(std::move(index)) {
  HYPERVEC_THROW_IF_NOT_MSG(this->transform != nullptr,
                            "IndexPreTransform: transform must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(this->index != nullptr,
                            "IndexPreTransform: index must not be null");
  HYPERVEC_THROW_IF_NOT_FMT(
      this->transform->d_in <= std::numeric_limits<int>::max(),
      "IndexPreTransform: input dimension exceeds %d",
      std::numeric_limits<int>::max());
  HYPERVEC_THROW_IF_NOT_FMT(
      this->transform->d_out == this->index->d,
      "IndexPreTransform: transform output dimension %" PRId64
      " does not match inner dimension %d",
      static_cast<int64_t>(this->transform->d_out), this->index->d);
  HYPERVEC_THROW_IF_NOT_MSG(
      this->transform->RequiresTraining() || this->transform->is_trained,
      "IndexPreTransform: fixed transform must already be configured");
  HYPERVEC_THROW_IF_NOT_MSG(
      this->index->n_total == 0 ||
          (this->transform->is_trained && this->index->is_trained),
      "IndexPreTransform: populated inner index requires a trained transform");

  d = static_cast<int>(this->transform->d_in);
  metric_type = this->index->metric_type;
  metric_arg = this->index->metric_arg;
  verbose = this->index->verbose;
  SyncState();
}

IndexCapabilities IndexPreTransform::GetCapabilities() const {
  IndexCapabilities capabilities = index->GetCapabilities();
  capabilities.requires_training =
      transform->RequiresTraining() || capabilities.requires_training;
  capabilities.supports_reconstruct =
      capabilities.supports_reconstruct && transform->IsReversible();
  capabilities.supports_merge = false;
  return capabilities;
}

void IndexPreTransform::ValidateReady(const char* operation) const {
  HYPERVEC_THROW_IF_NOT_FMT(is_trained, "%s: index is not trained", operation);
  HYPERVEC_THROW_IF_NOT_FMT(
      transform->is_trained && index->is_trained,
      "%s: wrapper and inner training state are inconsistent", operation);
  HYPERVEC_THROW_IF_NOT_FMT(
      n_total == index->n_total,
      "%s: wrapper and inner vector counts are inconsistent", operation);
}

void IndexPreTransform::ValidateTraining(idx_t n, const float* x) const {
  HYPERVEC_THROW_IF_NOT_MSG(n_total == 0,
                            "IndexPreTransform::Train: index must be empty");
  HYPERVEC_THROW_IF_NOT_MSG(
      !is_trained, "IndexPreTransform::Train: index is already trained");
  HYPERVEC_THROW_IF_NOT_MSG(n > 0,
                            "IndexPreTransform::Train: n must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr,
                            "IndexPreTransform::Train: x must not be null");
}

void IndexPreTransform::Train(idx_t n, const float* x) {
  ValidateTraining(n, x);
  if (transform->RequiresTraining()) {
    transform->Train(n, x);
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      transform->is_trained,
      "IndexPreTransform::Train: transform did not become trained");
  std::vector<float> transformed = transform->Apply(n, x);
  if (!index->is_trained) {
    index->Train(n, transformed.data());
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      index->is_trained,
      "IndexPreTransform::Train: inner index did not become trained");
  SyncState();
}

void IndexPreTransform::Train(idx_t n, const float* x, idx_t n_train_q,
                              const float* xq_train) {
  ValidateTraining(n, x);
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q >= 0,
      "IndexPreTransform::Train: query count must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q == 0 || xq_train != nullptr,
      "IndexPreTransform::Train: query data must not be null");
  if (transform->RequiresTraining()) {
    transform->Train(n, x);
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      transform->is_trained,
      "IndexPreTransform::Train: transform did not become trained");
  std::vector<float> transformed = transform->Apply(n, x);
  std::vector<float> transformed_queries;
  if (n_train_q > 0) {
    transformed_queries = transform->Apply(n_train_q, xq_train);
  }
  if (!index->is_trained) {
    index->Train(
        n, transformed.data(), n_train_q,
        transformed_queries.empty() ? nullptr : transformed_queries.data());
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      index->is_trained,
      "IndexPreTransform::Train: inner index did not become trained");
  SyncState();
}

void IndexPreTransform::Add(idx_t n, const float* x) {
  ValidateReady("IndexPreTransform::Add");
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexPreTransform::Add: n must be non-negative");
  if (n == 0) {
    return;
  }
  std::vector<float> transformed = transform->Apply(n, x);
  index->Add(n, transformed.data());
  SyncState();
}

void IndexPreTransform::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  ValidateReady("IndexPreTransform::AddWithIds");
  HYPERVEC_THROW_IF_NOT_MSG(
      n >= 0, "IndexPreTransform::AddWithIds: n must be non-negative");
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      xids != nullptr, "IndexPreTransform::AddWithIds: xids must not be null");
  std::vector<float> transformed = transform->Apply(n, x);
  index->AddWithIds(n, transformed.data(), xids);
  SyncState();
}

void IndexPreTransform::Search(idx_t n, const float* x, idx_t k,
                               float* distances, idx_t* labels,
                               const SearchParameters* params) const {
  ValidateReady("IndexPreTransform::Search");
  std::vector<float> transformed = transform->Apply(n, x);
  index->Search(n, transformed.data(), k, distances, labels, params);
}

void IndexPreTransform::Search1(const float* x, ResultHandler& handler,
                                SearchParameters* params) const {
  ValidateReady("IndexPreTransform::Search1");
  std::vector<float> transformed = transform->Apply(1, x);
  index->Search1(transformed.data(), handler, params);
}

void IndexPreTransform::RangeSearch(idx_t n, const float* x, float radius,
                                    RangeSearchResult* result,
                                    const SearchParameters* params) const {
  ValidateReady("IndexPreTransform::RangeSearch");
  std::vector<float> transformed = transform->Apply(n, x);
  index->RangeSearch(n, transformed.data(), radius, result, params);
}

void IndexPreTransform::SearchSubset(idx_t n, const float* x, idx_t k_base,
                                     const idx_t* base_labels, idx_t k,
                                     float* distances, idx_t* labels) const {
  ValidateReady("IndexPreTransform::SearchSubset");
  std::vector<float> transformed = transform->Apply(n, x);
  index->SearchSubset(n, transformed.data(), k_base, base_labels, k, distances,
                      labels);
}

void IndexPreTransform::Reset() {
  index->Reset();
  SyncState();
}

size_t IndexPreTransform::RemoveIds(const IDSelector& sel) {
  ValidateReady("IndexPreTransform::RemoveIds");
  const size_t removed = index->RemoveIds(sel);
  SyncState();
  return removed;
}

void IndexPreTransform::Reconstruct(idx_t key, float* recons) const {
  ValidateReady("IndexPreTransform::Reconstruct");
  HYPERVEC_THROW_IF_NOT_MSG(
      transform->IsReversible(),
      "IndexPreTransform::Reconstruct: transform is not reversible");
  std::vector<float> transformed(static_cast<size_t>(transform->d_out));
  index->Reconstruct(key, transformed.data());
  transform->ReverseTransform(1, transformed.data(), recons);
}

size_t IndexPreTransform::SaCodeSize() const {
  ValidateReady("IndexPreTransform::SaCodeSize");
  return index->SaCodeSize();
}

void IndexPreTransform::SaEncode(idx_t n, const float* x,
                                 uint8_t* bytes) const {
  ValidateReady("IndexPreTransform::SaEncode");
  std::vector<float> transformed = transform->Apply(n, x);
  index->SaEncode(n, transformed.data(), bytes);
}

void IndexPreTransform::SaDecode(idx_t n, const uint8_t* bytes,
                                 float* x) const {
  ValidateReady("IndexPreTransform::SaDecode");
  HYPERVEC_THROW_IF_NOT_MSG(
      n >= 0, "IndexPreTransform::SaDecode: n must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      transform->IsReversible(),
      "IndexPreTransform::SaDecode: transform is not reversible");
  const size_t count = mul_no_overflow(static_cast<size_t>(n),
                                       static_cast<size_t>(transform->d_out),
                                       "IndexPreTransform decoded vectors");
  std::vector<float> transformed(count);
  index->SaDecode(n, bytes, transformed.data());
  transform->ReverseTransform(n, transformed.data(), x);
}

void IndexPreTransform::AddSaCodes(idx_t n, const uint8_t* codes,
                                   const idx_t* xids) {
  ValidateReady("IndexPreTransform::AddSaCodes");
  index->AddSaCodes(n, codes, xids);
  SyncState();
}

void IndexPreTransform::SyncState() {
  n_total = index->n_total;
  is_trained = transform->is_trained && index->is_trained;
}

}  // namespace hypervec
