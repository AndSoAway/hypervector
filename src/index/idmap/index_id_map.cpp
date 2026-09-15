/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/hnsw/hnsw.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf.h>
#include <utils/common/range_search_result.h>
#include <utils/log/assert.h>

#include <typeinfo>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

class TranslatedSelector final : public IDSelector {
 public:
  TranslatedSelector(const IndexIDMap& id_map, const IDSelector& selector)
      : id_map_(id_map), selector_(selector) {}

  bool IsMember(idx_t internal_id) const override {
    const idx_t external_id = id_map_.from_internal(internal_id);
    return external_id >= 0 && selector_.IsMember(external_id);
  }

 private:
  const IndexIDMap& id_map_;
  const IDSelector& selector_;
};

template <typename Callback>
void WithTranslatedParameters(const IndexIDMap& id_map,
                              const SearchParameters* params,
                              Callback&& callback) {
  if (params == nullptr || params->sel == nullptr) {
    callback(params);
    return;
  }

  TranslatedSelector translated_selector(id_map, *params->sel);
  if (const auto* hnsw = dynamic_cast<const SearchParametersHNSW*>(params)) {
    SearchParametersHNSW translated = *hnsw;
    translated.sel = &translated_selector;
    callback(&translated);
    return;
  }
  if (const auto* ivf = dynamic_cast<const IVFSearchParameters*>(params)) {
    IVFSearchParameters translated = *ivf;
    translated.sel = &translated_selector;
    callback(&translated);
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      typeid(*params) == typeid(SearchParameters),
      "IndexIDMap cannot safely translate an unknown SearchParameters subtype");
  SearchParameters translated = *params;
  translated.sel = &translated_selector;
  callback(&translated);
}

}  // namespace

/*****************************************************
 * IndexIDMap implementation
 *******************************************************/

IndexIDMap::IndexIDMap() = default;

IndexIDMap::IndexIDMap(Index* index)
    : Index(index == nullptr ? 0 : index->d,
            index == nullptr ? kMetricL2 : index->metric_type),
      index(index) {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "IndexIDMap: underlying index is null");
  n_total = index->n_total;
  is_trained = index->is_trained;
  verbose = index->verbose;
  metric_arg = index->metric_arg;
  for (idx_t internal_id = 0; internal_id < n_total; ++internal_id) {
    id_map.emplace(internal_id, internal_id);
    rev_map.push_back(internal_id);
  }
}

IndexIDMap::IndexIDMap(std::unique_ptr<Index> index) : IndexIDMap(index.get()) {
  owned_index_ = std::move(index);
  this->index = owned_index_.get();
}

IndexIDMap::~IndexIDMap() = default;

idx_t IndexIDMap::to_internal(idx_t id) const {
  auto it = id_map.find(id);
  if (it == id_map.end()) {
    return -1;
  }
  return it->second;
}

idx_t IndexIDMap::from_internal(idx_t id) const {
  if (id >= 0 && id < static_cast<idx_t>(rev_map.size())) {
    return rev_map[static_cast<size_t>(id)];
  }
  for (const auto& entry : id_map) {
    if (entry.second == id) {
      return entry.first;
    }
  }
  return -1;
}

void IndexIDMap::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "IndexIDMap::Train: underlying index is null");
  try {
    index->Train(n, x);
  } catch (...) {
    is_trained = index->is_trained;
    throw;
  }
  is_trained = index->is_trained;
}

void IndexIDMap::Add(idx_t n, const float* x) { AddWithIds(n, x, nullptr); }

void IndexIDMap::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "IndexIDMap::AddWithIds: underlying index is null");
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexIDMap::AddWithIds: n must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      index->n_total == n_total,
      "IndexIDMap::AddWithIds: wrapper and underlying counts are inconsistent");
  if (n == 0) {
    return;
  }

  auto next_id_map = id_map;
  auto next_rev_map = rev_map;
  const idx_t internal_begin = index->n_total;
  if (maintain_rev_map) {
    HYPERVEC_THROW_IF_NOT_MSG(
        next_rev_map.size() == static_cast<size_t>(internal_begin),
        "IndexIDMap::AddWithIds: reverse map is inconsistent");
    next_rev_map.reserve(static_cast<size_t>(internal_begin + n));
  }

  for (idx_t i = 0; i < n; ++i) {
    const idx_t external_id = xids != nullptr ? xids[i] : n_total + i;
    HYPERVEC_THROW_IF_NOT_MSG(
        external_id >= 0,
        "IndexIDMap::AddWithIds: external IDs must be non-negative");
    const auto inserted =
        next_id_map.emplace(external_id, internal_begin + i).second;
    HYPERVEC_THROW_IF_NOT_MSG(
        inserted, "IndexIDMap::AddWithIds: external IDs must be unique");
    if (maintain_rev_map) {
      next_rev_map.push_back(external_id);
    }
  }

  index->Add(n, x);
  HYPERVEC_THROW_IF_NOT_MSG(
      index->n_total == internal_begin + n,
      "IndexIDMap::AddWithIds: underlying index added an unexpected count");

  id_map.swap(next_id_map);
  rev_map.swap(next_rev_map);
  n_total = index->n_total;
  is_trained = index->is_trained;
}

void IndexIDMap::Search(idx_t n, const float* x, idx_t k, float* distances,
                        idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "IndexIDMap::Search: underlying index is null");
  WithTranslatedParameters(
      *this, params, [&](const SearchParameters* translated_params) {
        index->Search(n, x, k, distances, labels, translated_params);
      });
  // Translate labels
  for (idx_t i = 0; i < n * k; i++) {
    if (labels[i] >= 0) {
      labels[i] = from_internal(labels[i]);
    }
  }
}

void IndexIDMap::RangeSearch(idx_t n, const float* x, float radius,
                             RangeSearchResult* result,
                             const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      index != nullptr, "IndexIDMap::RangeSearch: underlying index is null");
  WithTranslatedParameters(
      *this, params, [&](const SearchParameters* translated_params) {
        index->RangeSearch(n, x, radius, result, translated_params);
      });
  for (size_t i = 0; i < result->lims[result->nq]; ++i) {
    if (result->labels[i] >= 0) {
      result->labels[i] = from_internal(result->labels[i]);
    }
  }
}

void IndexIDMap::Reset() {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "IndexIDMap::Reset: underlying index is null");
  index->Reset();
  HYPERVEC_THROW_IF_NOT_MSG(
      index->n_total == 0,
      "IndexIDMap::Reset: underlying index did not clear its vectors");
  id_map.clear();
  rev_map.clear();
  n_total = index->n_total;
  is_trained = index->is_trained;
}

void IndexIDMap::Reconstruct(idx_t key, float* recons) const {
  idx_t internal_key = to_internal(key);
  if (internal_key < 0) {
    HYPERVEC_THROW_MSG("key not found");
  }
  index->Reconstruct(internal_key, recons);
}

void IndexIDMap::check_consistency() const {
  if (index == nullptr || index->n_total != n_total ||
      id_map.size() != static_cast<size_t>(n_total)) {
    HYPERVEC_THROW_MSG("inconsistency between index and ID map");
  }
  if (maintain_rev_map && id_map.size() != rev_map.size()) {
    HYPERVEC_THROW_MSG("inconsistency between id_map and rev_map");
  }
  std::vector<bool> seen_internal_ids(static_cast<size_t>(n_total), false);
  for (const auto& entry : id_map) {
    if (entry.first < 0 || entry.second < 0 || entry.second >= n_total ||
        seen_internal_ids[static_cast<size_t>(entry.second)] ||
        (maintain_rev_map &&
         rev_map[static_cast<size_t>(entry.second)] != entry.first)) {
      HYPERVEC_THROW_MSG("inconsistency in id_map / rev_map");
    }
    seen_internal_ids[static_cast<size_t>(entry.second)] = true;
  }
}

void IndexIDMap::MergeFrom(Index& otherIndex, idx_t add_id) {
  HYPERVEC_THROW_MSG("not implemented");
}

void IndexIDMap::construct_rev_map() {
  rev_map.assign(static_cast<size_t>(n_total), -1);
  for (const auto& entry : id_map) {
    HYPERVEC_THROW_IF_NOT_MSG(entry.second >= 0 && entry.second < n_total,
                              "invalid internal ID in id_map");
    rev_map[static_cast<size_t>(entry.second)] = entry.first;
  }
  for (const idx_t external_id : rev_map) {
    HYPERVEC_THROW_IF_NOT_MSG(external_id >= 0,
                              "id_map does not cover every internal ID");
  }
}

size_t IndexIDMap::RemoveIds(const IDSelector& sel) {
  HYPERVEC_THROW_MSG("not implemented");
  return 0;
}

bool IndexIDMap::OwnsIndex() const noexcept { return owned_index_ != nullptr; }

}  // namespace hypervec
