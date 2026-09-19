/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/hnsw/hnsw.h>
#include <index/idmap/index_id_map.h>
#include <utils/common/range_search_result.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cstdint>
#include <limits>
#include <memory>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace hypervec {

namespace {

class IDSelectorTranslated final : public IDSelector {
 public:
  IDSelectorTranslated(const IndexIDMap& index, const IDSelector& external)
      : index_(index), external_(external) {}

  bool IsMember(idx_t internal_id) const final {
    const idx_t external_id = index_.from_internal(internal_id);
    return external_id >= 0 && external_.IsMember(external_id);
  }

 private:
  const IndexIDMap& index_;
  const IDSelector& external_;
};

std::unique_ptr<SearchParameters> TranslateSearchParameters(
    const SearchParameters& params, IDSelector* translated_selector) {
  if (typeid(params) == typeid(SearchParameters)) {
    auto translated = std::make_unique<SearchParameters>(params);
    translated->sel = translated_selector;
    return translated;
  }
  if (typeid(params) == typeid(SearchParametersHNSW)) {
    auto translated = std::make_unique<SearchParametersHNSW>(
        static_cast<const SearchParametersHNSW&>(params));
    translated->sel = translated_selector;
    return translated;
  }
  HYPERVEC_THROW_MSG(
      "IndexIDMap: selector translation does not support this search "
      "parameter subtype");
}

void TranslateLabels(const IndexIDMap& index, idx_t n, idx_t* labels) {
  for (idx_t i = 0; i < n; ++i) {
    if (labels[i] < 0) {
      continue;
    }
    const idx_t external_id = index.from_internal(labels[i]);
    HYPERVEC_THROW_IF_NOT_MSG(
        external_id >= 0,
        "IndexIDMap: underlying index returned an unmapped internal id");
    labels[i] = external_id;
  }
}

idx_t ShiftExternalId(idx_t external_id, idx_t add_id) {
  HYPERVEC_THROW_IF_NOT_FMT(external_id >= 0,
                            "IndexIDMap::MergeFrom: source external id must be "
                            "non-negative, got %" PRId64,
                            static_cast<int64_t>(external_id));
  const idx_t max_id = std::numeric_limits<idx_t>::max();
  const idx_t min_id = std::numeric_limits<idx_t>::min();
  HYPERVEC_THROW_IF_NOT_MSG(
      !((add_id > 0 && external_id > max_id - add_id) ||
        (add_id < 0 && external_id < min_id - add_id)),
      "IndexIDMap::MergeFrom: external id shift overflows");
  const idx_t shifted_id = external_id + add_id;
  HYPERVEC_THROW_IF_NOT_FMT(shifted_id >= 0,
                            "IndexIDMap::MergeFrom: shifted external id must "
                            "be non-negative, got %" PRId64,
                            static_cast<int64_t>(shifted_id));
  return shifted_id;
}

}  // namespace

/*****************************************************
 * IndexIDMap implementation
 *******************************************************/

IndexIDMap::IndexIDMap(Index* index) : Index() {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "IndexIDMap: index must not be null");
  this->index = index;
  this->d = index->d;
  this->metric_type = index->metric_type;
  this->n_total = index->n_total;
  this->is_trained = index->is_trained;
  this->verbose = index->verbose;
  this->metric_arg = index->metric_arg;

  rev_map.reserve(static_cast<size_t>(n_total));
  id_map.reserve(static_cast<size_t>(n_total));
  for (idx_t i = 0; i < n_total; ++i) {
    id_map.emplace(i, i);
    rev_map.push_back(i);
  }
}

IndexIDMap::~IndexIDMap() {
  if (own_fields) {
    delete index;
  }
}

idx_t IndexIDMap::to_internal(idx_t id) const {
  auto it = id_map.find(id);
  if (it == id_map.end()) {
    return -1;
  }
  return it->second;
}

idx_t IndexIDMap::from_internal(idx_t id) const {
  if (id < 0 || id >= (idx_t)rev_map.size()) {
    return -1;
  }
  return rev_map[id];
}

void IndexIDMap::Train(idx_t n, const float* x) {
  try {
    index->Train(n, x);
  } catch (...) {
    is_trained = index->is_trained;
    throw;
  }
  is_trained = index->is_trained;
}

void IndexIDMap::Train(idx_t n, const float* x, idx_t n_train_q,
                       const float* xq_train) {
  try {
    index->Train(n, x, n_train_q, xq_train);
  } catch (...) {
    is_trained = index->is_trained;
    throw;
  }
  is_trained = index->is_trained;
}

void IndexIDMap::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0, "IndexIDMap::Add: n must be non-negative");
  if (n == 0) {
    return;
  }
  std::vector<idx_t> ids(static_cast<size_t>(n));
  for (idx_t i = 0; i < n; ++i) {
    ids[static_cast<size_t>(i)] = n_total + i;
  }
  AddWithIds(n, x, ids.data());
}

void IndexIDMap::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexIDMap::AddWithIds: n must be non-negative");
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(xids != nullptr,
                            "IndexIDMap::AddWithIds: xids must not be null");
  check_consistency();

  const idx_t internal_base = index->n_total;
  auto next_id_map = id_map;
  auto next_rev_map = rev_map;
  next_id_map.reserve(next_id_map.size() + static_cast<size_t>(n));
  next_rev_map.reserve(next_rev_map.size() + static_cast<size_t>(n));
  for (idx_t i = 0; i < n; ++i) {
    const idx_t external_id = xids[static_cast<size_t>(i)];
    HYPERVEC_THROW_IF_NOT_FMT(external_id >= 0,
                              "IndexIDMap::AddWithIds: external id must be "
                              "non-negative, got %" PRId64,
                              static_cast<int64_t>(external_id));
    const bool inserted =
        next_id_map.emplace(external_id, internal_base + i).second;
    HYPERVEC_THROW_IF_NOT_FMT(
        inserted, "IndexIDMap::AddWithIds: duplicate external id %" PRId64,
        static_cast<int64_t>(external_id));
    next_rev_map.push_back(external_id);
  }

  index->Add(n, x);
  HYPERVEC_THROW_IF_NOT_FMT(
      index->n_total == internal_base + n,
      "IndexIDMap::AddWithIds: underlying index added an unexpected number "
      "of vectors (expected total %" PRId64 ", got %" PRId64 ")",
      static_cast<int64_t>(internal_base + n),
      static_cast<int64_t>(index->n_total));

  id_map.swap(next_id_map);
  rev_map.swap(next_rev_map);
  n_total = index->n_total;
}

void IndexIDMap::Search(idx_t n, const float* x, idx_t k, float* distances,
                        idx_t* labels, const SearchParameters* params) const {
  if (params != nullptr && params->sel != nullptr) {
    IDSelectorTranslated translated_selector(*this, *params->sel);
    auto translated_params =
        TranslateSearchParameters(*params, &translated_selector);
    index->Search(n, x, k, distances, labels, translated_params.get());
  } else {
    index->Search(n, x, k, distances, labels, params);
  }
  TranslateLabels(*this, n * k, labels);
}

void IndexIDMap::RangeSearch(idx_t n, const float* x, float radius,
                             RangeSearchResult* result,
                             const SearchParameters* params) const {
  if (params != nullptr && params->sel != nullptr) {
    IDSelectorTranslated translated_selector(*this, *params->sel);
    auto translated_params =
        TranslateSearchParameters(*params, &translated_selector);
    index->RangeSearch(n, x, radius, result, translated_params.get());
  } else {
    index->RangeSearch(n, x, radius, result, params);
  }
  TranslateLabels(*this, static_cast<idx_t>(result->lims[result->nq]),
                  result->labels);
}

void IndexIDMap::Reset() {
  id_map.clear();
  rev_map.clear();
  index->Reset();
  this->n_total = 0;
}

void IndexIDMap::Reconstruct(idx_t key, float* recons) const {
  idx_t internal_key = to_internal(key);
  if (internal_key < 0) {
    HYPERVEC_THROW_MSG("key not found");
  }
  index->Reconstruct(internal_key, recons);
}

void IndexIDMap::check_consistency() const {
  if (index == nullptr || n_total != index->n_total || d != index->d ||
      metric_type != index->metric_type || metric_arg != index->metric_arg ||
      is_trained != index->is_trained || id_map.size() != rev_map.size() ||
      rev_map.size() != static_cast<size_t>(n_total)) {
    HYPERVEC_THROW_MSG("inconsistency between IndexIDMap and its storage");
  }
  for (const auto& p : id_map) {
    if (p.first < 0 || p.second < 0 || p.second >= n_total ||
        rev_map[static_cast<size_t>(p.second)] != p.first) {
      HYPERVEC_THROW_MSG("inconsistency in id_map / rev_map");
    }
  }
}

void IndexIDMap::CheckCompatibleForMerge(const Index& otherIndex) const {
  const auto* other = dynamic_cast<const IndexIDMap*>(&otherIndex);
  HYPERVEC_THROW_IF_NOT_MSG(
      other != nullptr,
      "IndexIDMap::MergeFrom: can only merge another IndexIDMap");
  check_consistency();
  other->check_consistency();
  HYPERVEC_THROW_IF_NOT_MSG(
      index != other->index,
      "IndexIDMap::MergeFrom: indexes must not share the same storage");
  HYPERVEC_THROW_IF_NOT_MSG(
      d == other->d && metric_type == other->metric_type &&
          metric_arg == other->metric_arg,
      "IndexIDMap::MergeFrom: incompatible index parameters");
  index->CheckCompatibleForMerge(*other->index);
}

void IndexIDMap::MergeFrom(Index& otherIndex, idx_t add_id) {
  HYPERVEC_THROW_IF_NOT_MSG(&otherIndex != this,
                            "IndexIDMap::MergeFrom: cannot merge an index "
                            "into itself");
  CheckCompatibleForMerge(otherIndex);
  auto& other = static_cast<IndexIDMap&>(otherIndex);

  const idx_t old_total = n_total;
  const idx_t other_total = other.n_total;
  HYPERVEC_THROW_IF_NOT_MSG(
      other_total <= std::numeric_limits<idx_t>::max() - old_total,
      "IndexIDMap::MergeFrom: total vector count overflows");
  const idx_t merged_total = old_total + other_total;

  auto next_id_map = id_map;
  auto next_rev_map = rev_map;
  next_id_map.reserve(static_cast<size_t>(merged_total));
  next_rev_map.reserve(static_cast<size_t>(merged_total));
  for (idx_t i = 0; i < other_total; ++i) {
    const idx_t external_id =
        ShiftExternalId(other.rev_map[static_cast<size_t>(i)], add_id);
    const bool inserted =
        next_id_map.emplace(external_id, old_total + i).second;
    HYPERVEC_THROW_IF_NOT_FMT(
        inserted,
        "IndexIDMap::MergeFrom: duplicate shifted external id %" PRId64,
        static_cast<int64_t>(external_id));
    next_rev_map.push_back(external_id);
  }

  index->MergeFrom(*other.index, 0);
  HYPERVEC_THROW_IF_NOT_FMT(
      index->n_total == merged_total && other.index->n_total == 0,
      "IndexIDMap::MergeFrom: underlying indexes have unexpected totals "
      "after merge (destination %" PRId64 ", source %" PRId64 ")",
      static_cast<int64_t>(index->n_total),
      static_cast<int64_t>(other.index->n_total));

  id_map.swap(next_id_map);
  rev_map.swap(next_rev_map);
  n_total = merged_total;
  other.id_map.clear();
  other.rev_map.clear();
  other.n_total = 0;
}

void IndexIDMap::construct_rev_map() {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == index->n_total &&
          id_map.size() == static_cast<size_t>(n_total),
      "IndexIDMap::construct_rev_map: inconsistent vector counts");
  std::vector<idx_t> next_rev_map(static_cast<size_t>(n_total));
  std::vector<bool> seen(static_cast<size_t>(n_total), false);
  for (const auto& p : id_map) {
    HYPERVEC_THROW_IF_NOT_MSG(
        p.first >= 0 && p.second >= 0 && p.second < n_total,
        "IndexIDMap::construct_rev_map: invalid id mapping");
    const size_t internal_id = static_cast<size_t>(p.second);
    HYPERVEC_THROW_IF_NOT_MSG(
        !seen[internal_id],
        "IndexIDMap::construct_rev_map: duplicate internal id");
    seen[internal_id] = true;
    next_rev_map[internal_id] = p.first;
  }
  rev_map.swap(next_rev_map);
  maintain_rev_map = true;
  check_consistency();
}

size_t IndexIDMap::RemoveIds(const IDSelector& sel) {
  check_consistency();

  std::unordered_map<idx_t, idx_t> next_id_map;
  std::vector<idx_t> next_rev_map;
  next_id_map.reserve(id_map.size());
  next_rev_map.reserve(rev_map.size());
  for (idx_t internal_id = 0; internal_id < n_total; ++internal_id) {
    const idx_t external_id = rev_map[static_cast<size_t>(internal_id)];
    if (sel.IsMember(external_id)) {
      continue;
    }
    const idx_t next_internal_id = static_cast<idx_t>(next_rev_map.size());
    next_id_map.emplace(external_id, next_internal_id);
    next_rev_map.push_back(external_id);
  }
  const size_t expected_removed = rev_map.size() - next_rev_map.size();

  IDSelectorTranslated translated_selector(*this, sel);
  const size_t removed = index->RemoveIds(translated_selector);
  HYPERVEC_THROW_IF_NOT_FMT(
      removed == expected_removed &&
          index->n_total == static_cast<idx_t>(next_rev_map.size()),
      "IndexIDMap::RemoveIds: underlying index removed an unexpected number "
      "of vectors (expected %zu, got %zu)",
      expected_removed, removed);

  id_map.swap(next_id_map);
  rev_map.swap(next_rev_map);
  n_total = index->n_total;
  return removed;
}

}  // namespace hypervec
