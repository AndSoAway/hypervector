/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// HNSW-only index implementation

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/visited_table.h>
#include <omp.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_pq.h>
#include <utils/common/range_search_result.h>
#include <utils/common/result_handler.h>
#include <utils/log/assert.h>
#include <utils/structures/random.h>
#include <utils/structures/sorting.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/hnsw/hnsw_build_utils.h"

namespace hypervec {

using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;

/**************************************************************
 * Add / Search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
  if (IsSimilarityMetric(storage->metric_type)) {
    return new NegativeDistanceComputer(storage->GetDistanceComputer());
  } else {
    return storage->GetDistanceComputer();
  }
}

std::vector<idx_t> ValidatePermutation(const idx_t* perm, idx_t count) {
  HYPERVEC_THROW_IF_NOT_MSG(count >= 0,
                            "IndexHNSW::PermuteEntries: negative vector count");
  std::vector<idx_t> inverse(static_cast<size_t>(count));
  if (count == 0) {
    return inverse;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      perm != nullptr,
      "IndexHNSW::PermuteEntries: permutation must not be null");
  std::vector<uint8_t> seen(static_cast<size_t>(count), 0);
  for (idx_t new_id = 0; new_id < count; ++new_id) {
    const idx_t old_id = perm[new_id];
    HYPERVEC_THROW_IF_NOT_FMT(old_id >= 0 && old_id < count,
                              "IndexHNSW::PermuteEntries: entry %" PRId64
                              " is outside [0, %" PRId64 ")",
                              static_cast<int64_t>(old_id),
                              static_cast<int64_t>(count));
    HYPERVEC_THROW_IF_NOT_FMT(
        seen[static_cast<size_t>(old_id)] == 0,
        "IndexHNSW::PermuteEntries: duplicate old id %" PRId64,
        static_cast<int64_t>(old_id));
    seen[static_cast<size_t>(old_id)] = 1;
    inverse[static_cast<size_t>(old_id)] = new_id;
  }
  return inverse;
}

struct PreparedStoragePermutation {
  MaybeOwnedVector<uint8_t>* storage_codes = nullptr;
  MaybeOwnedVector<uint8_t> codes;
  IndexFlatL2* flat_l2 = nullptr;
  IndexFlat1D* flat_1d = nullptr;
  bool replace_flat_1d_permutation = false;
  std::vector<idx_t> flat_1d_permutation;
};

PreparedStoragePermutation PrepareStoragePermutation(
    Index* storage, const idx_t* perm, idx_t count,
    const std::vector<idx_t>& inverse) {
  HYPERVEC_THROW_IF_NOT_MSG(
      storage != nullptr,
      "IndexHNSW::PermuteEntries: storage must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      storage->n_total == count,
      "IndexHNSW::PermuteEntries: storage and graph counts differ");
  size_t code_size = 0;
  MaybeOwnedVector<uint8_t>* storage_codes = nullptr;
  if (auto* flat_codes = dynamic_cast<IndexFlatCodes*>(storage)) {
    code_size = flat_codes->code_size;
    storage_codes = &flat_codes->codes;
  } else if (auto* pq = dynamic_cast<IndexPQ*>(storage)) {
    code_size = pq->pq.code_size;
    storage_codes = &pq->codes;
  } else if (auto* lvq = dynamic_cast<IndexLVQ*>(storage)) {
    code_size = lvq->lvq.code_size;
    storage_codes = &lvq->codes;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      storage_codes != nullptr,
      "IndexHNSW::PermuteEntries: storage does not support code reordering");
  HYPERVEC_THROW_IF_NOT_MSG(
      code_size > 0 || count == 0,
      "IndexHNSW::PermuteEntries: storage code size is invalid");
  const size_t expected_bytes = mul_no_overflow(
      static_cast<size_t>(count), code_size, "IndexHNSW permutation storage");
  HYPERVEC_THROW_IF_NOT_MSG(
      storage_codes->size() == expected_bytes,
      "IndexHNSW::PermuteEntries: encoded storage size is inconsistent");

  PreparedStoragePermutation prepared;
  prepared.storage_codes = storage_codes;
  prepared.codes.resize(expected_bytes);
  for (idx_t new_id = 0; new_id < count; ++new_id) {
    std::memcpy(
        prepared.codes.data() + static_cast<size_t>(new_id) * code_size,
        storage_codes->data() + static_cast<size_t>(perm[new_id]) * code_size,
        code_size);
  }

  prepared.flat_l2 = dynamic_cast<IndexFlatL2*>(storage);
  prepared.flat_1d = dynamic_cast<IndexFlat1D*>(storage);
  if (prepared.flat_1d != nullptr && !prepared.flat_1d->perm.empty()) {
    HYPERVEC_THROW_IF_NOT_MSG(
        prepared.flat_1d->perm.size() == static_cast<size_t>(count),
        "IndexHNSW::PermuteEntries: IndexFlat1D permutation is inconsistent");
    prepared.replace_flat_1d_permutation = true;
    prepared.flat_1d_permutation.resize(static_cast<size_t>(count));
    for (idx_t rank = 0; rank < count; ++rank) {
      const idx_t old_id = prepared.flat_1d->perm[static_cast<size_t>(rank)];
      HYPERVEC_THROW_IF_NOT_MSG(
          old_id >= 0 && old_id < count,
          "IndexHNSW::PermuteEntries: IndexFlat1D contains an invalid id");
      prepared.flat_1d_permutation[static_cast<size_t>(rank)] =
          inverse[static_cast<size_t>(old_id)];
    }
  }
  return prepared;
}

void CommitStoragePermutation(PreparedStoragePermutation* prepared) {
  using std::swap;
  swap(*prepared->storage_codes, prepared->codes);
  if (prepared->flat_l2 != nullptr) {
    prepared->flat_l2->cached_l2norms.clear();
  }
  if (prepared->replace_flat_1d_permutation) {
    prepared->flat_1d->perm.swap(prepared->flat_1d_permutation);
  }
}

void ValidateHNSWState(const IndexHNSW& index, const char* operation) {
  HYPERVEC_THROW_IF_NOT_FMT(index.storage != nullptr,
                            "%s: storage must not be null", operation);
  HYPERVEC_THROW_IF_NOT_FMT(index.n_total >= 0,
                            "%s: vector count must not be negative", operation);
  HYPERVEC_THROW_IF_NOT_FMT(index.storage->n_total == index.n_total,
                            "%s: storage and index counts differ", operation);

  const size_t count = static_cast<size_t>(index.n_total);
  HYPERVEC_THROW_IF_NOT_FMT(
      index.hnsw.levels.size() == count &&
          index.hnsw.offsets.size() == count + 1 &&
          index.hnsw.offsets.front() == 0 &&
          index.hnsw.offsets.back() == index.hnsw.neighbors.size(),
      "%s: graph storage is inconsistent", operation);
  if (count == 0) {
    HYPERVEC_THROW_IF_NOT_FMT(
        index.hnsw.entry_point == -1 && index.hnsw.max_level == -1,
        "%s: empty graph has an entry point or level", operation);
    return;
  }

  HYPERVEC_THROW_IF_NOT_FMT(
      index.hnsw.entry_point >= 0 &&
          static_cast<size_t>(index.hnsw.entry_point) < count,
      "%s: entry point is outside the index", operation);
  HYPERVEC_THROW_IF_NOT_FMT(
      index.hnsw.max_level >= 0 &&
          index.hnsw.levels[static_cast<size_t>(index.hnsw.entry_point)] >
              index.hnsw.max_level,
      "%s: entry-point level is inconsistent", operation);
}

std::vector<NodeDistFarther> CollectLevel0Neighbors(const HNSW& graph,
                                                    idx_t node_count,
                                                    idx_t node,
                                                    DistanceComputer& distance,
                                                    const char* operation) {
  size_t begin = 0;
  size_t end = 0;
  graph.NeighborRange(node, 0, &begin, &end);
  std::vector<NodeDistFarther> neighbors;
  neighbors.reserve(end - begin);
  std::unordered_set<storage_idx_t> seen;
  bool reached_end = false;
  for (size_t offset = begin; offset < end; ++offset) {
    const storage_idx_t neighbor = graph.neighbors[offset];
    if (neighbor < 0) {
      HYPERVEC_THROW_IF_NOT_FMT(neighbor == -1,
                                "%s: node %" PRId64
                                " has an invalid neighbor sentinel",
                                operation, static_cast<int64_t>(node));
      reached_end = true;
      continue;
    }
    HYPERVEC_THROW_IF_NOT_FMT(!reached_end,
                              "%s: node %" PRId64
                              " has a neighbor after the end sentinel",
                              operation, static_cast<int64_t>(node));
    HYPERVEC_THROW_IF_NOT_FMT(static_cast<idx_t>(neighbor) < node_count,
                              "%s: node %" PRId64
                              " has an out-of-range neighbor %d",
                              operation, static_cast<int64_t>(node), neighbor);
    HYPERVEC_THROW_IF_NOT_FMT(static_cast<idx_t>(neighbor) != node,
                              "%s: node %" PRId64 " contains a self link",
                              operation, static_cast<int64_t>(node));
    HYPERVEC_THROW_IF_NOT_FMT(seen.insert(neighbor).second,
                              "%s: node %" PRId64
                              " contains duplicate neighbor %d",
                              operation, static_cast<int64_t>(node), neighbor);
    const float value = distance.symmetric_dis(node, neighbor);
    HYPERVEC_THROW_IF_NOT_FMT(!std::isnan(value),
                              "%s: node %" PRId64
                              " has a NaN distance to neighbor %d",
                              operation, static_cast<int64_t>(node), neighbor);
    neighbors.emplace_back(value, neighbor);
  }
  return neighbors;
}

void LinkLevel0Points(HNSW& graph, const Index& storage,
                      DistanceComputer& distance, idx_t node_count,
                      int dimension, int point_count,
                      const storage_idx_t* points,
                      const storage_idx_t* nearests, bool keep_max_size,
                      const char* operation) {
  OmpLockArray lock_array(static_cast<size_t>(node_count));
  std::vector<omp_lock_t>& locks = lock_array.Get();
  VisitedTable visited(static_cast<size_t>(node_count),
                       graph.use_visited_hashset);
  std::vector<float> query(static_cast<size_t>(dimension));
  for (int input = 0; input < point_count; ++input) {
    const storage_idx_t point = points[input];
    const storage_idx_t nearest = nearests[input];
    storage.Reconstruct(point, query.data());
    distance.SetQuery(query.data());
    const float nearest_distance = distance(nearest);
    HYPERVEC_THROW_IF_NOT_FMT(
        !std::isnan(nearest_distance),
        "%s: point %d has a NaN distance to entry point %d", operation, point,
        nearest);
    omp_set_lock(&locks[static_cast<size_t>(point)]);
    try {
      graph.AddLinksStartingFrom(distance, point, nearest, nearest_distance, 0,
                                 locks.data(), visited, keep_max_size);
    } catch (...) {
      omp_unset_lock(&locks[static_cast<size_t>(point)]);
      throw;
    }
    omp_unset_lock(&locks[static_cast<size_t>(point)]);
  }
}

void ValidateRuntimeEfSearch(const IndexHNSW& index,
                             const SearchParameters* params,
                             const char* operation) {
  if (index.n_total == 0) {
    return;
  }
  int ef_search = index.hnsw.ef_search;
  if (const auto* hnsw_params =
          dynamic_cast<const SearchParametersHNSW*>(params)) {
    ef_search = hnsw_params->ef_search;
  }
  HYPERVEC_THROW_IF_NOT_FMT(ef_search > 0, "%s: ef_search must be positive",
                            operation);
}

class NegatingResultHandler final : public ResultHandler {
 public:
  explicit NegatingResultHandler(ResultHandler& delegate)
      : delegate_(delegate) {
    threshold = -delegate_.threshold;
  }

  bool AddResult(float distance, idx_t id) override {
    const bool updated = delegate_.AddResult(-distance, id);
    threshold = -delegate_.threshold;
    return updated;
  }

 private:
  ResultHandler& delegate_;
};

void ValidateAppendState(const IndexHNSW& index, idx_t n, const float* x,
                         const Index* secondary_storage) {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexHNSW::Add: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      x != nullptr, "IndexHNSW::Add: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(index.n_total >= 0,
                            "IndexHNSW::Add: vector count is negative");
  const idx_t capacity =
      static_cast<idx_t>((std::numeric_limits<storage_idx_t>::max)());
  HYPERVEC_THROW_IF_NOT_MSG(
      index.n_total <= capacity && n <= capacity - index.n_total,
      "IndexHNSW::Add: vector count exceeds graph ID capacity");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.storage->n_total == index.n_total,
      "IndexHNSW::Add: storage and graph counts are inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      secondary_storage == nullptr || secondary_storage != index.storage,
      "IndexHNSW::Add: secondary storage aliases primary storage");
  HYPERVEC_THROW_IF_NOT_MSG(
      secondary_storage == nullptr ||
          secondary_storage->n_total == index.n_total,
      "IndexHNSW::Add: secondary storage count is inconsistent");

  const size_t count = static_cast<size_t>(index.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      index.hnsw.neighbors.is_owned,
      "IndexHNSW::Add: memory-mapped graph storage is read-only");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.hnsw.levels.size() == count &&
          index.hnsw.offsets.size() == count + 1 &&
          index.hnsw.offsets.front() == 0 &&
          index.hnsw.offsets.back() == index.hnsw.neighbors.size(),
      "IndexHNSW::Add: graph storage is inconsistent");
  if (count == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        index.hnsw.entry_point == -1 && index.hnsw.max_level == -1,
        "IndexHNSW::Add: empty graph has an entry point or level");
  } else {
    HYPERVEC_THROW_IF_NOT_MSG(
        index.hnsw.entry_point >= 0 &&
            static_cast<size_t>(index.hnsw.entry_point) < count &&
            index.hnsw.max_level >= 0 &&
            index.hnsw.levels[index.hnsw.entry_point] > index.hnsw.max_level,
        "IndexHNSW::Add: graph entry point is inconsistent");
  }
}

class CodeStorageAppendGuard {
 public:
  explicit CodeStorageAppendGuard(Index* storage)
      : storage_(storage), old_total_(storage->n_total) {
    size_t code_size = 0;
    if (auto* flat = dynamic_cast<IndexFlatCodes*>(storage)) {
      codes_ = &flat->codes;
      code_size = flat->code_size;
      flat_l2_ = dynamic_cast<IndexFlatL2*>(storage);
      flat_1d_ = dynamic_cast<IndexFlat1D*>(storage);
    } else if (auto* pq = dynamic_cast<IndexPQ*>(storage)) {
      codes_ = &pq->codes;
      code_size = pq->pq.code_size;
    } else if (auto* lvq = dynamic_cast<IndexLVQ*>(storage)) {
      codes_ = &lvq->codes;
      code_size = lvq->lvq.code_size;
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        codes_ != nullptr,
        "IndexHNSW::Add: storage does not support transactional append");
    HYPERVEC_THROW_IF_NOT_MSG(
        codes_->is_owned,
        "IndexHNSW::Add: memory-mapped vector storage is read-only");
    old_code_size_ = mul_no_overflow(static_cast<size_t>(old_total_), code_size,
                                     "IndexHNSW::Add existing storage size");
    HYPERVEC_THROW_IF_NOT_MSG(
        codes_->size() == old_code_size_,
        "IndexHNSW::Add: vector storage size is inconsistent");
    if (flat_l2_ != nullptr) {
      old_norms_ = flat_l2_->cached_l2norms;
    }
    if (flat_1d_ != nullptr) {
      old_permutation_ = flat_1d_->perm;
    }
  }

  CodeStorageAppendGuard(const CodeStorageAppendGuard&) = delete;
  CodeStorageAppendGuard& operator=(const CodeStorageAppendGuard&) = delete;

  ~CodeStorageAppendGuard() {
    if (committed_) {
      return;
    }
    codes_->resize(old_code_size_);
    storage_->n_total = old_total_;
    if (flat_l2_ != nullptr) {
      flat_l2_->cached_l2norms.swap(old_norms_);
    }
    if (flat_1d_ != nullptr) {
      flat_1d_->perm.swap(old_permutation_);
    }
  }

  void Commit() noexcept { committed_ = true; }

 private:
  Index* storage_;
  idx_t old_total_;
  MaybeOwnedVector<uint8_t>* codes_ = nullptr;
  size_t old_code_size_ = 0;
  IndexFlatL2* flat_l2_ = nullptr;
  IndexFlat1D* flat_1d_ = nullptr;
  std::vector<float> old_norms_;
  std::vector<idx_t> old_permutation_;
  bool committed_ = false;
};

class GraphAppendGuard {
 public:
  GraphAppendGuard(HNSW* graph, size_t old_count)
      : graph_(graph),
        old_count_(old_count),
        old_neighbor_size_(graph->neighbors.size()),
        old_entry_point_(graph->entry_point),
        old_max_level_(graph->max_level),
        old_ef_construction_(graph->ef_construction),
        old_rng_(graph->rng.mt) {}

  GraphAppendGuard(const GraphAppendGuard&) = delete;
  GraphAppendGuard& operator=(const GraphAppendGuard&) = delete;

  ~GraphAppendGuard() {
    if (committed_) {
      return;
    }
    for (const NodeSnapshot& snapshot : snapshots_) {
      std::copy(snapshot.neighbors.begin(), snapshot.neighbors.end(),
                graph_->neighbors.data() + snapshot.begin);
    }
    graph_->neighbors.resize(old_neighbor_size_);
    graph_->levels.resize(old_count_);
    graph_->offsets.resize(old_count_ + 1);
    graph_->entry_point = old_entry_point_;
    graph_->max_level = old_max_level_;
    graph_->ef_construction = old_ef_construction_;
    graph_->rng.mt = old_rng_;
  }

  void CaptureNode(storage_idx_t node) {
    if (node < 0 || static_cast<size_t>(node) >= old_count_) {
      return;
    }
    const size_t index = static_cast<size_t>(node);
    if (!captured_.insert(index).second) {
      return;
    }
    const size_t begin = graph_->offsets[index];
    const size_t end = graph_->offsets[index + 1];
    NodeSnapshot snapshot;
    snapshot.begin = begin;
    snapshot.neighbors.assign(graph_->neighbors.data() + begin,
                              graph_->neighbors.data() + end);
    snapshots_.push_back(std::move(snapshot));
  }

  void Commit() noexcept { committed_ = true; }

 private:
  struct NodeSnapshot {
    size_t begin = 0;
    std::vector<storage_idx_t> neighbors;
  };

  HNSW* graph_;
  size_t old_count_;
  size_t old_neighbor_size_;
  storage_idx_t old_entry_point_;
  int old_max_level_;
  int old_ef_construction_;
  std::mt19937 old_rng_;
  std::unordered_set<size_t> captured_;
  std::vector<NodeSnapshot> snapshots_;
  bool committed_ = false;
};

}  // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M, MetricType metric)
  : Index(d, metric), hnsw(M), storage(nullptr) {}

IndexHNSW::IndexHNSW(Index* storage, int M)
  : Index(storage->d, storage->metric_type), hnsw(M), storage(storage) {
  metric_arg = storage->metric_arg;
  is_trained = storage->is_trained;
}

IndexHNSW::~IndexHNSW() {
  if (storage && own_fields) {
    delete storage;
  }
}

IndexCapabilities IndexHNSW::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.supports_range_search = true;
  if (storage != nullptr) {
    const IndexCapabilities storage_capabilities = storage->GetCapabilities();
    capabilities.requires_training = storage_capabilities.requires_training;
    capabilities.supports_reconstruct =
        storage_capabilities.supports_reconstruct;
  }
  return capabilities;
}

void IndexHNSW::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(storage != nullptr,
                            "IndexHNSW::Train: storage is null");
  storage->Train(n, x);
  is_trained = storage->is_trained;
}

void IndexHNSW::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(storage != nullptr,
                            "IndexHNSW::Add: storage is null");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained,
                            "IndexHNSW::Add: call Train before Add");
  AddImpl(n, x, storage, nullptr);
}

void IndexHNSW::AddImpl(idx_t n, const float* x, Index* construction_storage,
                        Index* secondary_storage) {
  const idx_t n0 = n_total;
  ValidateAppendState(*this, n, x, secondary_storage);
  if (n == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      construction_storage == storage ||
          construction_storage == secondary_storage,
      "IndexHNSW::Add: construction storage is not managed by this index");
  const idx_t new_total = n0 + n;
  GraphAppendGuard graph_guard(&hnsw, static_cast<size_t>(n0));
  CodeStorageAppendGuard storage_guard(storage);
  std::optional<CodeStorageAppendGuard> secondary_guard;
  if (secondary_storage != nullptr) {
    secondary_guard.emplace(secondary_storage);
    secondary_storage->Add(n, x);
  }

  storage->Add(n, x);
  HYPERVEC_THROW_IF_NOT_MSG(
      storage->n_total == new_total &&
          (secondary_storage == nullptr ||
           secondary_storage->n_total == new_total),
      "IndexHNSW::Add: storage did not add the requested number of vectors");

  if (hnsw.ef_construction == 0) {
    hnsw.ef_construction = 40;
  }
  hnsw.PrepareLevelTab(static_cast<size_t>(n), false);
  std::unique_ptr<DistanceComputer> dis(
      storage_distance_computer(construction_storage));
  OmpLockArray lock_array(static_cast<size_t>(new_total) + 1);
  VisitedTable vt(new_total);
  const std::function<void(storage_idx_t)> before_node_mutation =
      [&graph_guard](storage_idx_t node) { graph_guard.CaptureNode(node); };
  for (idx_t i = n0; i < new_total; i++) {
    int pt_level = hnsw.levels[i] - 1;  // levels store level+1 (1-based)
    dis->SetQuery(x + (i - n0) * d);
    hnsw.AddWithLocks(*dis, pt_level, static_cast<int>(i), lock_array.Get(), vt,
                      false, before_node_mutation);
  }

  n_total = new_total;
  storage_guard.Commit();
  if (secondary_guard.has_value()) {
    secondary_guard->Commit();
  }
  graph_guard.Commit();
}

void IndexHNSW::Reset() {
  hnsw.Reset();
  storage->Reset();
  n_total = 0;
}

void IndexHNSW::ShrinkLevel0Neighbors(int size) {
  constexpr const char* operation = "IndexHNSW::ShrinkLevel0Neighbors";
  ValidateHNSWState(*this, operation);
  const int capacity = hnsw.NbNeighbors(0);
  HYPERVEC_THROW_IF_NOT_FMT(size > 0 && size <= capacity,
                            "%s: size must be in [1, %d], got %d", operation,
                            capacity, size);
  if (n_total == 0) {
    return;
  }

  std::vector<storage_idx_t> updated(
      hnsw.neighbors.data(), hnsw.neighbors.data() + hnsw.neighbors.size());
  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  for (idx_t node = 0; node < n_total; ++node) {
    size_t begin = 0;
    size_t end = 0;
    hnsw.NeighborRange(node, 0, &begin, &end);
    const std::vector<NodeDistFarther> level_neighbors =
        CollectLevel0Neighbors(hnsw, n_total, node, *dis, operation);
    std::priority_queue<NodeDistFarther> candidates;
    for (const NodeDistFarther& neighbor : level_neighbors) {
      candidates.push(neighbor);
    }

    std::vector<NodeDistFarther> selected;
    HNSW::ShrinkNeighborList(*dis, candidates, selected, size);
    for (size_t offset = begin; offset < end; ++offset) {
      const size_t selected_offset = offset - begin;
      updated[offset] = selected_offset < selected.size()
                            ? selected[selected_offset].id
                            : storage_idx_t{-1};
    }
  }

  hnsw.neighbors = MaybeOwnedVector<storage_idx_t>(std::move(updated));
}

void IndexHNSW::InitLevel0FromKnngraph(int k, const float* distances,
                                       const idx_t* labels) {
  constexpr const char* operation = "IndexHNSW::InitLevel0FromKnngraph";
  ValidateHNSWState(*this, operation);
  HYPERVEC_THROW_IF_NOT_FMT(k > 0, "%s: k must be positive, got %d", operation,
                            k);
  if (n_total == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_FMT(distances != nullptr,
                            "%s: distances must not be null", operation);
  HYPERVEC_THROW_IF_NOT_FMT(labels != nullptr, "%s: labels must not be null",
                            operation);
  (void)mul_no_overflow(static_cast<size_t>(n_total), static_cast<size_t>(k),
                        "IndexHNSW::InitLevel0FromKnngraph input size");

  const int capacity = hnsw.NbNeighbors(0);
  const bool similarity = IsSimilarityMetric(metric_type);
  std::vector<storage_idx_t> updated(
      hnsw.neighbors.data(), hnsw.neighbors.data() + hnsw.neighbors.size());
  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  for (idx_t node = 0; node < n_total; ++node) {
    const size_t row_offset =
        static_cast<size_t>(node) * static_cast<size_t>(k);
    std::priority_queue<NodeDistFarther> candidates;
    std::unordered_set<idx_t> seen;
    bool reached_end = false;
    for (int column = 0; column < k; ++column) {
      const size_t input_offset = row_offset + static_cast<size_t>(column);
      const idx_t neighbor = labels[input_offset];
      if (neighbor < 0) {
        HYPERVEC_THROW_IF_NOT_FMT(neighbor == -1,
                                  "%s: node %" PRId64
                                  " has an invalid neighbor sentinel",
                                  operation, static_cast<int64_t>(node));
        reached_end = true;
        continue;
      }
      HYPERVEC_THROW_IF_NOT_FMT(!reached_end,
                                "%s: node %" PRId64
                                " has a neighbor after the end sentinel",
                                operation, static_cast<int64_t>(node));
      HYPERVEC_THROW_IF_NOT_FMT(neighbor < n_total,
                                "%s: node %" PRId64
                                " has an out-of-range neighbor %" PRId64,
                                operation, static_cast<int64_t>(node),
                                static_cast<int64_t>(neighbor));
      if (neighbor == node) {
        continue;
      }
      HYPERVEC_THROW_IF_NOT_FMT(seen.insert(neighbor).second,
                                "%s: node %" PRId64
                                " contains duplicate neighbor %" PRId64,
                                operation, static_cast<int64_t>(node),
                                static_cast<int64_t>(neighbor));
      const float public_value = distances[input_offset];
      HYPERVEC_THROW_IF_NOT_FMT(!std::isnan(public_value),
                                "%s: node %" PRId64
                                " has a NaN value for neighbor %" PRId64,
                                operation, static_cast<int64_t>(node),
                                static_cast<int64_t>(neighbor));
      candidates.emplace(similarity ? -public_value : public_value,
                         static_cast<storage_idx_t>(neighbor));
    }

    std::vector<NodeDistFarther> selected;
    HNSW::ShrinkNeighborList(*dis, candidates, selected, capacity);
    size_t begin = 0;
    size_t end = 0;
    hnsw.NeighborRange(node, 0, &begin, &end);
    for (size_t offset = begin; offset < end; ++offset) {
      const size_t selected_offset = offset - begin;
      updated[offset] = selected_offset < selected.size()
                            ? selected[selected_offset].id
                            : storage_idx_t{-1};
    }
  }

  hnsw.neighbors = MaybeOwnedVector<storage_idx_t>(std::move(updated));
}

void IndexHNSW::InitLevel0FromEntryPoints(int npt, const storage_idx_t* points,
                                          const storage_idx_t* nearests) {
  constexpr const char* operation = "IndexHNSW::InitLevel0FromEntryPoints";
  ValidateHNSWState(*this, operation);
  HYPERVEC_THROW_IF_NOT_FMT(npt >= 0, "%s: npt must be non-negative, got %d",
                            operation, npt);
  if (npt == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_FMT(points != nullptr, "%s: points must not be null",
                            operation);
  HYPERVEC_THROW_IF_NOT_FMT(nearests != nullptr,
                            "%s: nearests must not be null", operation);
  HYPERVEC_THROW_IF_NOT_FMT(hnsw.ef_construction > 0,
                            "%s: ef_construction must be positive", operation);
  for (int input = 0; input < npt; ++input) {
    const storage_idx_t point = points[input];
    const storage_idx_t nearest = nearests[input];
    HYPERVEC_THROW_IF_NOT_FMT(point >= 0 && static_cast<idx_t>(point) < n_total,
                              "%s: point %d is outside the index", operation,
                              point);
    HYPERVEC_THROW_IF_NOT_FMT(
        nearest >= 0 && static_cast<idx_t>(nearest) < n_total,
        "%s: entry point %d is outside the index", operation, nearest);
    HYPERVEC_THROW_IF_NOT_FMT(
        point != nearest, "%s: point and entry point must differ", operation);
  }

  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  for (idx_t node = 0; node < n_total; ++node) {
    (void)CollectLevel0Neighbors(hnsw, n_total, node, *dis, operation);
  }

  HNSW prepared = hnsw;
  std::vector<storage_idx_t> copied_neighbors(
      hnsw.neighbors.data(), hnsw.neighbors.data() + hnsw.neighbors.size());
  prepared.neighbors =
      MaybeOwnedVector<storage_idx_t>(std::move(copied_neighbors));
  LinkLevel0Points(prepared, *storage, *dis, n_total, d, npt, points, nearests,
                   keep_max_size_level0, operation);

  for (idx_t node = 0; node < n_total; ++node) {
    (void)CollectLevel0Neighbors(prepared, n_total, node, *dis, operation);
  }
  hnsw.neighbors = std::move(prepared.neighbors);
}

void IndexHNSW::ReorderLinks() {
  constexpr const char* operation = "IndexHNSW::ReorderLinks";
  ValidateHNSWState(*this, operation);
  if (n_total == 0) {
    return;
  }

  std::vector<storage_idx_t> updated(
      hnsw.neighbors.data(), hnsw.neighbors.data() + hnsw.neighbors.size());
  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  for (idx_t node = 0; node < n_total; ++node) {
    size_t begin = 0;
    size_t end = 0;
    hnsw.NeighborRange(node, 0, &begin, &end);
    std::vector<NodeDistFarther> level_neighbors =
        CollectLevel0Neighbors(hnsw, n_total, node, *dis, operation);
    std::sort(level_neighbors.begin(), level_neighbors.end(),
              [](const NodeDistFarther& left, const NodeDistFarther& right) {
                return left.d < right.d ||
                       (left.d == right.d && left.id < right.id);
              });
    for (size_t offset = 0; offset < level_neighbors.size(); ++offset) {
      updated[begin + offset] = level_neighbors[offset].id;
    }
  }

  hnsw.neighbors = MaybeOwnedVector<storage_idx_t>(std::move(updated));
}

void IndexHNSW::LinkSingletons() {
  constexpr const char* operation = "IndexHNSW::LinkSingletons";
  ValidateHNSWState(*this, operation);
  if (n_total <= 1) {
    return;
  }

  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  std::vector<bool> has_incoming(static_cast<size_t>(n_total), false);
  for (idx_t node = 0; node < n_total; ++node) {
    const auto neighbors =
        CollectLevel0Neighbors(hnsw, n_total, node, *dis, operation);
    for (const NodeDistFarther& neighbor : neighbors) {
      has_incoming[static_cast<size_t>(neighbor.id)] = true;
    }
  }

  std::vector<storage_idx_t> singletons;
  for (idx_t node = 0; node < n_total; ++node) {
    if (!has_incoming[static_cast<size_t>(node)]) {
      singletons.push_back(static_cast<storage_idx_t>(node));
    }
  }
  if (singletons.empty()) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_FMT(hnsw.ef_construction > 0,
                            "%s: ef_construction must be positive", operation);

  storage_idx_t anchor = -1;
  if (has_incoming[static_cast<size_t>(hnsw.entry_point)]) {
    anchor = hnsw.entry_point;
  } else {
    for (idx_t node = 0; node < n_total; ++node) {
      if (has_incoming[static_cast<size_t>(node)]) {
        anchor = static_cast<storage_idx_t>(node);
        break;
      }
    }
  }
  if (anchor < 0) {
    anchor = hnsw.entry_point;
  }

  HNSW prepared = hnsw;
  std::vector<storage_idx_t> copied_neighbors(
      hnsw.neighbors.data(), hnsw.neighbors.data() + hnsw.neighbors.size());
  prepared.neighbors =
      MaybeOwnedVector<storage_idx_t>(std::move(copied_neighbors));
  for (storage_idx_t singleton : singletons) {
    size_t begin = 0;
    size_t end = 0;
    prepared.NeighborRange(singleton, 0, &begin, &end);
    std::fill(prepared.neighbors.data() + begin,
              prepared.neighbors.data() + end, storage_idx_t{-1});
  }

  std::vector<storage_idx_t> points;
  std::vector<storage_idx_t> nearests;
  points.reserve(singletons.size());
  nearests.reserve(singletons.size());
  for (storage_idx_t singleton : singletons) {
    if (singleton != anchor) {
      points.push_back(singleton);
      nearests.push_back(anchor);
    }
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      !points.empty(),
      "IndexHNSW::LinkSingletons: no distinct anchor is available");
  LinkLevel0Points(prepared, *storage, *dis, n_total, d,
                   static_cast<int>(points.size()), points.data(),
                   nearests.data(), keep_max_size_level0, operation);

  std::fill(has_incoming.begin(), has_incoming.end(), false);
  for (idx_t node = 0; node < n_total; ++node) {
    const auto neighbors =
        CollectLevel0Neighbors(prepared, n_total, node, *dis, operation);
    for (const NodeDistFarther& neighbor : neighbors) {
      has_incoming[static_cast<size_t>(neighbor.id)] = true;
    }
  }
  for (idx_t node = 0; node < n_total; ++node) {
    HYPERVEC_THROW_IF_NOT_FMT(has_incoming[static_cast<size_t>(node)],
                              "%s: repair left node %" PRId64
                              " without an incoming link",
                              operation, static_cast<int64_t>(node));
  }
  hnsw.neighbors = std::move(prepared.neighbors);
}

void IndexHNSW::Search(idx_t n, const float* x, idx_t k, float* distances,
                       idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexHNSW::Search: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexHNSW::Search: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      k <= (std::numeric_limits<int>::max)(),
      "IndexHNSW::Search: k exceeds the supported graph-search range");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexHNSW::Search: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexHNSW::Search: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexHNSW::Search: labels must not be null when n is positive");
  ValidateHNSWState(*this, "IndexHNSW::Search");
  if (n == 0) {
    return;
  }
  (void)mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                        "IndexHNSW::Search output size");
  ValidateRuntimeEfSearch(*this, params, "IndexHNSW::Search");

  // Use HNSW graph-based Search
  // Get distance computer from storage.
  // Must go through storage_distance_computer() so similarity metrics are
  // negated — HNSW graph traversal assumes "smaller is better".
  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

  // Do not mutate the process-global OpenMP thread count here. Concurrent
  // Python callers may run Search() after the SWIG layer releases the GIL, so
  // thread count should be controlled by the caller via the OpenMP runtime
  // environment instead of per query.

  // Create result handler
  using RH = HeapBlockResultHandler<HNSW::C>;
  RH bres(n, distances, labels, k);
  typename RH::SingleResultHandler res(bres);

  // Create visited table
  VisitedTable vt(n_total);
  const bool similarity = IsSimilarityMetric(metric_type);

  // Search each query
  for (idx_t i = 0; i < n; i++) {
    dis->SetQuery(x + i * d);
    res.begin(i);
    hnsw.Search(*dis, this, res, vt, params);
    res.end();
    if (similarity) {
      const size_t output_offset =
          static_cast<size_t>(i) * static_cast<size_t>(k);
      for (idx_t result = 0; result < k; ++result) {
        distances[output_offset + static_cast<size_t>(result)] =
            -distances[output_offset + static_cast<size_t>(result)];
      }
    }
  }
}

void IndexHNSW::SearchLevel0(idx_t n, const float* x, idx_t k,
                             const storage_idx_t* nearest,
                             const float* nearest_d, float* distances,
                             idx_t* labels, int nprobe, int search_type,
                             const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexHNSW::SearchLevel0: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexHNSW::SearchLevel0: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      k <= (std::numeric_limits<int>::max)(),
      "IndexHNSW::SearchLevel0: k exceeds the supported graph-search range");
  HYPERVEC_THROW_IF_NOT_MSG(nprobe > 0,
                            "IndexHNSW::SearchLevel0: nprobe must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      search_type == 1 || search_type == 2,
      "IndexHNSW::SearchLevel0: search_type must be 1 or 2");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexHNSW::SearchLevel0: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || nearest != nullptr,
      "IndexHNSW::SearchLevel0: nearest must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || nearest_d != nullptr,
      "IndexHNSW::SearchLevel0: nearest_d must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexHNSW::SearchLevel0: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexHNSW::SearchLevel0: labels must not be null when n is positive");
  ValidateHNSWState(*this, "IndexHNSW::SearchLevel0");
  if (n == 0) {
    return;
  }

  (void)mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                        "IndexHNSW::SearchLevel0 output size");
  (void)mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(nprobe),
                        "IndexHNSW::SearchLevel0 entry-point size");
  if (n_total > 0) {
    ValidateRuntimeEfSearch(*this, params, "IndexHNSW::SearchLevel0");
    for (idx_t query = 0; query < n; ++query) {
      const size_t offset =
          static_cast<size_t>(query) * static_cast<size_t>(nprobe);
      for (int probe = 0; probe < nprobe; ++probe) {
        const storage_idx_t entry = nearest[offset + probe];
        if (entry < 0) {
          break;
        }
        HYPERVEC_THROW_IF_NOT_MSG(
            static_cast<idx_t>(entry) < n_total,
            "IndexHNSW::SearchLevel0: entry point is outside the index");
      }
    }
  }

  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  using RH = HeapBlockResultHandler<HNSW::C>;
  RH block(n, distances, labels, k);
  typename RH::SingleResultHandler result(block);
  VisitedTable visited(n_total, use_visited_hashset);
  HNSWStats search_stats;
  const bool similarity = IsSimilarityMetric(metric_type);
  std::vector<float> internal_nearest_distances(
      similarity ? static_cast<size_t>(nprobe) : 0);

  for (idx_t query = 0; query < n; ++query) {
    result.begin(query);
    if (n_total > 0) {
      const size_t entry_offset =
          static_cast<size_t>(query) * static_cast<size_t>(nprobe);
      const float* query_nearest_distances = nearest_d + entry_offset;
      if (similarity) {
        for (int probe = 0; probe < nprobe; ++probe) {
          internal_nearest_distances[static_cast<size_t>(probe)] =
              -query_nearest_distances[probe];
        }
        query_nearest_distances = internal_nearest_distances.data();
      }
      dis->SetQuery(x + query * d);
      hnsw.SearchLevel0(*dis, result, nprobe, nearest + entry_offset,
                        query_nearest_distances, search_type, search_stats,
                        visited, params);
    }
    result.end();
    visited.advance();

    if (similarity) {
      const size_t output_offset =
          static_cast<size_t>(query) * static_cast<size_t>(k);
      for (idx_t output = 0; output < k; ++output) {
        distances[output_offset + static_cast<size_t>(output)] =
            -distances[output_offset + static_cast<size_t>(output)];
      }
    }
  }
}

void IndexHNSW::RangeSearch(idx_t n, const float* x, float radius,
                            RangeSearchResult* result,
                            const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexHNSW::RangeSearch: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexHNSW::RangeSearch: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || result != nullptr,
      "IndexHNSW::RangeSearch: result must not be null when n is positive");
  if (n > 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        result->nq == static_cast<size_t>(n),
        "IndexHNSW::RangeSearch: result query count does not match n");
    HYPERVEC_THROW_IF_NOT_MSG(
        result->lims != nullptr,
        "IndexHNSW::RangeSearch: result limits must not be null");
  }
  ValidateHNSWState(*this, "IndexHNSW::RangeSearch");
  if (n == 0) {
    return;
  }
  ValidateRuntimeEfSearch(*this, params, "IndexHNSW::RangeSearch");

  const bool similarity = IsSimilarityMetric(metric_type);
  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH block(result, similarity ? -radius : radius);
    typename RH::SingleResultHandler handler(block);
    VisitedTable visited(n_total);
    for (idx_t query = 0; query < n; ++query) {
      dis->SetQuery(x + query * d);
      handler.begin(query);
      hnsw.Search(*dis, this, handler, visited, params);
      handler.end();
    }
  }

  if (similarity) {
    for (size_t output = 0; output < result->lims[static_cast<size_t>(n)];
         ++output) {
      result->distances[output] = -result->distances[output];
    }
  }
}

void IndexHNSW::Search1(const float* x, ResultHandler& handler,
                        SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr,
                            "IndexHNSW::Search1: x must not be null");
  ValidateHNSWState(*this, "IndexHNSW::Search1");
  ValidateRuntimeEfSearch(*this, params, "IndexHNSW::Search1");

  std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
  dis->SetQuery(x);
  VisitedTable visited(n_total);
  if (IsSimilarityMetric(metric_type)) {
    NegatingResultHandler external_results(handler);
    hnsw.Search(*dis, this, external_results, visited, params);
  } else {
    hnsw.Search(*dis, this, handler, visited, params);
  }
}

void IndexHNSW::PermuteEntries(const idx_t* perm) {
  PermuteEntriesImpl(perm, nullptr);
}

void IndexHNSW::PermuteEntriesImpl(const idx_t* perm,
                                   Index* secondary_storage) {
  const std::vector<idx_t> inverse = ValidatePermutation(perm, n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.levels.size() == static_cast<size_t>(n_total),
      "IndexHNSW::PermuteEntries: graph and index counts differ");

  PreparedStoragePermutation primary =
      PrepareStoragePermutation(storage, perm, n_total, inverse);
  std::optional<PreparedStoragePermutation> secondary;
  if (secondary_storage != nullptr) {
    HYPERVEC_THROW_IF_NOT_MSG(
        secondary_storage != storage,
        "IndexHNSW::PermuteEntries: secondary storage aliases storage");
    secondary.emplace(
        PrepareStoragePermutation(secondary_storage, perm, n_total, inverse));
  }

  // HNSW validates and stages its replacement arrays before swapping them.
  // Once it succeeds, the prepared code-buffer swaps below cannot allocate.
  hnsw.PermuteEntries(perm);
  CommitStoragePermutation(&primary);
  if (secondary.has_value()) {
    CommitStoragePermutation(&secondary.value());
  }
}

void IndexHNSW::Reconstruct(idx_t key, float* recons) const {
  storage->Reconstruct(key, recons);
}

DistanceComputer* IndexHNSW::GetDistanceComputer() const {
  HYPERVEC_THROW_IF_NOT_MSG(
      storage != nullptr,
      "IndexHNSW::GetDistanceComputer: storage must not be null");
  return storage->GetDistanceComputer();
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
  is_trained = true;
}

static Index* make_hnsw_flat_storage(int d, MetricType metric) {
  if (metric == kMetricL2) {
    return new IndexFlatL2(d);
  }
  if (metric == kMetricInnerProduct) {
    return new IndexFlatIP(d);
  }
  return new IndexFlat(d, metric);
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
  : IndexHNSW(make_hnsw_flat_storage(d, metric), M) {
  own_fields = true;
  is_trained = true;
}

}  // namespace hypervec
