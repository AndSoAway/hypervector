/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * HNSW-only index read implementation
 */

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/lsh/index_lsh.h>
#include <index/nsg/index_nsg.h>
#include <index/nsw/index_nsw.h>
#include <invlists/inverted_lists.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <persistence/mapped_io.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <utils/log/assert.h>
#include <utils/structures/maybe_owned_vector.h>

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
#include <utility>
#include <vector>

namespace hypervec {

namespace {
size_t deserialization_loop_limit_ = 0;
size_t deserialization_vector_byte_limit_ = uint64_t{1} << 40;  // 1 TB

template <typename T>
void ValidateElementCount(size_t count, const char* context) {
  const size_t bytes = mul_no_overflow(count, sizeof(T), context);
  HYPERVEC_THROW_IF_NOT_FMT(bytes < deserialization_vector_byte_limit_,
                            "%s payload is too large: %zu bytes (limit %zu)",
                            context, bytes, deserialization_vector_byte_limit_);
}

template <typename Vector>
void ReadVectorExact(Vector& values, size_t expected_size, IOReader* f,
                     const char* context) {
  size_t serialized_size;
  READANDCHECK(&serialized_size, 1);
  HYPERVEC_THROW_IF_NOT_FMT(
      serialized_size == expected_size,
      "%s has invalid element count: expected %zu, got %zu", context,
      expected_size, serialized_size);
  using value_type = typename Vector::value_type;
  ValidateElementCount<value_type>(serialized_size, context);
  values.resize(serialized_size);
  READANDCHECK(values.data(), serialized_size);
}

size_t ValidatePqMetadata(const ProductQuantizer& pq) {
  HYPERVEC_THROW_IF_NOT_MSG(pq.d > 0,
                            "ProductQuantizer deserialize: d must be > 0");
  HYPERVEC_THROW_IF_NOT_MSG(pq.M > 0 && pq.d % pq.M == 0,
                            "ProductQuantizer deserialize: M must divide d");
  HYPERVEC_THROW_IF_NOT_MSG(
      pq.nbits >= 1 && pq.nbits <= HYPERVEC_PQ_MAX_NBITS,
      "ProductQuantizer deserialize: nbits is out of range");
  const size_t ksub = size_t{1} << pq.nbits;
  const size_t subquantizer_size = mul_no_overflow(
      ksub, static_cast<size_t>(pq.d / pq.M), "ProductQuantizer centroids");
  const size_t centroid_count =
      mul_no_overflow(static_cast<size_t>(pq.M), subquantizer_size,
                      "ProductQuantizer centroids");
  ValidateElementCount<float>(centroid_count, "ProductQuantizer centroids");
  return centroid_count;
}

struct LvqPayloadSizes {
  size_t local_centroids;
  size_t residual_codebooks;
};

LvqPayloadSizes ValidateLvqMetadata(const LocalVectorQuantizer& lvq) {
  HYPERVEC_THROW_IF_NOT_MSG(lvq.d > 0,
                            "LocalVectorQuantizer deserialize: d must be > 0");
  HYPERVEC_THROW_IF_NOT_MSG(
      lvq.nlocal > 0, "LocalVectorQuantizer deserialize: nlocal must be > 0");
  HYPERVEC_THROW_IF_NOT_MSG(
      lvq.nbits >= 1 && lvq.nbits <= HYPERVEC_LVQ_MAX_NBITS,
      "LocalVectorQuantizer deserialize: nbits is out of range");
  const size_t local_centroids =
      mul_no_overflow(static_cast<size_t>(lvq.nlocal),
                      static_cast<size_t>(lvq.d), "LVQ local centroids");
  const size_t residual_per_local =
      mul_no_overflow(size_t{1} << lvq.nbits, static_cast<size_t>(lvq.d),
                      "LVQ residual codebooks");
  const size_t residual_codebooks =
      mul_no_overflow(static_cast<size_t>(lvq.nlocal), residual_per_local,
                      "LVQ residual codebooks");
  ValidateElementCount<float>(local_centroids, "LVQ local centroids");
  ValidateElementCount<float>(residual_codebooks, "LVQ residual codebooks");
  return {local_centroids, residual_codebooks};
}

size_t ValidateIvfMetadata(const IndexIVF& index) {
  HYPERVEC_THROW_IF_NOT_MSG(index.d > 0, "IndexIVF deserialize: d must be > 0");
  HYPERVEC_THROW_IF_NOT_MSG(index.nlist > 0,
                            "IndexIVF deserialize: nlist must be > 0");
  HYPERVEC_THROW_IF_NOT_MSG(index.nprobe > 0,
                            "IndexIVF deserialize: nprobe must be > 0");
  HYPERVEC_THROW_IF_NOT_FMT(
      deserialization_loop_limit_ == 0 ||
          static_cast<size_t>(index.nlist) <= deserialization_loop_limit_,
      "IndexIVF deserialize: nlist exceeds loop limit (%" PRId64 " > %zu)",
      static_cast<int64_t>(index.nlist), deserialization_loop_limit_);
  const size_t centroid_count =
      mul_no_overflow(static_cast<size_t>(index.nlist),
                      static_cast<size_t>(index.d), "IndexIVF centroids");
  ValidateElementCount<float>(centroid_count, "IndexIVF centroids");
  return centroid_count;
}

void ReadInvertedLists(IndexIVF& index, size_t code_size, IOReader* f) {
  HYPERVEC_THROW_IF_NOT_MSG(code_size > 0,
                            "IndexIVF deserialize: code size must be positive");
  const uint64_t expected_total_u64 = static_cast<uint64_t>(index.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      expected_total_u64 <= std::numeric_limits<size_t>::max(),
      "IndexIVF deserialize: n_total does not fit in size_t");
  const size_t expected_total = static_cast<size_t>(expected_total_u64);
  const size_t nlist = static_cast<size_t>(index.nlist);
  auto loaded = std::make_unique<ArrayInvertedLists>(nlist, code_size);

  size_t actual_total = 0;
  for (size_t list_no = 0; list_no < nlist; ++list_no) {
    size_t list_size;
    READ1(list_size);
    actual_total =
        add_no_overflow(actual_total, list_size, "IndexIVF entry count");
    HYPERVEC_THROW_IF_NOT_FMT(
        actual_total <= expected_total,
        "IndexIVF deserialize: list entries exceed n_total (%zu > %zu)",
        actual_total, expected_total);
    if (list_size == 0) {
      continue;
    }

    ValidateElementCount<idx_t>(list_size, "IndexIVF list ids");
    const size_t code_count =
        mul_no_overflow(list_size, code_size, "IndexIVF list codes");
    ValidateElementCount<uint8_t>(code_count, "IndexIVF list codes");
    std::vector<idx_t> ids(list_size);
    std::vector<uint8_t> codes(code_count);
    READANDCHECK(ids.data(), list_size);
    READANDCHECK(codes.data(), code_count);
    loaded->ids[list_no] = MaybeOwnedVector<idx_t>(std::move(ids));
    loaded->codes[list_no] = MaybeOwnedVector<uint8_t>(std::move(codes));
  }

  HYPERVEC_THROW_IF_NOT_FMT(
      actual_total == expected_total,
      "IndexIVF deserialize: list entries do not match n_total (%zu != %zu)",
      actual_total, expected_total);
  if (index.own_invlists) {
    delete index.invlists;
  }
  index.invlists = loaded.release();
  index.own_invlists = true;
}

void ValidateHnswGraph(const Index& index, const HNSW& hnsw,
                       int level0_capacity, int serialized_last_level) {
  HYPERVEC_THROW_IF_NOT_MSG(
      index.n_total <= std::numeric_limits<HNSW::storage_idx_t>::max(),
      "IndexHNSW deserialize: n_total exceeds graph ID capacity");
  const size_t total = static_cast<size_t>(index.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      level0_capacity > 0,
      "IndexHNSW deserialize: level-0 capacity must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.ef_construction > 0,
      "IndexHNSW deserialize: ef_construction must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.cum_nneighbor_per_level.size() >= 2 &&
          hnsw.cum_nneighbor_per_level.front() == 0 &&
          hnsw.cum_nneighbor_per_level[1] == level0_capacity,
      "IndexHNSW deserialize: invalid neighbor-capacity table");
  for (size_t i = 1; i < hnsw.cum_nneighbor_per_level.size(); ++i) {
    HYPERVEC_THROW_IF_NOT_MSG(
        hnsw.cum_nneighbor_per_level[i] > hnsw.cum_nneighbor_per_level[i - 1],
        "IndexHNSW deserialize: neighbor capacities must increase");
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.levels.size() == total && hnsw.offsets.size() == total + 1,
      "IndexHNSW deserialize: graph arrays do not match n_total");
  HYPERVEC_THROW_IF_NOT_MSG(
      !hnsw.offsets.empty() && hnsw.offsets.front() == 0,
      "IndexHNSW deserialize: offsets must start at zero");

  if (total == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        hnsw.max_level == -1 && hnsw.entry_point == -1 &&
            serialized_last_level == 0 && hnsw.offsets.back() == 0 &&
            hnsw.neighbors.size() == 0,
        "IndexHNSW deserialize: invalid empty graph metadata");
    return;
  }

  HYPERVEC_THROW_IF_NOT_MSG(hnsw.max_level >= 0 && hnsw.entry_point >= 0 &&
                                static_cast<size_t>(hnsw.entry_point) < total,
                            "IndexHNSW deserialize: invalid graph entry point");
  int observed_levels = 0;
  for (size_t node = 0; node < total; ++node) {
    const int node_levels = hnsw.levels[node];
    HYPERVEC_THROW_IF_NOT_MSG(
        node_levels > 0 && static_cast<size_t>(node_levels) <
                               hnsw.cum_nneighbor_per_level.size(),
        "IndexHNSW deserialize: invalid node level");
    observed_levels = std::max(observed_levels, node_levels);
    const size_t expected_span = static_cast<size_t>(
        hnsw.cum_nneighbor_per_level[static_cast<size_t>(node_levels)]);
    HYPERVEC_THROW_IF_NOT_MSG(
        hnsw.offsets[node + 1] >= hnsw.offsets[node] &&
            hnsw.offsets[node + 1] - hnsw.offsets[node] == expected_span,
        "IndexHNSW deserialize: node offsets do not match its levels");
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.max_level == observed_levels - 1 &&
          hnsw.levels[static_cast<size_t>(hnsw.entry_point)] ==
              observed_levels &&
          serialized_last_level == hnsw.levels.back(),
      "IndexHNSW deserialize: inconsistent maximum-level metadata");
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.offsets.back() == hnsw.neighbors.size(),
      "IndexHNSW deserialize: offsets do not match neighbor storage");

  for (size_t node = 0; node < total; ++node) {
    for (int level = 0; level < hnsw.levels[node]; ++level) {
      const size_t begin =
          hnsw.offsets[node] +
          static_cast<size_t>(
              hnsw.cum_nneighbor_per_level[static_cast<size_t>(level)]);
      const size_t end =
          hnsw.offsets[node] +
          static_cast<size_t>(
              hnsw.cum_nneighbor_per_level[static_cast<size_t>(level + 1)]);
      bool reached_padding = false;
      for (size_t position = begin; position < end; ++position) {
        const HNSW::storage_idx_t neighbor = hnsw.neighbors[position];
        if (neighbor == -1) {
          reached_padding = true;
          continue;
        }
        HYPERVEC_THROW_IF_NOT_MSG(
            !reached_padding && neighbor >= 0 &&
                static_cast<size_t>(neighbor) < total &&
                static_cast<size_t>(neighbor) != node &&
                hnsw.levels[static_cast<size_t>(neighbor)] > level,
            "IndexHNSW deserialize: invalid neighbor entry");
      }
    }
  }
}

void RebuildHnswLevelProbabilities(HNSW& hnsw) {
  int base_capacity = hnsw.cum_nneighbor_per_level[1] / 2;
  if (hnsw.cum_nneighbor_per_level.size() >= 3) {
    base_capacity =
        hnsw.cum_nneighbor_per_level[2] - hnsw.cum_nneighbor_per_level[1];
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      base_capacity > 1,
      "IndexHNSW deserialize: invalid upper-level neighbor capacity");
  auto neighbor_capacities = std::move(hnsw.cum_nneighbor_per_level);
  hnsw.assign_probas.clear();
  hnsw.cum_nneighbor_per_level.clear();
  hnsw.SetDefaultProbas(base_capacity,
                        static_cast<float>(1.0 / std::log(base_capacity)));
  hnsw.cum_nneighbor_per_level = std::move(neighbor_capacities);
}

void ValidateHnswStorage(const IndexHNSW& index) {
  HYPERVEC_THROW_IF_NOT_MSG(index.storage != nullptr,
                            "IndexHNSW deserialize: storage is missing");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.storage->d == index.d && index.storage->n_total == index.n_total &&
          index.storage->metric_type == index.metric_type &&
          index.storage->metric_arg == index.metric_arg &&
          index.storage->is_trained == index.is_trained,
      "IndexHNSW deserialize: storage metadata does not match the graph");
}

void RestoreIdMap(IndexIDMap& index) {
  HYPERVEC_THROW_IF_NOT_MSG(index.index != nullptr,
                            "IndexIDMap deserialize: storage is missing");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.index->d == index.d && index.index->n_total == index.n_total &&
          index.index->metric_type == index.metric_type &&
          index.index->metric_arg == index.metric_arg &&
          index.index->is_trained == index.is_trained,
      "IndexIDMap deserialize: storage metadata does not match the wrapper");

  index.id_map.reserve(index.rev_map.size());
  for (size_t internal_id = 0; internal_id < index.rev_map.size();
       ++internal_id) {
    const idx_t external_id = index.rev_map[internal_id];
    HYPERVEC_THROW_IF_NOT_MSG(
        external_id >= 0,
        "IndexIDMap deserialize: external IDs must be non-negative");
    const bool inserted =
        index.id_map.emplace(external_id, static_cast<idx_t>(internal_id))
            .second;
    HYPERVEC_THROW_IF_NOT_MSG(
        inserted, "IndexIDMap deserialize: external IDs must be unique");
  }
  index.maintain_rev_map = true;
  index.check_consistency();
}
}  // namespace

size_t get_deserialization_loop_limit() { return deserialization_loop_limit_; }

void set_deserialization_loop_limit(size_t value) {
  deserialization_loop_limit_ = value;
}

size_t get_deserialization_vector_byte_limit() {
  return deserialization_vector_byte_limit_;
}

void set_deserialization_vector_byte_limit(size_t value) {
  deserialization_vector_byte_limit_ = value;
}

struct IndexHeaderData {
  int d = 0;
  idx_t n_total = 0;
  bool is_trained = false;
  MetricType metric_type = kMetricL2;
  float metric_arg = 0.0F;
};

static IndexHeaderData read_index_header_data(IOReader* f) {
  IndexHeaderData header;
  READ1(header.d);
  READ1(header.n_total);
  HYPERVEC_CHECK_RANGE(header.d, 0, (1 << 20) + 1);
  HYPERVEC_THROW_IF_NOT_FMT(header.n_total >= 0,
                            "invalid n_total %" PRId64 " read from index",
                            (int64_t)header.n_total);
  idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  uint8_t is_trained_raw;
  READ1(is_trained_raw);
  HYPERVEC_THROW_IF_NOT_MSG(
      is_trained_raw <= 1,
      "index deserialize: is_trained must be encoded as 0 or 1");
  header.is_trained = is_trained_raw != 0;
  int metric_type_int;
  READ1(metric_type_int);
  header.metric_type = MetricTypeFromInt(metric_type_int);
  if (header.metric_type > 1) {
    READ1(header.metric_arg);
  }
  return header;
}

static void read_index_header(Index& idx, IOReader* f) {
  const IndexHeaderData header = read_index_header_data(f);
  idx.d = header.d;
  idx.n_total = header.n_total;
  idx.is_trained = header.is_trained;
  idx.metric_type = header.metric_type;
  idx.metric_arg = header.metric_arg;
  idx.verbose = false;
}

static void read_pq(ProductQuantizer& pq, IOReader* f) {
  READ1(pq.d);
  READ1(pq.M);
  READ1(pq.nbits);
  const size_t centroid_count = ValidatePqMetadata(pq);
  pq.SetDerivedValues();
  ReadVectorExact(pq.centroids, centroid_count, f,
                  "ProductQuantizer centroids");
  pq.is_trained = true;
}

static void read_lvq(LocalVectorQuantizer& lvq, IOReader* f) {
  READ1(lvq.d);
  READ1(lvq.nlocal);
  READ1(lvq.nbits);
  const LvqPayloadSizes sizes = ValidateLvqMetadata(lvq);
  lvq.SetDerivedValues();
  ReadVectorExact(lvq.local_centroids, sizes.local_centroids, f,
                  "LVQ local centroids");
  ReadVectorExact(lvq.residual_codebooks, sizes.residual_codebooks, f,
                  "LVQ residual codebooks");
  lvq.BuildDecodedCodebooks();
  lvq.is_trained = true;
}

static void read_HNSW(HNSW& hnsw, const Index& index, IOReader* f) {
  HYPERVEC_THROW_IF_NOT_MSG(
      index.n_total <= std::numeric_limits<HNSW::storage_idx_t>::max(),
      "IndexHNSW deserialize: n_total exceeds graph ID capacity");
  int level0_capacity;
  READ1(level0_capacity);
  READ1(hnsw.ef_construction);
  READ1(hnsw.max_level);
  READ1(hnsw.entry_point);
  int serialized_last_level;
  READ1(serialized_last_level);
  READVECTOR(hnsw.cum_nneighbor_per_level);
  ReadVectorExact(hnsw.levels, static_cast<size_t>(index.n_total), f,
                  "IndexHNSW levels");
  READVECTOR(hnsw.neighbors);
  const size_t offset_count = add_no_overflow(
      static_cast<size_t>(index.n_total), 1, "IndexHNSW offsets");
  ReadVectorExact(hnsw.offsets, offset_count, f, "IndexHNSW offsets");
  ValidateHnswGraph(index, hnsw, level0_capacity, serialized_last_level);
  RebuildHnswLevelProbabilities(hnsw);
}

static bool read_graph_bool(IOReader* f, const char* context) {
  uint8_t value;
  READ1(value);
  HYPERVEC_THROW_IF_NOT_FMT(value <= 1, "%s must be encoded as 0 or 1",
                            context);
  return value != 0;
}

static std::unique_ptr<IndexNSWFlat> read_nsw_flat(
    const IndexHeaderData& header, IOReader* f) {
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  const uint64_t total_u64 = static_cast<uint64_t>(header.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      total_u64 <= std::numeric_limits<size_t>::max(),
      "IndexNSWFlat deserialize: n_total does not fit in size_t");
  const size_t total = static_cast<size_t>(total_u64);
  HYPERVEC_THROW_IF_NOT_MSG(
      total <= kGraphCapacity,
      "IndexNSWFlat deserialize: n_total exceeds graph ID capacity");
  HYPERVEC_THROW_IF_NOT_FMT(
      deserialization_loop_limit_ == 0 || total <= deserialization_loop_limit_,
      "IndexNSWFlat deserialize: n_total exceeds loop limit (%zu > %zu)", total,
      deserialization_loop_limit_);

  NSWIndexOptions options;
  READ1(options.max_degree);
  READ1(options.ef_construction);
  READ1(options.ef_search);
  options.check_relative_distance =
      read_graph_bool(f, "IndexNSWFlat check_relative_distance");
  options.fill_to_max_degree =
      read_graph_bool(f, "IndexNSWFlat fill_to_max_degree");
  GraphId entry_point;
  READ1(entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      (total == 0 && entry_point == kInvalidGraphId) ||
          (total > 0 && entry_point >= 0 &&
           static_cast<size_t>(entry_point) < total),
      "IndexNSWFlat deserialize: entry point is inconsistent with n_total");

  auto index = std::make_unique<IndexNSWFlat>(header.d, header.metric_type,
                                              options, header.metric_arg);
  HYPERVEC_THROW_IF_NOT_MSG(
      header.is_trained == index->is_trained,
      "IndexNSWFlat deserialize: training state does not match flat storage");

  const size_t code_count = mul_no_overflow(
      total, index->CodeStore().CodeSize(), "IndexNSWFlat codes");
  std::vector<uint8_t> codes;
  ReadVectorExact(codes, code_count, f, "IndexNSWFlat codes");

  const size_t offset_count = add_no_overflow(total, 1, "IndexNSWFlat offsets");
  std::vector<size_t> offsets;
  ReadVectorExact(offsets, offset_count, f, "IndexNSWFlat offsets");
  HYPERVEC_THROW_IF_NOT_MSG(offsets.front() == 0,
                            "IndexNSWFlat offsets must start at zero");
  for (size_t node = 0; node < total; ++node) {
    HYPERVEC_THROW_IF_NOT_MSG(
        offsets[node] <= offsets[node + 1] &&
            offsets[node + 1] - offsets[node] <= options.max_degree,
        "IndexNSWFlat offsets contain an invalid neighbor span");
  }
  const size_t max_edges = mul_no_overflow(total, options.max_degree,
                                           "IndexNSWFlat maximum edge count");
  HYPERVEC_THROW_IF_NOT_MSG(
      offsets.back() <= max_edges,
      "IndexNSWFlat edge count exceeds the configured degree bound");
  std::vector<GraphId> edges;
  ReadVectorExact(edges, offsets.back(), f, "IndexNSWFlat edges");

  MutableBoundedGraph graph(total, options.max_degree);
  for (size_t node = 0; node < total; ++node) {
    const size_t degree = offsets[node + 1] - offsets[node];
    const GraphId* neighbors =
        degree == 0 ? nullptr : edges.data() + offsets[node];
    graph.SetNeighbors(static_cast<GraphId>(node),
                       GraphNeighborView(neighbors, degree));
  }
  InMemoryCodeStore code_store(index->CodeStore().CodeSize());
  code_store.Append(header.n_total, codes.data());
  index->RestoreState(std::move(code_store), std::move(graph), entry_point);
  return index;
}

static std::unique_ptr<IndexNSGFlat> read_nsg_flat(
    const IndexHeaderData& header, IOReader* f) {
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  const uint64_t total_u64 = static_cast<uint64_t>(header.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      total_u64 <= std::numeric_limits<size_t>::max(),
      "IndexNSGFlat deserialize: n_total does not fit in size_t");
  const size_t total = static_cast<size_t>(total_u64);
  HYPERVEC_THROW_IF_NOT_MSG(
      total <= kGraphCapacity,
      "IndexNSGFlat deserialize: n_total exceeds graph ID capacity");
  HYPERVEC_THROW_IF_NOT_FMT(
      deserialization_loop_limit_ == 0 || total <= deserialization_loop_limit_,
      "IndexNSGFlat deserialize: n_total exceeds loop limit (%zu > %zu)", total,
      deserialization_loop_limit_);

  NSGIndexOptions options;
  READ1(options.knn_degree);
  READ1(options.nn_descent_iterations);
  READ1(options.nn_descent_convergence_threshold);
  READ1(options.random_seed);
  READ1(options.max_degree);
  READ1(options.build_search_width);
  READ1(options.candidate_pool_size);
  READ1(options.ef_search);
  options.check_relative_distance =
      read_graph_bool(f, "IndexNSGFlat check_relative_distance");
  GraphId entry_point;
  READ1(entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      (total == 0 && entry_point == kInvalidGraphId) ||
          (total > 0 && entry_point >= 0 &&
           static_cast<size_t>(entry_point) < total),
      "IndexNSGFlat deserialize: entry point is inconsistent with n_total");

  auto index = std::make_unique<IndexNSGFlat>(header.d, header.metric_type,
                                              options, header.metric_arg);
  HYPERVEC_THROW_IF_NOT_MSG(
      header.is_trained == index->is_trained,
      "IndexNSGFlat deserialize: training state does not match flat storage");

  const size_t code_count = mul_no_overflow(
      total, index->CodeStore().CodeSize(), "IndexNSGFlat codes");
  std::vector<uint8_t> codes;
  ReadVectorExact(codes, code_count, f, "IndexNSGFlat codes");

  const size_t offset_count = add_no_overflow(total, 1, "IndexNSGFlat offsets");
  std::vector<size_t> offsets;
  ReadVectorExact(offsets, offset_count, f, "IndexNSGFlat offsets");
  HYPERVEC_THROW_IF_NOT_MSG(offsets.front() == 0,
                            "IndexNSGFlat offsets must start at zero");
  for (size_t node = 0; node < total; ++node) {
    HYPERVEC_THROW_IF_NOT_MSG(
        offsets[node] <= offsets[node + 1] &&
            offsets[node + 1] - offsets[node] <= options.max_degree,
        "IndexNSGFlat offsets contain an invalid neighbor span");
  }
  const size_t max_edges = mul_no_overflow(total, options.max_degree,
                                           "IndexNSGFlat maximum edge count");
  HYPERVEC_THROW_IF_NOT_MSG(
      offsets.back() <= max_edges,
      "IndexNSGFlat edge count exceeds the configured degree bound");
  std::vector<GraphId> edges;
  ReadVectorExact(edges, offsets.back(), f, "IndexNSGFlat edges");

  MutableBoundedGraph graph(total, options.max_degree);
  for (size_t node = 0; node < total; ++node) {
    const size_t degree = offsets[node + 1] - offsets[node];
    const GraphId* neighbors =
        degree == 0 ? nullptr : edges.data() + offsets[node];
    graph.SetNeighbors(static_cast<GraphId>(node),
                       GraphNeighborView(neighbors, degree));
  }
  InMemoryCodeStore code_store(index->CodeStore().CodeSize());
  code_store.Append(header.n_total, codes.data());
  index->RestoreState(std::move(code_store), std::move(graph), entry_point);
  return index;
}

static std::unique_ptr<IndexLSH> read_lsh(const IndexHeaderData& header,
                                          IOReader* f) {
  const uint64_t total_u64 = static_cast<uint64_t>(header.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      total_u64 <= std::numeric_limits<size_t>::max(),
      "IndexLSH deserialize: n_total does not fit in size_t");
  const size_t total = static_cast<size_t>(total_u64);
  HYPERVEC_THROW_IF_NOT_FMT(
      deserialization_loop_limit_ == 0 || total <= deserialization_loop_limit_,
      "IndexLSH deserialize: n_total exceeds loop limit (%zu > %zu)", total,
      deserialization_loop_limit_);

  LSHIndexOptions options;
  READ1(options.table_count);
  READ1(options.bits_per_table);
  READ1(options.probe_count);
  READ1(options.candidate_limit);
  READ1(options.random_seed);
  HYPERVEC_THROW_IF_NOT_MSG(
      options.table_count > 0 && options.bits_per_table > 0 &&
          options.bits_per_table <= 63 && options.probe_count > 0 &&
          options.probe_count <= options.bits_per_table + 1,
      "IndexLSH deserialize: invalid hash table options");
  HYPERVEC_THROW_IF_NOT_FMT(
      deserialization_loop_limit_ == 0 ||
          options.table_count <= deserialization_loop_limit_,
      "IndexLSH deserialize: table_count exceeds loop limit (%zu > %zu)",
      options.table_count, deserialization_loop_limit_);
  const size_t hyperplane_count = mul_no_overflow(
      mul_no_overflow(options.table_count, options.bits_per_table,
                      "IndexLSH hyperplane count"),
      static_cast<size_t>(header.d), "IndexLSH hyperplane elements");
  ValidateElementCount<float>(hyperplane_count, "IndexLSH hyperplanes");
  auto index =
      std::make_unique<IndexLSH>(header.d, header.metric_type, options);
  HYPERVEC_THROW_IF_NOT_MSG(
      header.is_trained == index->is_trained,
      "IndexLSH deserialize: training state does not match flat storage");

  std::vector<float> hyperplanes;
  ReadVectorExact(hyperplanes, hyperplane_count, f, "IndexLSH hyperplanes");
  const size_t code_count =
      mul_no_overflow(total, index->CodeStore().CodeSize(), "IndexLSH codes");
  std::vector<uint8_t> codes;
  ReadVectorExact(codes, code_count, f, "IndexLSH codes");

  InMemoryCodeStore code_store(index->CodeStore().CodeSize());
  code_store.Append(header.n_total, codes.data());
  index->RestoreState(std::move(code_store), std::move(hyperplanes));
  return index;
}

Index* ReadIndex(IOReader* f, int io_flags) {
  (void)io_flags;

  uint32_t h;
  READ1(h);

  if (h == fourcc("ILSh")) {
    const IndexHeaderData header = read_index_header_data(f);
    return read_lsh(header, f).release();
  }

  if (h == fourcc("INSf")) {
    const IndexHeaderData header = read_index_header_data(f);
    return read_nsw_flat(header, f).release();
  }

  if (h == fourcc("INGf")) {
    const IndexHeaderData header = read_index_header_data(f);
    return read_nsg_flat(header, f).release();
  }

  if (h == fourcc("IxMp")) {
    auto idx = std::make_unique<IndexIDMap>();
    read_index_header(*idx, f);
    ReadVectorExact(idx->rev_map, static_cast<size_t>(idx->n_total), f,
                    "IndexIDMap external IDs");
    idx->index = ReadIndex(f, 0);
    idx->own_fields = true;
    RestoreIdMap(*idx);
    return idx.release();
  }

  if (h == fourcc("IHNf")) {
    auto idxhnsw = std::make_unique<IndexHNSWFlat>();
    read_index_header(*idxhnsw, f);
    read_HNSW(idxhnsw->hnsw, *idxhnsw, f);
    idxhnsw->storage = ReadIndex(f, 0);
    idxhnsw->own_fields = true;
    HYPERVEC_THROW_IF_NOT_MSG(
        dynamic_cast<IndexFlat*>(idxhnsw->storage) != nullptr,
        "IndexHNSWFlat deserialize: inner storage is not an IndexFlat");
    ValidateHnswStorage(*idxhnsw);
    return idxhnsw.release();
  }

  if (h == fourcc("IHNp")) {
    auto idxhnsw = std::make_unique<IndexHNSWPQ>();
    read_index_header(*idxhnsw, f);
    read_HNSW(idxhnsw->hnsw, *idxhnsw, f);
    idxhnsw->storage = ReadIndex(f, 0);
    idxhnsw->own_fields = true;
    HYPERVEC_THROW_IF_NOT_MSG(
        dynamic_cast<IndexPQ*>(idxhnsw->storage) != nullptr,
        "IndexHNSWPQ deserialize: inner storage is not an IndexPQ");
    ValidateHnswStorage(*idxhnsw);
    return idxhnsw.release();
  }

  if (h == fourcc("IHNl")) {
    auto idxhnsw = std::make_unique<IndexHNSWLVQ>();
    read_index_header(*idxhnsw, f);
    read_HNSW(idxhnsw->hnsw, *idxhnsw, f);
    idxhnsw->storage = ReadIndex(f, 0);
    idxhnsw->own_fields = true;
    HYPERVEC_THROW_IF_NOT_MSG(
        dynamic_cast<IndexLVQ*>(idxhnsw->storage) != nullptr,
        "IndexHNSWLVQ deserialize: inner storage is not an IndexLVQ");
    ValidateHnswStorage(*idxhnsw);
    return idxhnsw.release();
  }

  if (h == fourcc("IFlm") || h == fourcc("IFll")) {
    auto idx = std::make_unique<IndexFlatL2>();
    read_index_header(*idx, f);
    idx->code_size = mul_no_overflow(sizeof(float), static_cast<size_t>(idx->d),
                                     "IndexFlat code size");
    const size_t code_count = mul_no_overflow(
        static_cast<size_t>(idx->n_total), idx->code_size, "IndexFlat codes");
    ReadVectorExact(idx->codes, code_count, f, "IndexFlat codes");
    return idx.release();
  }

  if (h == fourcc("IFlp")) {
    auto idx = std::make_unique<IndexFlatIP>();
    read_index_header(*idx, f);
    idx->code_size = mul_no_overflow(sizeof(float), static_cast<size_t>(idx->d),
                                     "IndexFlat code size");
    const size_t code_count = mul_no_overflow(
        static_cast<size_t>(idx->n_total), idx->code_size, "IndexFlat codes");
    ReadVectorExact(idx->codes, code_count, f, "IndexFlat codes");
    return idx.release();
  }

  if (h == fourcc("IVFf")) {
    auto idx = std::make_unique<IndexIVFFlat>();
    read_index_header(*idx, f);
    READ1(idx->nlist);
    READ1(idx->nprobe);
    const size_t centroid_count = ValidateIvfMetadata(*idx);
    ReadVectorExact(idx->centroids, centroid_count, f,
                    "IndexIVFFlat centroids");

    const size_t code_size = static_cast<size_t>(idx->d) * sizeof(float);
    ReadInvertedLists(*idx, code_size, f);
    return idx.release();
  }

  if (h == fourcc("IPQ8")) {
    auto idx = std::make_unique<IndexPQ>();
    read_index_header(*idx, f);
    read_pq(idx->pq, f);
    HYPERVEC_THROW_IF_NOT_FMT(
        idx->pq.d == idx->d,
        "IndexPQ deserialize: pq.d (%" PRId64 ") != index.d (%" PRId64 ")",
        static_cast<int64_t>(idx->pq.d), static_cast<int64_t>(idx->d));
    HYPERVEC_THROW_IF_NOT_MSG(
        idx->metric_type == kMetricL2,
        "IndexPQ deserialize: only kMetricL2 is supported");
    idx->pq.is_trained = idx->is_trained;
    const size_t code_count = mul_no_overflow(
        static_cast<size_t>(idx->n_total), idx->pq.code_size, "IndexPQ codes");
    ReadVectorExact(idx->codes, code_count, f, "IndexPQ codes");
    return idx.release();
  }

  if (h == fourcc("ILVQ")) {
    auto idx = std::make_unique<IndexLVQ>();
    read_index_header(*idx, f);
    read_lvq(idx->lvq, f);
    HYPERVEC_THROW_IF_NOT_FMT(
        idx->lvq.d == idx->d,
        "IndexLVQ deserialize: lvq.d (%" PRId64 ") != index.d (%" PRId64 ")",
        static_cast<int64_t>(idx->lvq.d), static_cast<int64_t>(idx->d));
    HYPERVEC_THROW_IF_NOT_MSG(
        idx->metric_type == kMetricL2,
        "IndexLVQ deserialize: only kMetricL2 is supported");
    idx->lvq.is_trained = idx->is_trained;
    const size_t code_count =
        mul_no_overflow(static_cast<size_t>(idx->n_total), idx->lvq.code_size,
                        "IndexLVQ codes");
    ReadVectorExact(idx->codes, code_count, f, "IndexLVQ codes");
    return idx.release();
  }

  if (h == fourcc("IVPQ")) {
    auto idx = std::make_unique<IndexIVFPQ>();
    read_index_header(*idx, f);
    READ1(idx->nlist);
    READ1(idx->nprobe);
    const size_t centroid_count = ValidateIvfMetadata(*idx);
    ReadVectorExact(idx->centroids, centroid_count, f, "IndexIVFPQ centroids");
    int8_t by_residual_raw;
    int upt;
    READ1(by_residual_raw);
    READ1(upt);
    HYPERVEC_THROW_IF_NOT_MSG(
        by_residual_raw == 0 || by_residual_raw == 1,
        "IndexIVFPQ deserialize: by_residual must be encoded as 0 or 1");
    HYPERVEC_THROW_IF_NOT_MSG(
        upt == 0 || upt == 1,
        "IndexIVFPQ deserialize: invalid precomputed-table mode");
    HYPERVEC_THROW_IF_NOT_MSG(
        upt == 0 || by_residual_raw == 1,
        "IndexIVFPQ deserialize: precomputed tables require residual codes");
    idx->by_residual = (by_residual_raw != 0);
    idx->use_precomputed_table = upt;
    read_pq(idx->pq, f);
    HYPERVEC_THROW_IF_NOT_FMT(
        idx->pq.d == idx->d,
        "IndexIVFPQ deserialize: pq.d (%" PRId64 ") != index.d (%" PRId64 ")",
        static_cast<int64_t>(idx->pq.d), static_cast<int64_t>(idx->d));
    HYPERVEC_THROW_IF_NOT_MSG(
        idx->metric_type == kMetricL2,
        "IndexIVFPQ deserialize: only kMetricL2 is supported");
    idx->pq.is_trained = idx->is_trained;
    size_t precomputed_count = 0;
    if (idx->is_trained && idx->use_precomputed_table != 0) {
      const size_t entries_per_list = mul_no_overflow(
          static_cast<size_t>(idx->pq.M), static_cast<size_t>(idx->pq.ksub),
          "IndexIVFPQ precomputed table");
      precomputed_count =
          mul_no_overflow(static_cast<size_t>(idx->nlist), entries_per_list,
                          "IndexIVFPQ precomputed table");
    }
    ReadVectorExact(idx->precomputed_table, precomputed_count, f,
                    "IndexIVFPQ precomputed table");

    ReadInvertedLists(*idx, idx->pq.code_size, f);
    return idx.release();
  }

  if (h == fourcc("IVLQ")) {
    auto idx = std::make_unique<IndexIVFLVQ>();
    read_index_header(*idx, f);
    READ1(idx->nlist);
    READ1(idx->nprobe);
    const size_t centroid_count = ValidateIvfMetadata(*idx);
    ReadVectorExact(idx->centroids, centroid_count, f, "IndexIVFLVQ centroids");
    int8_t by_residual_raw;
    READ1(by_residual_raw);
    HYPERVEC_THROW_IF_NOT_MSG(
        by_residual_raw == 0 || by_residual_raw == 1,
        "IndexIVFLVQ deserialize: by_residual must be encoded as 0 or 1");
    idx->by_residual = (by_residual_raw != 0);
    read_lvq(idx->lvq, f);
    HYPERVEC_THROW_IF_NOT_FMT(
        idx->lvq.d == idx->d,
        "IndexIVFLVQ deserialize: lvq.d (%" PRId64 ") != index.d (%" PRId64 ")",
        static_cast<int64_t>(idx->lvq.d), static_cast<int64_t>(idx->d));
    HYPERVEC_THROW_IF_NOT_MSG(
        idx->metric_type == kMetricL2,
        "IndexIVFLVQ deserialize: only kMetricL2 is supported");
    idx->lvq.is_trained = idx->is_trained;

    ReadInvertedLists(*idx, idx->lvq.code_size, f);
    return idx.release();
  }

  HYPERVEC_THROW_MSG("unknown index type");
}

ProductQuantizer* read_ProductQuantizer(IOReader* f) {
  uint32_t h;
  READ1(h);
  HYPERVEC_THROW_IF_NOT_MSG(h == fourcc("PqPq"),
                            "read_ProductQuantizer: bad magic");
  auto pq = std::make_unique<ProductQuantizer>();
  read_pq(*pq, f);
  return pq.release();
}

ProductQuantizer* read_ProductQuantizer(const char* fname) {
  std::unique_ptr<IOReader> f(new FileIOReader(fname));
  return read_ProductQuantizer(f.get());
}

std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(IOReader* f) {
  return std::unique_ptr<ProductQuantizer>(read_ProductQuantizer(f));
}

std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(const char* fname) {
  return std::unique_ptr<ProductQuantizer>(read_ProductQuantizer(fname));
}

LocalVectorQuantizer* read_LocalVectorQuantizer(IOReader* f) {
  uint32_t h;
  READ1(h);
  HYPERVEC_THROW_IF_NOT_MSG(h == fourcc("LvQq"),
                            "read_LocalVectorQuantizer: bad magic");
  auto lvq = std::make_unique<LocalVectorQuantizer>();
  read_lvq(*lvq, f);
  return lvq.release();
}

LocalVectorQuantizer* read_LocalVectorQuantizer(const char* fname) {
  std::unique_ptr<IOReader> f(new FileIOReader(fname));
  return read_LocalVectorQuantizer(f.get());
}

std::unique_ptr<LocalVectorQuantizer> read_LocalVectorQuantizer_up(
    IOReader* f) {
  return std::unique_ptr<LocalVectorQuantizer>(read_LocalVectorQuantizer(f));
}

std::unique_ptr<LocalVectorQuantizer> read_LocalVectorQuantizer_up(
    const char* fname) {
  return std::unique_ptr<LocalVectorQuantizer>(
      read_LocalVectorQuantizer(fname));
}

Index* ReadIndex(FILE* f, int io_flags) {
  FileIOReader reader(f);
  return ReadIndex(&reader, io_flags);
}

Index* ReadIndex(const char* fname, int io_flags) {
  std::unique_ptr<IOReader> f(new FileIOReader(fname));
  return ReadIndex(f.get(), io_flags);
}

std::unique_ptr<Index> ReadIndexUp(IOReader* reader, int io_flags) {
  return std::unique_ptr<Index>(ReadIndex(reader, io_flags));
}

std::unique_ptr<Index> ReadIndexUp(FILE* f, int io_flags) {
  return std::unique_ptr<Index>(ReadIndex(f, io_flags));
}

std::unique_ptr<Index> ReadIndexUp(const char* fname, int io_flags) {
  return std::unique_ptr<Index>(ReadIndex(fname, io_flags));
}

}  // namespace hypervec
