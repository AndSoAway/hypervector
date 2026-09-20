/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * HNSW-only index read implementation
 */

#include <index/diskann/index_diskann.h>
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
#include <index/pretransform/index_pre_transform.h>
#include <invlists/inverted_lists.h>
#include <persistence/index_io.h>
#include <persistence/index_io_builtins.h>
#include <persistence/index_io_registry.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <persistence/mapped_io.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <quantization/rabitq/index_ivf_rabitq.h>
#include <transform/opq_matrix.h>
#include <transform/vector_transform.h>
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
thread_local size_t pretransform_read_depth_ = 0;

class PreTransformReadGuard {
 public:
  PreTransformReadGuard() {
    HYPERVEC_THROW_IF_NOT_FMT(
        deserialization_loop_limit_ == 0 ||
            pretransform_read_depth_ < deserialization_loop_limit_,
        "IndexPreTransform deserialize: transform chain exceeds loop limit "
        "(%zu)",
        deserialization_loop_limit_);
    ++pretransform_read_depth_;
  }

  ~PreTransformReadGuard() { --pretransform_read_depth_; }
};

template <typename T>
void ValidateElementCount(size_t count, const char* context) {
  const size_t bytes = mul_no_overflow(count, sizeof(T), context);
  HYPERVEC_THROW_IF_NOT_FMT(bytes < deserialization_vector_byte_limit_,
                            "%s payload is too large: %zu bytes (limit %zu)",
                            context, bytes, deserialization_vector_byte_limit_);
}

template <typename Vector>
void ReadElements(Vector& values, size_t count, IOReader* f,
                  const char* context) {
  (void)context;
  values.resize(count);
  READANDCHECK(values.data(), count);
}

template <typename T>
void ReadElements(MaybeOwnedVector<T>& values, size_t count, IOReader* f,
                  const char* context) {
  auto* mapped_reader = dynamic_cast<MappedFileIOReader*>(f);
  if (mapped_reader == nullptr || count == 0) {
    values.resize(count);
    READANDCHECK(values.data(), count);
    return;
  }

  void* address = nullptr;
  const size_t mapped_count = mapped_reader->mmap(&address, sizeof(T), count);
  HYPERVEC_THROW_IF_NOT_FMT(mapped_count == count,
                            "read error for %s in %s: %zu != %zu", context,
                            f->name.c_str(), mapped_count, count);

  if (reinterpret_cast<uintptr_t>(address) % alignof(T) == 0) {
    values = MaybeOwnedVector<T>::create_view(address, count,
                                              mapped_reader->mmap_owner);
    return;
  }

  values.resize(count);
  std::memcpy(values.data(), address, count * sizeof(T));
}

template <typename Vector>
void ReadVector(Vector& values, IOReader* f, const char* context) {
  size_t serialized_size;
  READANDCHECK(&serialized_size, 1);
  using value_type = typename Vector::value_type;
  ValidateElementCount<value_type>(serialized_size, context);
  ReadElements(values, serialized_size, f, context);
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
  ReadElements(values, serialized_size, f, context);
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
    MaybeOwnedVector<idx_t> ids;
    MaybeOwnedVector<uint8_t> codes;
    ReadElements(ids, list_size, f, "IndexIVF list ids");
    ReadElements(codes, code_count, f, "IndexIVF list codes");
    loaded->ids[list_no] = std::move(ids);
    loaded->codes[list_no] = std::move(codes);
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

void ValidateRaBitQCodes(const IndexIVFRaBitQ& index) {
  HYPERVEC_THROW_IF_NOT_MSG(
      index.rabitq != nullptr && index.invlists != nullptr &&
          index.invlists->code_size == index.rabitq->CodeSize(),
      "IndexIVFRaBitQ deserialize: code storage is inconsistent");
  const size_t factor_offset = index.rabitq->BitBytes();
  for (size_t list_no = 0; list_no < static_cast<size_t>(index.nlist);
       ++list_no) {
    const size_t list_size = index.invlists->list_size(list_no);
    if (list_size == 0) {
      continue;
    }
    InvertedLists::ScopedCodes codes(index.invlists, list_no);
    for (size_t offset = 0; offset < list_size; ++offset) {
      const uint8_t* code = codes.get() + offset * index.rabitq->CodeSize();
      float norm_squared;
      float scale;
      std::memcpy(&norm_squared, code + factor_offset, sizeof(float));
      std::memcpy(&scale, code + factor_offset + sizeof(float), sizeof(float));
      HYPERVEC_THROW_IF_NOT_MSG(
          std::isfinite(norm_squared) && norm_squared >= 0.0F &&
              std::isfinite(scale) && scale >= 0.0F,
          "IndexIVFRaBitQ deserialize: code factors are invalid");
    }
  }
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
  ReadVector(hnsw.cum_nneighbor_per_level, f,
             "IndexHNSW neighbor-capacity table");
  ReadVectorExact(hnsw.levels, static_cast<size_t>(index.n_total), f,
                  "IndexHNSW levels");
  ReadVector(hnsw.neighbors, f, "IndexHNSW neighbors");
  const size_t offset_count = add_no_overflow(
      static_cast<size_t>(index.n_total), 1, "IndexHNSW offsets");
  ReadVectorExact(hnsw.offsets, offset_count, f, "IndexHNSW offsets");
  ValidateHnswGraph(index, hnsw, level0_capacity, serialized_last_level);
  RebuildHnswLevelProbabilities(hnsw);
}

static bool read_bool(IOReader* f, const char* context) {
  uint8_t value;
  READ1(value);
  HYPERVEC_THROW_IF_NOT_FMT(value <= 1, "%s must be encoded as 0 or 1",
                            context);
  return value != 0;
}

struct LinearTransformState {
  idx_t d_in = 0;
  idx_t d_out = 0;
  bool is_orthonormal = false;
  std::vector<float> matrix;
  std::vector<float> bias;
};

static LinearTransformState read_linear_transform(IOReader* f) {
  LinearTransformState state;
  READ1(state.d_in);
  READ1(state.d_out);
  HYPERVEC_THROW_IF_NOT_MSG(
      state.d_in > 0 && state.d_in <= (1 << 20) && state.d_out > 0 &&
          state.d_out <= (1 << 20),
      "LinearTransform deserialize: dimensions are out of range");
  state.is_orthonormal = read_bool(f, "LinearTransform is_orthonormal");
  const bool has_bias = read_bool(f, "LinearTransform has_bias");
  const size_t matrix_size = mul_no_overflow(static_cast<size_t>(state.d_in),
                                             static_cast<size_t>(state.d_out),
                                             "LinearTransform matrix");
  ReadVectorExact(state.matrix, matrix_size, f, "LinearTransform matrix");
  if (has_bias) {
    ReadVectorExact(state.bias, static_cast<size_t>(state.d_out), f,
                    "LinearTransform bias");
  }
  return state;
}

static std::unique_ptr<VectorTransform> read_transform(IOReader* f) {
  uint32_t h;
  READ1(h);
  if (h == fourcc("LiTr")) {
    LinearTransformState state = read_linear_transform(f);
    auto transform = std::make_unique<LinearTransform>(state.d_in, state.d_out);
    transform->SetTransform(std::move(state.matrix), std::move(state.bias),
                            state.is_orthonormal);
    return transform;
  }

  if (h == fourcc("OPQt")) {
    idx_t subquantizer_count;
    int nbits;
    OPQParameters parameters;
    READ1(subquantizer_count);
    READ1(nbits);
    READ1(parameters.iterations);
    READ1(parameters.pq_parameters.niter);
    READ1(parameters.pq_parameters.seed);
    READ1(parameters.pq_parameters.nredo);
    parameters.pq_parameters.verbose = read_bool(f, "OPQMatrix verbose");
    HYPERVEC_THROW_IF_NOT_MSG(
        parameters.iterations > 0 && parameters.pq_parameters.niter > 0 &&
            parameters.pq_parameters.nredo > 0,
        "OPQMatrix deserialize: training parameters must be positive");

    LinearTransformState state = read_linear_transform(f);
    HYPERVEC_THROW_IF_NOT_MSG(
        state.d_in == state.d_out && state.is_orthonormal && state.bias.empty(),
        "OPQMatrix deserialize: transform must be an unbiased orthogonal "
        "rotation");
    auto transform =
        std::make_unique<OPQMatrix>(state.d_in, subquantizer_count, nbits);
    transform->parameters = parameters;
    transform->SetTransform(std::move(state.matrix), {}, true);
    return transform;
  }

  HYPERVEC_THROW_MSG("unknown vector transform type");
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
      read_bool(f, "IndexNSWFlat check_relative_distance");
  options.fill_to_max_degree = read_bool(f, "IndexNSWFlat fill_to_max_degree");
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

static std::unique_ptr<IndexDiskANNFlat> read_diskann_flat(
    const IndexHeaderData& header, IOReader* f) {
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  const uint64_t total_u64 = static_cast<uint64_t>(header.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      total_u64 <= std::numeric_limits<size_t>::max(),
      "IndexDiskANNFlat deserialize: n_total does not fit in size_t");
  const size_t total = static_cast<size_t>(total_u64);
  HYPERVEC_THROW_IF_NOT_MSG(
      total <= kGraphCapacity,
      "IndexDiskANNFlat deserialize: n_total exceeds graph ID capacity");
  HYPERVEC_THROW_IF_NOT_FMT(
      deserialization_loop_limit_ == 0 || total <= deserialization_loop_limit_,
      "IndexDiskANNFlat deserialize: n_total exceeds loop limit (%zu > %zu)",
      total, deserialization_loop_limit_);

  DiskAnnIndexOptions options;
  READ1(options.max_degree);
  READ1(options.build_search_width);
  READ1(options.candidate_pool_size);
  READ1(options.alpha);
  READ1(options.build_passes);
  READ1(options.random_seed);
  READ1(options.search_width);
  options.check_relative_distance =
      read_bool(f, "IndexDiskANNFlat check_relative_distance");
  READ1(options.page_size);
  READ1(options.cache_capacity_pages);
  GraphId entry_point;
  READ1(entry_point);

  auto index =
      std::make_unique<IndexDiskANNFlat>(header.d, header.metric_type, options);
  HYPERVEC_THROW_IF_NOT_MSG(
      header.is_trained == index->is_trained && header.metric_arg == 0.0F,
      "IndexDiskANNFlat deserialize: header metadata is inconsistent");
  const size_t code_count = mul_no_overflow(
      total, index->CodeStore().CodeSize(), "IndexDiskANNFlat codes");
  std::vector<uint8_t> codes;
  ReadVectorExact(codes, code_count, f, "IndexDiskANNFlat codes");

  const DiskAnnNodeLayout layout(total, static_cast<size_t>(header.d),
                                 options.max_degree, options.page_size);
  HYPERVEC_THROW_IF_NOT_MSG(
      layout.StorageSize() <= std::numeric_limits<size_t>::max(),
      "IndexDiskANNFlat deserialize: node payload does not fit in size_t");
  std::vector<uint8_t> node_data;
  ReadVectorExact(node_data, static_cast<size_t>(layout.StorageSize()), f,
                  "IndexDiskANNFlat node data");
  for (size_t node = 0; node < total; ++node) {
    const DiskAnnNodeLocation location =
        layout.Locate(static_cast<GraphId>(node));
    const size_t page_offset =
        mul_no_overflow(static_cast<size_t>(location.page_id),
                        layout.PageSize(), "IndexDiskANNFlat node page offset");
    const size_t vector_offset = add_no_overflow(
        page_offset, location.offset, "IndexDiskANNFlat vector offset");
    const uint8_t* vector = node_data.data() + vector_offset;
    const uint8_t* code = codes.data() + node * index->CodeStore().CodeSize();
    HYPERVEC_THROW_IF_NOT_MSG(
        std::memcmp(vector, code, layout.VectorBytes()) == 0,
        "IndexDiskANNFlat deserialize: traversal codes and raw vectors "
        "differ");
    for (size_t component = 0; component < layout.Dimension(); ++component) {
      float value;
      std::memcpy(&value, vector + component * sizeof(float), sizeof(value));
      HYPERVEC_THROW_IF_NOT_MSG(
          std::isfinite(value),
          "IndexDiskANNFlat deserialize: raw vectors must be finite");
    }
  }

  InMemoryCodeStore code_store(index->CodeStore().CodeSize());
  code_store.Append(header.n_total, codes.data());
  std::shared_ptr<RandomAccessReader> reader;
  if (total != 0) {
    reader = std::make_shared<VectorRandomAccessReader>(std::move(node_data));
  }
  index->RestoreState(std::move(code_store), std::move(reader), entry_point);
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
      read_bool(f, "IndexNSGFlat check_relative_distance");
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

namespace persistence_internal {

std::unique_ptr<Index> ReadFlatL2Payload(IOReader* reader, int io_flags) {
  (void)io_flags;
  auto index = std::make_unique<IndexFlatL2>();
  read_index_header(*index, reader);
  index->code_size = mul_no_overflow(
      sizeof(float), static_cast<size_t>(index->d), "IndexFlat code size");
  const size_t code_count = mul_no_overflow(
      static_cast<size_t>(index->n_total), index->code_size, "IndexFlat codes");
  ReadVectorExact(index->codes, code_count, reader, "IndexFlat codes");
  return index;
}

std::unique_ptr<Index> ReadFlatIPPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  auto index = std::make_unique<IndexFlatIP>();
  read_index_header(*index, reader);
  index->code_size = mul_no_overflow(
      sizeof(float), static_cast<size_t>(index->d), "IndexFlat code size");
  const size_t code_count = mul_no_overflow(
      static_cast<size_t>(index->n_total), index->code_size, "IndexFlat codes");
  ReadVectorExact(index->codes, code_count, reader, "IndexFlat codes");
  return index;
}

std::unique_ptr<Index> ReadPQPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  auto index = std::make_unique<IndexPQ>();
  read_index_header(*index, reader);
  read_pq(index->pq, reader);
  HYPERVEC_THROW_IF_NOT_FMT(
      index->pq.d == index->d,
      "IndexPQ deserialize: pq.d (%" PRId64 ") != index.d (%" PRId64 ")",
      static_cast<int64_t>(index->pq.d), static_cast<int64_t>(index->d));
  HYPERVEC_THROW_IF_NOT_MSG(index->metric_type == kMetricL2,
                            "IndexPQ deserialize: only kMetricL2 is supported");
  index->pq.is_trained = index->is_trained;
  const size_t code_count =
      mul_no_overflow(static_cast<size_t>(index->n_total), index->pq.code_size,
                      "IndexPQ codes");
  ReadVectorExact(index->codes, code_count, reader, "IndexPQ codes");
  return index;
}

std::unique_ptr<Index> ReadLVQPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  auto index = std::make_unique<IndexLVQ>();
  read_index_header(*index, reader);
  read_lvq(index->lvq, reader);
  HYPERVEC_THROW_IF_NOT_FMT(
      index->lvq.d == index->d,
      "IndexLVQ deserialize: lvq.d (%" PRId64 ") != index.d (%" PRId64 ")",
      static_cast<int64_t>(index->lvq.d), static_cast<int64_t>(index->d));
  HYPERVEC_THROW_IF_NOT_MSG(
      index->metric_type == kMetricL2,
      "IndexLVQ deserialize: only kMetricL2 is supported");
  index->lvq.is_trained = index->is_trained;
  const size_t code_count =
      mul_no_overflow(static_cast<size_t>(index->n_total), index->lvq.code_size,
                      "IndexLVQ codes");
  ReadVectorExact(index->codes, code_count, reader, "IndexLVQ codes");
  return index;
}

std::unique_ptr<Index> ReadIVFFlatPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  IOReader* f = reader;
  auto index = std::make_unique<IndexIVFFlat>();
  read_index_header(*index, reader);
  READ1(index->nlist);
  READ1(index->nprobe);
  const size_t centroid_count = ValidateIvfMetadata(*index);
  ReadVectorExact(index->centroids, centroid_count, reader,
                  "IndexIVFFlat centroids");
  const size_t code_size = mul_no_overflow(
      static_cast<size_t>(index->d), sizeof(float), "IndexIVFFlat code size");
  ReadInvertedLists(*index, code_size, reader);
  return index;
}

std::unique_ptr<Index> ReadIVFPQPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  IOReader* f = reader;
  auto index = std::make_unique<IndexIVFPQ>();
  read_index_header(*index, reader);
  READ1(index->nlist);
  READ1(index->nprobe);
  const size_t centroid_count = ValidateIvfMetadata(*index);
  ReadVectorExact(index->centroids, centroid_count, reader,
                  "IndexIVFPQ centroids");
  int8_t by_residual_raw;
  int precomputed_mode;
  READ1(by_residual_raw);
  READ1(precomputed_mode);
  HYPERVEC_THROW_IF_NOT_MSG(
      by_residual_raw == 0 || by_residual_raw == 1,
      "IndexIVFPQ deserialize: by_residual must be encoded as 0 or 1");
  HYPERVEC_THROW_IF_NOT_MSG(
      precomputed_mode == 0 || precomputed_mode == 1,
      "IndexIVFPQ deserialize: invalid precomputed-table mode");
  HYPERVEC_THROW_IF_NOT_MSG(
      precomputed_mode == 0 || by_residual_raw == 1,
      "IndexIVFPQ deserialize: precomputed tables require residual codes");
  index->by_residual = (by_residual_raw != 0);
  index->use_precomputed_table = precomputed_mode;
  read_pq(index->pq, reader);
  HYPERVEC_THROW_IF_NOT_FMT(
      index->pq.d == index->d,
      "IndexIVFPQ deserialize: pq.d (%" PRId64 ") != index.d (%" PRId64 ")",
      static_cast<int64_t>(index->pq.d), static_cast<int64_t>(index->d));
  HYPERVEC_THROW_IF_NOT_MSG(
      index->metric_type == kMetricL2,
      "IndexIVFPQ deserialize: only kMetricL2 is supported");
  index->pq.is_trained = index->is_trained;
  size_t precomputed_count = 0;
  if (index->is_trained && index->use_precomputed_table != 0) {
    const size_t entries_per_list = mul_no_overflow(
        static_cast<size_t>(index->pq.M), static_cast<size_t>(index->pq.ksub),
        "IndexIVFPQ precomputed table");
    precomputed_count =
        mul_no_overflow(static_cast<size_t>(index->nlist), entries_per_list,
                        "IndexIVFPQ precomputed table");
  }
  ReadVectorExact(index->precomputed_table, precomputed_count, reader,
                  "IndexIVFPQ precomputed table");
  ReadInvertedLists(*index, index->pq.code_size, reader);
  return index;
}

std::unique_ptr<Index> ReadIVFLVQPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  IOReader* f = reader;
  auto index = std::make_unique<IndexIVFLVQ>();
  read_index_header(*index, reader);
  READ1(index->nlist);
  READ1(index->nprobe);
  const size_t centroid_count = ValidateIvfMetadata(*index);
  ReadVectorExact(index->centroids, centroid_count, reader,
                  "IndexIVFLVQ centroids");
  int8_t by_residual_raw;
  READ1(by_residual_raw);
  HYPERVEC_THROW_IF_NOT_MSG(
      by_residual_raw == 0 || by_residual_raw == 1,
      "IndexIVFLVQ deserialize: by_residual must be encoded as 0 or 1");
  index->by_residual = (by_residual_raw != 0);
  read_lvq(index->lvq, reader);
  HYPERVEC_THROW_IF_NOT_FMT(
      index->lvq.d == index->d,
      "IndexIVFLVQ deserialize: lvq.d (%" PRId64 ") != index.d (%" PRId64 ")",
      static_cast<int64_t>(index->lvq.d), static_cast<int64_t>(index->d));
  HYPERVEC_THROW_IF_NOT_MSG(
      index->metric_type == kMetricL2,
      "IndexIVFLVQ deserialize: only kMetricL2 is supported");
  index->lvq.is_trained = index->is_trained;
  ReadInvertedLists(*index, index->lvq.code_size, reader);
  return index;
}

std::unique_ptr<Index> ReadIVFRaBitQPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  IOReader* f = reader;
  auto index = std::make_unique<IndexIVFRaBitQ>();
  read_index_header(*index, reader);
  READ1(index->nlist);
  READ1(index->nprobe);
  const size_t centroid_count = ValidateIvfMetadata(*index);
  HYPERVEC_THROW_IF_NOT_MSG(
      index->is_trained && index->metric_type == kMetricL2,
      "IndexIVFRaBitQ deserialize: index must be trained with kMetricL2");
  ReadVectorExact(index->centroids, centroid_count, reader,
                  "IndexIVFRaBitQ centroids");
  for (float centroid : index->centroids) {
    HYPERVEC_THROW_IF_NOT_MSG(
        std::isfinite(centroid),
        "IndexIVFRaBitQ deserialize: centroids must be finite");
  }
  index->by_residual = read_bool(reader, "IndexIVFRaBitQ by_residual");
  uint64_t random_seed;
  int rotation_rounds;
  READ1(random_seed);
  READ1(rotation_rounds);
  index->rabitq =
      std::make_unique<RaBitQQuantizer>(index->d, random_seed, rotation_rounds);
  ReadInvertedLists(*index, index->rabitq->CodeSize(), reader);
  ValidateRaBitQCodes(*index);
  return index;
}

std::unique_ptr<Index> ReadIDMapPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  auto index = std::make_unique<IndexIDMap>();
  read_index_header(*index, reader);
  ReadVectorExact(index->rev_map, static_cast<size_t>(index->n_total), reader,
                  "IndexIDMap external IDs");
  index->index = ReadIndex(reader, 0);
  index->own_fields = true;
  RestoreIdMap(*index);
  return index;
}

std::unique_ptr<Index> ReadPreTransformPayload(IOReader* reader, int io_flags) {
  (void)io_flags;
  const PreTransformReadGuard depth_guard;
  const IndexHeaderData header = read_index_header_data(reader);
  std::unique_ptr<VectorTransform> transform = read_transform(reader);
  std::unique_ptr<Index> inner(ReadIndex(reader, 0));
  auto index = std::make_unique<IndexPreTransform>(std::move(transform),
                                                   std::move(inner));
  HYPERVEC_THROW_IF_NOT_MSG(
      index->d == header.d && index->n_total == header.n_total &&
          index->is_trained == header.is_trained &&
          index->metric_type == header.metric_type &&
          index->metric_arg == header.metric_arg,
      "IndexPreTransform deserialize: component metadata does not match "
      "the wrapper");
  return index;
}

}  // namespace persistence_internal

Index* ReadIndex(IOReader* f, int io_flags) {
  HYPERVEC_THROW_IF_NOT_FMT((io_flags & ~IO_FLAG_MMAP_IFC) == 0,
                            "ReadIndex: unsupported I/O flags 0x%x", io_flags);
  if ((io_flags & IO_FLAG_MMAP_IFC) != 0 &&
      dynamic_cast<MappedFileIOReader*>(f) == nullptr) {
    auto* file_reader = dynamic_cast<FileIOReader*>(f);
    HYPERVEC_THROW_IF_NOT_MSG(
        file_reader != nullptr,
        "ReadIndex: IO_FLAG_MMAP_IFC requires a file-backed reader");
    const auto position = std::ftell(file_reader->f);
    HYPERVEC_THROW_IF_NOT_MSG(
        position >= 0,
        "ReadIndex: could not determine the file position for mmap");
    auto owner = std::make_shared<MmappedFileMappingOwner>(file_reader->f);
    HYPERVEC_THROW_IF_NOT_FMT(
        static_cast<size_t>(position) <= owner->size(),
        "ReadIndex: file position %ld exceeds mapped size %zu", position,
        owner->size());
    MappedFileIOReader mapped_reader(owner);
    mapped_reader.name = f->name;
    mapped_reader.pos = static_cast<size_t>(position);
    return ReadIndex(&mapped_reader, io_flags & ~IO_FLAG_MMAP_IFC);
  }

  uint32_t h;
  READ1(h);

  IndexIORegistry& registry = persistence_internal::GetBuiltinIndexIORegistry();
  if (registry.Contains(h)) {
    return registry.ReadPayload(h, f, io_flags).release();
  }

  if (h == fourcc("ILSh")) {
    const IndexHeaderData header = read_index_header_data(f);
    return read_lsh(header, f).release();
  }

  if (h == fourcc("IDAf")) {
    const IndexHeaderData header = read_index_header_data(f);
    return read_diskann_flat(header, f).release();
  }

  if (h == fourcc("INSf")) {
    const IndexHeaderData header = read_index_header_data(f);
    return read_nsw_flat(header, f).release();
  }

  if (h == fourcc("INGf")) {
    const IndexHeaderData header = read_index_header_data(f);
    return read_nsg_flat(header, f).release();
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
