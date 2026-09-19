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
#include <index/ivf/index_ivf.h>
#include <index/ivf/index_ivf_flat.h>
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

#include <cinttypes>
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

static void read_index_header(Index& idx, IOReader* f) {
  READ1(idx.d);
  READ1(idx.n_total);
  HYPERVEC_CHECK_RANGE(idx.d, 0, (1 << 20) + 1);
  HYPERVEC_THROW_IF_NOT_FMT(idx.n_total >= 0,
                            "invalid n_total %" PRId64 " read from index",
                            (int64_t)idx.n_total);
  idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  uint8_t is_trained_raw;
  READ1(is_trained_raw);
  HYPERVEC_THROW_IF_NOT_MSG(
      is_trained_raw <= 1,
      "index deserialize: is_trained must be encoded as 0 or 1");
  idx.is_trained = is_trained_raw != 0;
  int metric_type_int;
  READ1(metric_type_int);
  idx.metric_type = MetricTypeFromInt(metric_type_int);
  if (idx.metric_type > 1) {
    READ1(idx.metric_arg);
  }
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

static void read_HNSW(HNSW& hnsw, IOReader* f) {
  int M;
  READ1(M);
  hnsw.SetDefaultProbas(M, 1.0f / log(M));
  READ1(hnsw.ef_construction);
  READ1(hnsw.max_level);
  READ1(hnsw.entry_point);
  int nb_levels;
  READ1(nb_levels);
  (void)nb_levels;
  READVECTOR(hnsw.cum_nneighbor_per_level);
  READVECTOR(hnsw.levels);
  READVECTOR(hnsw.neighbors);
  READVECTOR(hnsw.offsets);
}

Index* ReadIndex(IOReader* f, int io_flags) {
  (void)io_flags;

  uint32_t h;
  READ1(h);

  if (h == fourcc("IHNf")) {
    auto idxhnsw = std::make_unique<IndexHNSWFlat>();
    read_index_header(*idxhnsw, f);
    read_HNSW(idxhnsw->hnsw, f);
    idxhnsw->storage = ReadIndex(f, 0);
    HYPERVEC_THROW_IF_NOT_MSG(
        dynamic_cast<IndexFlat*>(idxhnsw->storage) != nullptr,
        "IndexHNSWFlat deserialize: inner storage is not an IndexFlat");
    HYPERVEC_THROW_IF_NOT_MSG(
        idxhnsw->storage->is_trained,
        "IndexHNSWFlat deserialize: inner IndexFlat is not trained");
    idxhnsw->own_fields = true;
    idxhnsw->is_trained = true;
    return idxhnsw.release();
  }

  if (h == fourcc("IHNp")) {
    auto idxhnsw = std::make_unique<IndexHNSWPQ>();
    read_index_header(*idxhnsw, f);
    read_HNSW(idxhnsw->hnsw, f);
    idxhnsw->storage = ReadIndex(f, 0);
    HYPERVEC_THROW_IF_NOT_MSG(
        dynamic_cast<IndexPQ*>(idxhnsw->storage) != nullptr,
        "IndexHNSWPQ deserialize: inner storage is not an IndexPQ");
    HYPERVEC_THROW_IF_NOT_MSG(
        idxhnsw->storage->is_trained,
        "IndexHNSWPQ deserialize: inner IndexPQ is not trained");
    idxhnsw->own_fields = true;
    return idxhnsw.release();
  }

  if (h == fourcc("IHNl")) {
    auto idxhnsw = std::make_unique<IndexHNSWLVQ>();
    read_index_header(*idxhnsw, f);
    read_HNSW(idxhnsw->hnsw, f);
    idxhnsw->storage = ReadIndex(f, 0);
    HYPERVEC_THROW_IF_NOT_MSG(
        dynamic_cast<IndexLVQ*>(idxhnsw->storage) != nullptr,
        "IndexHNSWLVQ deserialize: inner storage is not an IndexLVQ");
    HYPERVEC_THROW_IF_NOT_MSG(
        idxhnsw->storage->is_trained,
        "IndexHNSWLVQ deserialize: inner IndexLVQ is not trained");
    idxhnsw->own_fields = true;
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
