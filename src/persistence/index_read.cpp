/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 (the "License") found in the
 * LICENSE file in the root directory of this source tree.

 * HNSW-only index read implementation
 */

#include <utils/log/assert.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <persistence/mapped_io.h>
#include <invlists/inverted_lists.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>

namespace hypervec {

namespace {
size_t deserialization_loop_limit_ = 0;
size_t deserialization_vector_byte_limit_ = uint64_t{1} << 40;  // 1 TB
}  // namespace

size_t get_deserialization_loop_limit() {
  return deserialization_loop_limit_;
}

void set_deserialization_loop_limit(size_t value) {
  deserialization_loop_limit_ = value;
}

size_t get_deserialization_vector_byte_limit() {
  return deserialization_vector_byte_limit_;
}

void set_deserialization_vector_byte_limit(size_t value) {
  deserialization_vector_byte_limit_ = value;
}

/*************************************************************
 * Read
 **************************************************************/

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
  READ1(idx.is_trained);
  int metric_type_int;
  READ1(metric_type_int);
  idx.metric_type = MetricTypeFromInt(metric_type_int);
  if (idx.metric_type > 1) {
    READ1(idx.metric_arg);
  }
  idx.verbose = false;
}

static void read_pq(ProductQuantizer& pq, IOReader* f) {
  // Inverse of write_pq. SetDerivedValues validates (d, M, nbits) and
  // resizes the centroid table; the subsequent READVECTOR reuses the same
  // underlying std::vector storage.
  READ1(pq.d);
  READ1(pq.M);
  READ1(pq.nbits);
  pq.SetDerivedValues();
  READVECTOR(pq.centroids);
  pq.is_trained = true;
}

static void read_HNSW(HNSW& hnsw, IOReader* f) {
  // M is not directly stored, compute from cum_nneighbor_per_level
  int M;
  READ1(M);
  hnsw.SetDefaultProbas(M, 1.0f / log(M));
  READ1(hnsw.ef_construction);
  READ1(hnsw.max_level);
  READ1(hnsw.entry_point);
  int nb_levels;
  READ1(nb_levels);
  (void)nb_levels;  // stored in levels.size() implicitly
  READVECTOR(hnsw.cum_nneighbor_per_level);
  READVECTOR(hnsw.levels);
  READVECTOR(hnsw.neighbors);
  READVECTOR(hnsw.offsets);
}

/*************************************************************
 * HNSW index reading
 **************************************************************/

Index* ReadIndex(IOReader* f, int io_flags) {
  (void)io_flags;  // not used in HNSW-only implementation

  uint32_t h;
  READ1(h);

  // IHNs = HNSW Flat
  if (h == fourcc("IHNf")) {
    auto idxhnsw = std::make_unique<IndexHNSWFlat>();
    read_index_header(*idxhnsw, f);
    read_HNSW(idxhnsw->hnsw, f);
    idxhnsw->storage = ReadIndex(f, 0);
    return idxhnsw.release();
  }

  // Basic indexes
  // IFlm = IndexFlatL2
  if (h == fourcc("IFlm") || h == fourcc("IFll")) {
    auto idx = std::make_unique<IndexFlatL2>();
    read_index_header(*idx, f);
    READVECTOR(idx->codes);
    return idx.release();
  }

  // IFlp = IndexFlatIP
  if (h == fourcc("IFlp")) {
    auto idx = std::make_unique<IndexFlatIP>();
    read_index_header(*idx, f);
    READVECTOR(idx->codes);
    return idx.release();
  }

  // IPQ8 = IndexPQ (any nbits in [1, 16] — magic name is historic)
  if (h == fourcc("IPQ8")) {
    auto idx = std::make_unique<IndexPQ>();
    read_index_header(*idx, f);
    read_pq(idx->pq, f);
    HYPERVEC_THROW_IF_NOT_FMT(
      idx->pq.d == idx->d,
      "IndexPQ deserialize: pq.d (%lld) != index.d (%lld)",
      static_cast<long long>(idx->pq.d), static_cast<long long>(idx->d));
    READVECTOR(idx->codes);
    return idx.release();
  }

  // IVPQ = IndexIVFPQ
  if (h == fourcc("IVPQ")) {
    auto idx = std::make_unique<IndexIVFPQ>();
    read_index_header(*idx, f);
    READ1(idx->nlist);
    READ1(idx->nprobe);
    READVECTOR(idx->centroids);
    int8_t by_residual_raw;
    int upt;
    READ1(by_residual_raw);
    READ1(upt);
    idx->by_residual = (by_residual_raw != 0);
    idx->use_precomputed_table = upt;
    read_pq(idx->pq, f);
    HYPERVEC_THROW_IF_NOT_FMT(
      idx->pq.d == idx->d,
      "IndexIVFPQ deserialize: pq.d (%lld) != index.d (%lld)",
      static_cast<long long>(idx->pq.d), static_cast<long long>(idx->d));
    READVECTOR(idx->precomputed_table);

    // Replace the empty default ArrayInvertedLists installed by the base
    // ctor with one sized to (nlist, pq.code_size).
    delete idx->invlists;
    idx->invlists = new ArrayInvertedLists(static_cast<size_t>(idx->nlist),
                                           idx->pq.code_size);
    idx->own_invlists = true;

    // Per-list payload — see the matching write loop.
    for (size_t list_no = 0; list_no < static_cast<size_t>(idx->nlist);
         list_no++) {
      size_t sz;
      READ1(sz);
      if (sz == 0) {
        continue;
      }
      // Bounds-check against the configurable byte budget so a corrupted
      // file can't trigger a giant allocation.
      HYPERVEC_THROW_IF_NOT(sz < (get_deserialization_vector_byte_limit() /
                                  sizeof(idx_t)));
      std::vector<idx_t> ids(sz);
      std::vector<uint8_t> codes(sz * idx->pq.code_size);
      READANDCHECK(ids.data(), sz);
      READANDCHECK(codes.data(), sz * idx->pq.code_size);
      idx->invlists->add_entries(list_no, sz, ids.data(), codes.data());
    }
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