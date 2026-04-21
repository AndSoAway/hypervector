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
#include <invlists/block_inverted_lists.h>
#include <persistence/index_io.h>
#include <persistence/index_read_utils.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <persistence/mapped_io.h>

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

  HYPERVEC_THROW_MSG("unknown index type");
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