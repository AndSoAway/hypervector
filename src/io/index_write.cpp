/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 (the "License") found in the
 * LICENSE file in the root directory of this source tree.

 * HNSW-only index write implementation
 */

#include <core/hypervec_assert.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <invlists/block_inverted_lists.h>
#include <io/index_io.h>
#include <io/io.h>
#include <io/io_macros.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hypervec {

/*************************************************************
 * Write
 **************************************************************/

static void write_index_header(const Index& idx, IOWriter* f) {
  WRITE1(idx.d);
  WRITE1(idx.n_total);
  idx_t dummy = 1 << 20;
  WRITE1(dummy);
  WRITE1(dummy);
  WRITE1(idx.is_trained);
  int metric = (int)idx.metric_type;
  WRITE1(metric);
  if (idx.metric_type > 1) {
    WRITE1(idx.metric_arg);
  }
}

static void write_HNSW(const HNSW& hnsw, IOWriter* f) {
  int M = hnsw.NbNeighbors(0);
  WRITE1(M);
  WRITE1(hnsw.ef_construction);
  WRITE1(hnsw.max_level);
  WRITE1(hnsw.entry_point);
  int nb_levels =
    hnsw.levels.size() > 0 ? hnsw.levels[hnsw.levels.size() - 1] : 0;
  WRITE1(nb_levels);
  WRITEVECTOR(hnsw.cum_nneighbor_per_level);
  WRITEVECTOR(hnsw.levels);
  WRITEVECTOR(hnsw.neighbors);
  WRITEVECTOR(hnsw.offsets);
}

/*************************************************************
 * HNSW index writing
 *************************************************************/

void WriteIndex(const Index* index, IOWriter* f, int io_flags) {
  (void)io_flags;  // not used in HNSW-only implementation

  const IndexHNSWFlat* hnswflat = dynamic_cast<const IndexHNSWFlat*>(index);
  if (hnswflat) {
    uint32_t h = fourcc("IHNf");
    WRITE1(h);
    write_index_header(*hnswflat, f);
    write_HNSW(hnswflat->hnsw, f);
    if (hnswflat->storage) {
      WriteIndex(hnswflat->storage, f, 0);
    }
    return;
  }

  const IndexHNSW* hnsw = dynamic_cast<const IndexHNSW*>(index);
  if (hnsw) {
    uint32_t h = fourcc("IHNf");
    WRITE1(h);
    write_index_header(*hnsw, f);
    write_HNSW(hnsw->hnsw, f);
    if (hnsw->storage) {
      WriteIndex(hnsw->storage, f, 0);
    }
    return;
  }

  const IndexFlatL2* iflatl2 = dynamic_cast<const IndexFlatL2*>(index);
  if (iflatl2) {
    uint32_t h = fourcc("IFlm");
    WRITE1(h);
    write_index_header(*iflatl2, f);
    WRITEVECTOR(iflatl2->codes);
    return;
  }

  const IndexFlatIP* iflatip = dynamic_cast<const IndexFlatIP*>(index);
  if (iflatip) {
    uint32_t h = fourcc("IFlp");
    WRITE1(h);
    write_index_header(*iflatip, f);
    WRITEVECTOR(iflatip->codes);
    return;
  }

  HYPERVEC_THROW_MSG("unsupported index type for writing");
}

void WriteIndex(const Index* index, const char* fname, int io_flags) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  WriteIndex(index, f.get(), io_flags);
}

}  // namespace hypervec