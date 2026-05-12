/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 (the "License") found in the
 * LICENSE file in the root directory of this source tree.

 * HNSW-only index write implementation
 */

#include <utils/log/assert.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <invlists/inverted_lists.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>

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

static void write_pq(const ProductQuantizer& pq, IOWriter* f) {
  // Body of the PQ payload — used both standalone (after a PqPq magic) and
  // embedded inside an IPQ8 / IVPQ index. The Index header carries d/metric,
  // so when called from IPQ8 the d field is redundant but harmless; the
  // standalone PqPq path needs it.
  WRITE1(pq.d);
  WRITE1(pq.M);
  WRITE1(pq.nbits);
  WRITEVECTOR(pq.centroids);
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

  // IHNp = IndexHNSWPQ. Persists only the PQ-compressed `storage`; the
  // raw-vector scaffold is build-time only, so a deserialized index is
  // implicitly frozen.
  const IndexHNSWPQ* hnswpq = dynamic_cast<const IndexHNSWPQ*>(index);
  if (hnswpq) {
    uint32_t h = fourcc("IHNp");
    WRITE1(h);
    write_index_header(*hnswpq, f);
    write_HNSW(hnswpq->hnsw, f);
    HYPERVEC_THROW_IF_NOT(hnswpq->storage != nullptr);
    WriteIndex(hnswpq->storage, f, 0);
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

  const IndexPQ* ipq = dynamic_cast<const IndexPQ*>(index);
  if (ipq) {
    uint32_t h = fourcc("IPQ8");
    WRITE1(h);
    write_index_header(*ipq, f);
    write_pq(ipq->pq, f);
    WRITEVECTOR(ipq->codes);
    return;
  }

  const IndexIVFPQ* ivfpq = dynamic_cast<const IndexIVFPQ*>(index);
  if (ivfpq) {
    uint32_t h = fourcc("IVPQ");
    WRITE1(h);
    write_index_header(*ivfpq, f);
    WRITE1(ivfpq->nlist);
    WRITE1(ivfpq->nprobe);
    WRITEVECTOR(ivfpq->centroids);
    // by_residual / use_precomputed_table travel as fixed-width integers so
    // the on-disk format isn't bool-ABI-dependent.
    int8_t by_residual = ivfpq->by_residual ? 1 : 0;
    int upt = ivfpq->use_precomputed_table;
    WRITE1(by_residual);
    WRITE1(upt);
    write_pq(ivfpq->pq, f);
    WRITEVECTOR(ivfpq->precomputed_table);

    // Per-list payload: size, ids[], codes[]. Empty lists still write a
    // zero size so the read loop stays symmetric.
    for (size_t list_no = 0; list_no < ivfpq->nlist; list_no++) {
      const size_t sz = ivfpq->invlists->list_size(list_no);
      WRITE1(sz);
      if (sz == 0) {
        continue;
      }
      InvertedLists::ScopedIds ids(ivfpq->invlists, list_no);
      InvertedLists::ScopedCodes codes(ivfpq->invlists, list_no);
      WRITEANDCHECK(ids.get(), sz);
      WRITEANDCHECK(codes.get(), sz * ivfpq->pq.code_size);
    }
    return;
  }

  HYPERVEC_THROW_MSG("unsupported index type for writing");
}

void write_ProductQuantizer(const ProductQuantizer* pq, IOWriter* f) {
  uint32_t h = fourcc("PqPq");
  WRITE1(h);
  write_pq(*pq, f);
}

void write_ProductQuantizer(const ProductQuantizer* pq, const char* fname) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  write_ProductQuantizer(pq, f.get());
}

void WriteIndex(const Index* index, const char* fname, int io_flags) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  WriteIndex(index, f.get(), io_flags);
}

}  // namespace hypervec