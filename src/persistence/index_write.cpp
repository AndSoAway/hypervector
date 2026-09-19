/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * HNSW-only index write implementation
 */

#include <index/flat/index_flat.h>
#include <index/graph/graph_validation.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/nsw/index_nsw.h>
#include <invlists/inverted_lists.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <utils/log/assert.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace hypervec {

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
  WRITE1(pq.d);
  WRITE1(pq.M);
  WRITE1(pq.nbits);
  WRITEVECTOR(pq.centroids);
}

static void write_lvq(const LocalVectorQuantizer& lvq, IOWriter* f) {
  WRITE1(lvq.d);
  WRITE1(lvq.nlocal);
  WRITE1(lvq.nbits);
  WRITEVECTOR(lvq.local_centroids);
  WRITEVECTOR(lvq.residual_codebooks);
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

void WriteIndex(const Index* index, IOWriter* f, int io_flags) {
  (void)io_flags;

  const auto* id_map = dynamic_cast<const IndexIDMap*>(index);
  if (id_map) {
    id_map->check_consistency();
    uint32_t h = fourcc("IxMp");
    WRITE1(h);
    write_index_header(*id_map, f);
    WRITEVECTOR(id_map->rev_map);
    WriteIndex(id_map->index, f, 0);
    return;
  }

  const auto* nswflat = dynamic_cast<const IndexNSWFlat*>(index);
  if (nswflat) {
    HYPERVEC_THROW_IF_NOT_MSG(
        nswflat->CodeStore().Size() == nswflat->n_total &&
            nswflat->Graph().NodeCount() ==
                static_cast<size_t>(nswflat->n_total),
        "IndexNSWFlat serialize: stored counts do not match n_total");
    HYPERVEC_THROW_IF_NOT_MSG(
        nswflat->QuantizerModel().Dimension() == nswflat->d &&
            nswflat->QuantizerModel().Metric() == nswflat->metric_type &&
            nswflat->QuantizerModel().IsTrained() == nswflat->is_trained,
        "IndexNSWFlat serialize: quantizer metadata does not match the index");
    const GraphValidationReport report =
        ValidateGraph(nswflat->Graph(), nswflat->EntryPoint());
    HYPERVEC_THROW_IF_NOT_MSG(
        report.IsStructurallyValid(),
        "IndexNSWFlat serialize: graph structure is invalid");

    uint32_t h = fourcc("INSf");
    WRITE1(h);
    write_index_header(*nswflat, f);
    const NSWIndexOptions& options = nswflat->Options();
    WRITE1(options.max_degree);
    WRITE1(options.ef_construction);
    WRITE1(options.ef_search);
    const uint8_t check_relative_distance = options.check_relative_distance;
    const uint8_t fill_to_max_degree = options.fill_to_max_degree;
    WRITE1(check_relative_distance);
    WRITE1(fill_to_max_degree);
    const GraphId entry_point = nswflat->EntryPoint();
    WRITE1(entry_point);

    const size_t code_bytes =
        mul_no_overflow(static_cast<size_t>(nswflat->n_total),
                        nswflat->CodeStore().CodeSize(), "IndexNSWFlat codes");
    WRITE1(code_bytes);
    WRITEANDCHECK(nswflat->CodeStore().Data(), code_bytes);
    const CsrGraph graph(nswflat->Graph());
    WRITEVECTOR(graph.Offsets());
    WRITEVECTOR(graph.Edges());
    return;
  }

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

  const IndexHNSWLVQ* hnswlvq = dynamic_cast<const IndexHNSWLVQ*>(index);
  if (hnswlvq) {
    uint32_t h = fourcc("IHNl");
    WRITE1(h);
    write_index_header(*hnswlvq, f);
    write_HNSW(hnswlvq->hnsw, f);
    HYPERVEC_THROW_IF_NOT(hnswlvq->storage != nullptr);
    WriteIndex(hnswlvq->storage, f, 0);
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

  const IndexLVQ* ilvq = dynamic_cast<const IndexLVQ*>(index);
  if (ilvq) {
    uint32_t h = fourcc("ILVQ");
    WRITE1(h);
    write_index_header(*ilvq, f);
    write_lvq(ilvq->lvq, f);
    WRITEVECTOR(ilvq->codes);
    return;
  }

  const IndexIVFFlat* ivfflat = dynamic_cast<const IndexIVFFlat*>(index);
  if (ivfflat) {
    uint32_t h = fourcc("IVFf");
    WRITE1(h);
    write_index_header(*ivfflat, f);
    WRITE1(ivfflat->nlist);
    WRITE1(ivfflat->nprobe);
    WRITEVECTOR(ivfflat->centroids);

    const size_t code_size = static_cast<size_t>(ivfflat->d) * sizeof(float);
    for (size_t list_no = 0; list_no < ivfflat->nlist; list_no++) {
      const size_t sz = ivfflat->invlists->list_size(list_no);
      WRITE1(sz);
      if (sz == 0) {
        continue;
      }
      InvertedLists::ScopedIds ids(ivfflat->invlists, list_no);
      InvertedLists::ScopedCodes codes(ivfflat->invlists, list_no);
      WRITEANDCHECK(ids.get(), sz);
      WRITEANDCHECK(codes.get(), sz * code_size);
    }
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
    int8_t by_residual = ivfpq->by_residual ? 1 : 0;
    int upt = ivfpq->use_precomputed_table;
    WRITE1(by_residual);
    WRITE1(upt);
    write_pq(ivfpq->pq, f);
    WRITEVECTOR(ivfpq->precomputed_table);

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

  const IndexIVFLVQ* ivflvq = dynamic_cast<const IndexIVFLVQ*>(index);
  if (ivflvq) {
    uint32_t h = fourcc("IVLQ");
    WRITE1(h);
    write_index_header(*ivflvq, f);
    WRITE1(ivflvq->nlist);
    WRITE1(ivflvq->nprobe);
    WRITEVECTOR(ivflvq->centroids);
    int8_t by_residual = ivflvq->by_residual ? 1 : 0;
    WRITE1(by_residual);
    write_lvq(ivflvq->lvq, f);
    for (size_t list_no = 0; list_no < ivflvq->nlist; list_no++) {
      const size_t sz = ivflvq->invlists->list_size(list_no);
      WRITE1(sz);
      if (sz == 0) {
        continue;
      }
      InvertedLists::ScopedIds ids(ivflvq->invlists, list_no);
      InvertedLists::ScopedCodes codes(ivflvq->invlists, list_no);
      WRITEANDCHECK(ids.get(), sz);
      WRITEANDCHECK(codes.get(), sz * ivflvq->lvq.code_size);
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

void write_LocalVectorQuantizer(const LocalVectorQuantizer* lvq, IOWriter* f) {
  uint32_t h = fourcc("LvQq");
  WRITE1(h);
  write_lvq(*lvq, f);
}

void write_LocalVectorQuantizer(const LocalVectorQuantizer* lvq,
                                const char* fname) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  write_LocalVectorQuantizer(lvq, f.get());
}

void WriteIndex(const Index* index, FILE* f, int io_flags) {
  FileIOWriter writer(f);
  WriteIndex(index, &writer, io_flags);
}

void WriteIndex(const Index* index, const char* fname, int io_flags) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  WriteIndex(index, f.get(), io_flags);
}

}  // namespace hypervec
