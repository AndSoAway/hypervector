/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// Internal header: shared write helpers for Index::write_body() implementations.
// Not part of the public API. Intentionally not listed in HYPERVEC_HEADERS.

#pragma once

#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <quantization/lvq/lvq.h>
#include <quantization/pq/pq.h>
#include <index/hnsw/hnsw.h>
#include <persistence/index_io.h>

#include <cerrno>
#include <cstring>

namespace hypervec {

static inline void write_pq(const ProductQuantizer& pq, IOWriter* f) {
  WRITE1(pq.d);
  WRITE1(pq.M);
  WRITE1(pq.nbits);
  WRITEVECTOR(pq.centroids);
}

static inline void write_lvq(const LocalVectorQuantizer& lvq, IOWriter* f) {
  WRITE1(lvq.d);
  WRITE1(lvq.nlocal);
  WRITE1(lvq.nbits);
  WRITEVECTOR(lvq.local_centroids);
  WRITEVECTOR(lvq.residual_codebooks);
}

static inline void write_HNSW(const HNSW& hnsw, IOWriter* f) {
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

}  // namespace hypervec
