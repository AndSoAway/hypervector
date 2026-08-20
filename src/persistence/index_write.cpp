/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * HNSW-only index write implementation
 */

#include <utils/log/assert.h>
#include <persistence/index_io.h>
#include <persistence/index_write_utils.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>

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

void WriteIndex(const Index* index, IOWriter* f, int io_flags) {
  (void)io_flags;
  uint32_t h = index->fourcc();
  WRITE1(h);
  write_index_header(*index, f);
  index->write_body(f);
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
