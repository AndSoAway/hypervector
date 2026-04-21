/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// I/O code for indexes

#ifndef HYPERVEC_INDEX_IO_H
#define HYPERVEC_INDEX_IO_H

#include <cstdio>
#include <memory>

/** I/O functions can read/write to a filename, a file handle or to an
 * object that abstracts the medium.
 *
 * The read functions come in two forms:
 * - read_*_up() returns a std::unique_ptr that owns the result.
 * - read_*() returns a raw pointer for backward compatibility.
 *   The caller is responsible for deleting the returned object.
 *
 * All references within these objects are owned by the object.
 */

namespace hypervec {

struct Index;
struct IndexBinary;
struct VectorTransform;
struct ProductQuantizer;
struct IOReader;
struct IOWriter;
struct InvertedLists;

/// skip the storage for graph-based indexes
const int IO_FLAG_SKIP_STORAGE = 1;

void WriteIndex(const Index* idx, const char* fname, int io_flags = 0);
void WriteIndex(const Index* idx, FILE* f, int io_flags = 0);
void WriteIndex(const Index* idx, IOWriter* writer, int io_flags = 0);

void WriteIndexBinary(const IndexBinary* idx, const char* fname);
void WriteIndexBinary(const IndexBinary* idx, FILE* f);
void WriteIndexBinary(const IndexBinary* idx, IOWriter* writer);

// The ReadIndex flags are implemented only for a subset of index types.
const int IO_FLAG_READ_ONLY = 2;
// strip directory component from ondisk filename, and assume it's in
// the same directory as the index file
const int IO_FLAG_ONDISK_SAME_DIR = 4;
// don't load IVF data to RAM, only list sizes
const int IO_FLAG_SKIP_IVF_DATA = 8;
// don't initialize precomputed table after loading
const int IO_FLAG_SKIP_PRECOMPUTE_TABLE = 16;
// don't compute the sdc table for PQ-based indices
// this will prevent distances from being computed
// between elements in the index. For indices like HNSWPQ,
// this will prevent graph building because sdc
// computations are required to construct the graph
const int IO_FLAG_PQ_SKIP_SDC_TABLE = 32;
// try to memmap data (useful to load an ArrayInvertedLists as an
// OnDiskInvertedLists)
const int IO_FLAG_MMAP = IO_FLAG_SKIP_IVF_DATA | 0x646f0000;
// mmap that handles codes for IndexFlatCodes-derived indices and HNSW.
// this is a temporary solution, it is expected to be merged with IO_FLAG_MMAP
//   after OnDiskInvertedLists get properly updated.
const int IO_FLAG_MMAP_IFC = 1 << 9;

Index* ReadIndex(const char* fname, int io_flags = 0);
Index* ReadIndex(FILE* f, int io_flags = 0);
Index* ReadIndex(IOReader* reader, int io_flags = 0);

std::unique_ptr<Index> ReadIndexUp(const char* fname, int io_flags = 0);
std::unique_ptr<Index> ReadIndexUp(FILE* f, int io_flags = 0);
std::unique_ptr<Index> ReadIndexUp(IOReader* reader, int io_flags = 0);

IndexBinary* ReadIndexBinary(const char* fname, int io_flags = 0);
IndexBinary* ReadIndexBinary(FILE* f, int io_flags = 0);
IndexBinary* ReadIndexBinary(IOReader* reader, int io_flags = 0);

std::unique_ptr<IndexBinary> ReadIndexBinaryUp(const char* fname,
                                                  int io_flags = 0);
std::unique_ptr<IndexBinary> ReadIndexBinaryUp(FILE* f, int io_flags = 0);
std::unique_ptr<IndexBinary> ReadIndexBinaryUp(IOReader* reader,
                                                  int io_flags = 0);

void write_VectorTransform(const VectorTransform* vt, const char* fname);
void write_VectorTransform(const VectorTransform* vt, IOWriter* f);

VectorTransform* read_VectorTransform(const char* fname);
VectorTransform* read_VectorTransform(IOReader* f);

std::unique_ptr<VectorTransform> read_VectorTransform_up(const char* fname);
std::unique_ptr<VectorTransform> read_VectorTransform_up(IOReader* f);

ProductQuantizer* read_ProductQuantizer(const char* fname);
ProductQuantizer* read_ProductQuantizer(IOReader* reader);

std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(const char* fname);
std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(IOReader* reader);

void write_ProductQuantizer(const ProductQuantizer* pq, const char* fname);
void write_ProductQuantizer(const ProductQuantizer* pq, IOWriter* f);

void write_InvertedLists(const InvertedLists* ils, IOWriter* f);
InvertedLists* read_InvertedLists(IOReader* reader, int io_flags = 0);

std::unique_ptr<InvertedLists> read_InvertedLists_up(IOReader* reader,
                                                     int io_flags = 0);

// Returns the current deserialization loop limit.
// When nonzero, deserialization rejects loop-driving fields (nlist,
// nsplits, VT chain length, nhash, etc.) that exceed this value.
// Default: 0 (no limit).
size_t get_deserialization_loop_limit();

// Sets the deserialization loop limit.
// NOT thread-safe: set before any concurrent deserialization calls
// and do not modify while deserialization is in progress on other threads.
void set_deserialization_loop_limit(size_t value);

// Returns the maximum number of bytes that a single READVECTOR call
// may allocate.  Default: 1 TB (1 << 40).
size_t get_deserialization_vector_byte_limit();

// Sets the per-vector byte limit for deserialization.
// NOT thread-safe: set before any concurrent deserialization calls
// and do not modify while deserialization is in progress on other threads.
void set_deserialization_vector_byte_limit(size_t value);

}  // namespace hypervec

#endif
