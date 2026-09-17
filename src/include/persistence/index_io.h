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

#include <factory/index_config.h>

#include <cstdint>
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
struct ProductQuantizer;
struct LocalVectorQuantizer;
struct IOReader;
struct IOWriter;

const int IO_FLAG_MMAP_IFC = 1 << 9;

inline constexpr uint16_t kIndexFileFormatMajor = 2;
inline constexpr uint16_t kIndexFileFormatMinor = 0;
inline constexpr uint32_t kIndexAlgorithmVersion = 1;

/** Version and normalized construction metadata carried by an index file.
 *
 * Version-2 files store this metadata in a checksummed envelope before the
 * existing algorithm payload. Legacy files have format_major=1,
 * algorithm_version=0, and metadata inferred from the loaded payload.
 */
struct IndexFileMetadata {
  uint16_t format_major = kIndexFileFormatMajor;
  uint16_t format_minor = kIndexFileFormatMinor;
  uint32_t algorithm_version = kIndexAlgorithmVersion;
  IndexConfig config;
  bool is_legacy = false;
};

/** Describe the normalized configuration that would be persisted for index. */
IndexFileMetadata GetIndexFileMetadata(const Index& index);

void WriteIndex(const Index* idx, const char* fname, int io_flags = 0);
void WriteIndex(const Index* idx, FILE* f, int io_flags = 0);
void WriteIndex(const Index* idx, IOWriter* writer, int io_flags = 0);

Index* ReadIndex(const char* fname, int io_flags = 0);
Index* ReadIndex(FILE* f, int io_flags = 0);
Index* ReadIndex(IOReader* reader, int io_flags = 0);

std::unique_ptr<Index> ReadIndexUp(const char* fname, int io_flags = 0);
std::unique_ptr<Index> ReadIndexUp(FILE* f, int io_flags = 0);
std::unique_ptr<Index> ReadIndexUp(IOReader* reader, int io_flags = 0);

/** Read an index and return its file metadata through metadata when non-null.
 */
std::unique_ptr<Index> ReadIndexUp(const char* fname,
                                   IndexFileMetadata* metadata,
                                   int io_flags = 0);
std::unique_ptr<Index> ReadIndexUp(FILE* f, IndexFileMetadata* metadata,
                                   int io_flags = 0);
std::unique_ptr<Index> ReadIndexUp(IOReader* reader,
                                   IndexFileMetadata* metadata,
                                   int io_flags = 0);

ProductQuantizer* read_ProductQuantizer(const char* fname);
ProductQuantizer* read_ProductQuantizer(IOReader* reader);
std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(const char* fname);
std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(IOReader* reader);
void write_ProductQuantizer(const ProductQuantizer* pq, const char* fname);
void write_ProductQuantizer(const ProductQuantizer* pq, IOWriter* f);

LocalVectorQuantizer* read_LocalVectorQuantizer(const char* fname);
LocalVectorQuantizer* read_LocalVectorQuantizer(IOReader* reader);
std::unique_ptr<LocalVectorQuantizer> read_LocalVectorQuantizer_up(
    const char* fname);
std::unique_ptr<LocalVectorQuantizer> read_LocalVectorQuantizer_up(
    IOReader* reader);
void write_LocalVectorQuantizer(const LocalVectorQuantizer* lvq,
                                const char* fname);
void write_LocalVectorQuantizer(const LocalVectorQuantizer* lvq, IOWriter* f);

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
