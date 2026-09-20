/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <memory>

namespace hypervec {

struct Index;
struct IOReader;
struct IOWriter;
class IndexIORegistry;

namespace persistence_internal {

IndexIORegistry& GetBuiltinIndexIORegistry();

void WriteFlatL2Payload(const Index& index, IOWriter* writer, int io_flags);
void WriteFlatIPPayload(const Index& index, IOWriter* writer, int io_flags);
void WritePQPayload(const Index& index, IOWriter* writer, int io_flags);
void WriteLVQPayload(const Index& index, IOWriter* writer, int io_flags);
void WriteIVFFlatPayload(const Index& index, IOWriter* writer, int io_flags);
void WriteIVFPQPayload(const Index& index, IOWriter* writer, int io_flags);
void WriteIVFLVQPayload(const Index& index, IOWriter* writer, int io_flags);
void WriteIVFRaBitQPayload(const Index& index, IOWriter* writer, int io_flags);
void ValidateIVFRaBitQForWrite(const Index& index, int io_flags);
void WriteIDMapPayload(const Index& index, IOWriter* writer, int io_flags);
void ValidateIDMapForWrite(const Index& index, int io_flags);
void WritePreTransformPayload(const Index& index, IOWriter* writer,
                              int io_flags);
void ValidatePreTransformForWrite(const Index& index, int io_flags);
void WriteLSHPayload(const Index& index, IOWriter* writer, int io_flags);
void ValidateLSHForWrite(const Index& index, int io_flags);
void WriteNSWFlatPayload(const Index& index, IOWriter* writer, int io_flags);
void ValidateNSWFlatForWrite(const Index& index, int io_flags);
void WriteNSGFlatPayload(const Index& index, IOWriter* writer, int io_flags);
void ValidateNSGFlatForWrite(const Index& index, int io_flags);

std::unique_ptr<Index> ReadFlatL2Payload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadFlatIPPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadPQPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadLVQPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadIVFFlatPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadIVFPQPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadIVFLVQPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadIVFRaBitQPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadIDMapPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadPreTransformPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadLSHPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadNSWFlatPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadNSGFlatPayload(IOReader* reader, int io_flags);

}  // namespace persistence_internal

}  // namespace hypervec
