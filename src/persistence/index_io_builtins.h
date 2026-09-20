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

std::unique_ptr<Index> ReadFlatL2Payload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadFlatIPPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadPQPayload(IOReader* reader, int io_flags);
std::unique_ptr<Index> ReadLVQPayload(IOReader* reader, int io_flags);

}  // namespace persistence_internal

}  // namespace hypervec
