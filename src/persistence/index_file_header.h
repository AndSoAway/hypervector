/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <persistence/index_io.h>

#include <cstdint>

namespace hypervec {

struct Index;
struct IOReader;
struct IOWriter;

namespace persistence_detail {

uint32_t IndexFileEnvelopeMagic();

void WriteIndexFileHeader(const IndexFileMetadata& metadata, IOWriter* writer);

/** Read a version-2 header after its magic has already been consumed. */
IndexFileMetadata ReadIndexFileHeader(IOReader* reader);

/** Restore metadata-only state and verify metadata against the payload. */
void ApplyAndValidateIndexFileMetadata(Index* index,
                                       const IndexFileMetadata& metadata);

}  // namespace persistence_detail
}  // namespace hypervec
