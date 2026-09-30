/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace hypervec::eval_cli {

/** Flat "key=value" record naming the inputs an artifact came from. */
using ProvenanceRecord = std::unordered_map<std::string, std::string>;
/** Write a record, keys sorted for byte-stable output. */
void WriteProvenance(const std::filesystem::path& path,
                     const ProvenanceRecord& record);
/** Read a record; nullopt when the file cannot be opened. */
std::optional<ProvenanceRecord> ReadProvenance(
    const std::filesystem::path& path);

}  // namespace hypervec::eval_cli
