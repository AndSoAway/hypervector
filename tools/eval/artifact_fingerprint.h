/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>

namespace hypervec::eval_cli {

/** Stable identity and byte size of one evaluation artifact. */
struct ArtifactFingerprint {
  uint64_t size_bytes = 0;
  std::string sha256;
};

/** Read a complete regular file and compute its SHA-256 fingerprint. */
ArtifactFingerprint FingerprintFile(const std::filesystem::path& path);

/** Write one fingerprint and its normalized path as a JSON object. */
void WriteArtifactJson(std::ostream& output, const std::filesystem::path& path,
                       const ArtifactFingerprint& fingerprint,
                       std::string_view indentation);

}  // namespace hypervec::eval_cli
