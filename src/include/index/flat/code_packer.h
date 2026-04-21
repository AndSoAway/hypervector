/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstddef>

namespace hypervec {

/// Code packer interface for encoding/decoding codes
struct CodePacker {
  size_t code_size;

  virtual ~CodePacker() = default;

  virtual void pack(const idx_t* ids, const uint8_t* code,
                    uint8_t* out) const = 0;
  virtual void unpack(const uint8_t* in, idx_t* ids, uint8_t* code) const = 0;
};

/// Flat code packer - simple copy
struct CodePackerFlat : CodePacker {
  explicit CodePackerFlat(size_t code_size) : CodePacker() {
    this->code_size = code_size;
  }

  void pack(const idx_t* ids, const uint8_t* code,
            uint8_t* out) const override {
    memcpy(out, code, code_size);
  }

  void unpack(const uint8_t* in, idx_t* ids, uint8_t* code) const override {
    memcpy(code, in, code_size);
  }
};

}  // namespace hypervec