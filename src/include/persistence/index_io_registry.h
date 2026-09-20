/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

namespace hypervec {

struct IOReader;
struct IOWriter;

/** Stable binary-format metadata for one persisted index type.
 *
 * write_tag is emitted for new files. read_tags contains compatible current or
 * legacy tags; Register() always adds write_tag if the caller omitted it.
 */
struct IndexIODescriptor {
  std::string name;
  uint32_t write_tag = 0;
  std::vector<uint32_t> read_tags;
};

/** Write or read an index payload after the fourcc tag. */
using IndexPayloadWriter = std::function<void(const Index&, IOWriter*, int)>;
using IndexPayloadResult = std::unique_ptr<Index>;
using IndexPayloadReader = std::function<IndexPayloadResult(IOReader*, int)>;

/** Thread-safe pairing of an exact C++ index type with its binary codec.
 *
 * The registry owns dispatch and the fourcc framing. Payload callbacks own the
 * type-specific format. Exact dynamic types are used intentionally so a base
 * codec cannot silently serialize an unknown derived class.
 */
class IndexIORegistry {
 public:
  IndexIORegistry();
  ~IndexIORegistry();

  IndexIORegistry(const IndexIORegistry&) = delete;
  IndexIORegistry& operator=(const IndexIORegistry&) = delete;

  void Register(IndexIODescriptor descriptor, std::type_index index_type,
                IndexPayloadWriter writer, IndexPayloadReader reader);

  bool Contains(std::type_index index_type) const;
  bool Contains(uint32_t read_tag) const;
  std::vector<IndexIODescriptor> List() const;

  /** Write the canonical tag followed by the registered payload. */
  void Write(const Index& index, IOWriter* writer, int io_flags = 0) const;

  /** Read a tag and its registered payload. */
  std::unique_ptr<Index> Read(IOReader* reader, int io_flags = 0) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hypervec
