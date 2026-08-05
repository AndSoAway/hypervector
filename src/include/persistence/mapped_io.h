/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/structures/maybe_owned_vector.h>
#include <persistence/io.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace hypervec {

// holds a memory-mapped region over a file
struct MmappedFileMappingOwner : public MaybeOwnedVectorOwner {
  explicit MmappedFileMappingOwner(const std::string& filename);
  explicit MmappedFileMappingOwner(FILE* f);
  ~MmappedFileMappingOwner();

  void* data() const;
  size_t size() const;

  struct PImpl;
  std::unique_ptr<PImpl> p_impl;
};

// A deserializer that supports memory-mapped files.
// All de-allocations should happen as soon as the index gets destroyed,
//   after all underlying the MaybeOwnerVector objects are destroyed.
struct MappedFileIOReader : IOReader {
  std::shared_ptr<MmappedFileMappingOwner> mmap_owner;

  size_t pos = 0;

  explicit MappedFileIOReader(
    const std::shared_ptr<MmappedFileMappingOwner>& owner);

  // perform a copy
  size_t operator()(void* ptr, size_t size, size_t nitems) override;
  // perform a quasi-read that returns a mmapped address, owned by mmap_owner,
  //   and updates the position
  size_t mmap(void** ptr, size_t size, size_t nitems);

  int filedescriptor() override;
};

}  // namespace hypervec
