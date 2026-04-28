/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <stdint.h>

#include <vector>

namespace hypervec {

/****************************************************************
 * BufferList - List of temporary buffers used to store results
 * before they are copied to the RangeSearchResult object.
 *****************************************************************/

struct BufferList {
  // buffer sizes in # entries
  size_t buffer_size;

  struct Buffer {
    idx_t* ids;
    float* dis;
  };

  std::vector<Buffer> buffers;
  size_t wp;  ///< write pointer in the last buffer.

  explicit BufferList(size_t buffer_size);

  ~BufferList();

  /// create a new buffer
  void AppendBuffer();

  /// Add one result, possibly appending a new buffer if needed
  void Add(idx_t id, float dis);

  /// copy elements ofs:ofs+n-1 seen as linear data in the buffers to
  /// tables dest_ids, dest_dis
  void CopyRange(size_t ofs, size_t n, idx_t* dest_ids, float* dest_dis);
};

}  // namespace hypervec