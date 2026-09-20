/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/hnsw/index_hnsw.h>
#include <persistence/index_clone.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <utils/log/assert.h>

#include <memory>
#include <utility>

namespace hypervec {
namespace {

std::unique_ptr<Index> CloneThroughPersistence(const Index* index) {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "clone_index: index must not be null");

  VectorIOWriter writer;
  WriteIndex(index, &writer);

  VectorIOReader reader;
  reader.data = std::move(writer.data);
  return ReadIndexUp(&reader);
}

}  // namespace

IndexHNSW* clone_IndexHNSW(const IndexHNSW* index) {
  std::unique_ptr<Index> clone = CloneThroughPersistence(index);
  auto* hnsw_clone = dynamic_cast<IndexHNSW*>(clone.get());
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw_clone != nullptr,
      "clone_IndexHNSW: persistence returned a non-HNSW index");
  clone.release();
  return hnsw_clone;
}

Index* clone_index(const Index* index) {
  return CloneThroughPersistence(index).release();
}

}  // namespace hypervec
