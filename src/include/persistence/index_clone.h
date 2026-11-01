/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

// I/O code for indexes

#pragma once

namespace hypervec {

struct Index;
struct VectorTransform;
struct IndexHNSW;

/** Return a fully independent deep copy of an index.
 *
 * The clone preserves trained state, stored vectors, graph/quantizer state,
 * wrappers, and runtime defaults supported by the persistence protocol. The
 * caller owns the returned pointer.
 */
Index* clone_index(const Index*);

/** Type-preserving deep-copy entry point for HNSW indexes. */
IndexHNSW* clone_IndexHNSW(const IndexHNSW* index);

}  // namespace hypervec
