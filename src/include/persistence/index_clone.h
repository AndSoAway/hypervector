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

/* cloning functions */
Index* clone_index(const Index*);

IndexHNSW* clone_IndexHNSW(const IndexHNSW* index);

}  // namespace hypervec
