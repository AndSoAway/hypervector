/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

/* Random generators. Implemented here for speed and to make
 * sequences reproducible.
 */

#pragma once

#include <stdint.h>

#include <random>

namespace hypervec {

/// random generator that can be used in multithreaded contexts
struct RandomGenerator {
  std::mt19937 mt;

  /// random positive integer
  int rand_int();

  /// random int64_t
  int64_t rand_int64();

  /// generate random integer between 0 and max-1
  int rand_int(int max);

  /// between 0 and 1
  float rand_float();

  double rand_double();

  explicit RandomGenerator(int64_t seed = 1234);
};

/* Generate an array of uniform random floats / multi-threaded implementation */
void FloatRand(float* x, size_t n, int64_t seed);

/* random permutation */
void rand_perm(int* perm, size_t n, int64_t seed);

}  // namespace hypervec
