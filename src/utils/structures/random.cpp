/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#include <utils/structures/random.h>

#include <algorithm>

namespace hypervec {

RandomGenerator::RandomGenerator(int64_t seed) : mt((unsigned int)seed) {}

int RandomGenerator::rand_int() {
  return mt() & 0x7fffffff;
}

int64_t RandomGenerator::rand_int64() {
  return int64_t(rand_int()) | int64_t(rand_int()) << 31;
}

int RandomGenerator::rand_int(int max) {
  return mt() % max;
}

float RandomGenerator::rand_float() {
  return mt() / float(mt.max());
}

double RandomGenerator::rand_double() {
  return mt() / double(mt.max());
}

/* Generate a set of random floating point values such that x[i] in [0,1].
 * Uses re-entrant RNG so the work can be split across threads. */
void FloatRand(float* x, size_t n, int64_t seed) {
  // only try to parallelize on large enough arrays
  const size_t nblock = n < 1024 ? 1 : 1024;

  RandomGenerator rng0(seed);
  int a0 = rng0.rand_int(), b0 = rng0.rand_int();

#pragma omp parallel for
  for (int64_t j = 0; j < nblock; j++) {
    RandomGenerator rng(a0 + j * b0);

    const size_t istart = j * n / nblock;
    const size_t iend = (j + 1) * n / nblock;

    for (size_t i = istart; i < iend; i++) {
      x[i] = rng.rand_float();
    }
  }
}

void rand_perm(int* perm, size_t n, int64_t seed) {
  for (size_t i = 0; i < n; i++) {
    perm[i] = i;
  }

  RandomGenerator rng(seed);

  for (size_t i = 0; i + 1 < n; i++) {
    int i2 = i + rng.rand_int(n - i);
    std::swap(perm[i], perm[i2]);
  }
}

}  // namespace hypervec
