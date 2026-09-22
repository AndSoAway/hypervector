/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#include <utils/log/assert.h>
#include <utils/common/platform_macros.h>
#include <utils/distances/distances.h>
#include <utils/simd/simdlib/simdlib_dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef THE_SIMD_LEVEL
#define THE_SIMD_LEVEL SIMDLevel::NONE
#endif

#include <utils/distances/simd_impl/distances_autovec-inl.h>
#include <utils/distances/simd_impl/distances_simdlib256.h>

namespace hypervec {

/*******
Shared implementations compiled separately for each supported SIMD level.
*/

template <>
void fvec_madd<THE_SIMD_LEVEL>(size_t n, const float* a, float bf,
                               const float* b, float* c) {
  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] + bf * b[i];
  }
}

template <>
void fvec_L2sqr_ny_transposed<THE_SIMD_LEVEL>(float* dis, const float* x,
                                              const float* y,
                                              const float* y_sqlen, size_t d,
                                              size_t d_offset, size_t ny) {
  float x_sqlen = 0;
  for (size_t j = 0; j < d; j++) {
    x_sqlen += x[j] * x[j];
  }

  for (size_t i = 0; i < ny; i++) {
    float dp = 0;
    for (size_t j = 0; j < d; j++) {
      dp += x[j] * y[i + j * d_offset];
    }

    dis[i] = x_sqlen + y_sqlen[i] - 2 * dp;
  }
}

template <>
void fvec_inner_products_ny<THE_SIMD_LEVEL>(float* ip, const float* x,
                                            const float* y, size_t d,
                                            size_t ny) {
  // BLAS sgemv was tried here and was slower than the scalar loop
  // for the typical sizes in this codebase.
  for (size_t i = 0; i < ny; i++) {
    ip[i] = fvec_inner_product<THE_SIMD_LEVEL>(x, y, d);
    y += d;
  }
}

template <>
void fvec_L2sqr_ny<THE_SIMD_LEVEL>(float* dis, const float* x, const float* y,
                                   size_t d, size_t ny) {
  for (size_t i = 0; i < ny; i++) {
    dis[i] = fvec_L2sqr<THE_SIMD_LEVEL>(x, y, d);
    y += d;
  }
}

template <>
size_t fvec_L2sqr_ny_nearest<THE_SIMD_LEVEL>(float* distances_tmp_buffer,
                                             const float* x, const float* y,
                                             size_t d, size_t ny) {
  fvec_L2sqr_ny<THE_SIMD_LEVEL>(distances_tmp_buffer, x, y, d, ny);

  size_t nearest_idx = 0;
  float min_dis = HUGE_VALF;

  for (size_t i = 0; i < ny; i++) {
    if (distances_tmp_buffer[i] < min_dis) {
      min_dis = distances_tmp_buffer[i];
      nearest_idx = i;
    }
  }

  return nearest_idx;
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<THE_SIMD_LEVEL>(
    float* distances_tmp_buffer, const float* x, const float* y,
    const float* y_sqlen, size_t d, size_t d_offset, size_t ny) {
  fvec_L2sqr_ny_transposed<THE_SIMD_LEVEL>(distances_tmp_buffer, x, y, y_sqlen,
                                           d, d_offset, ny);

  size_t nearest_idx = 0;
  float min_dis = HUGE_VALF;

  for (size_t i = 0; i < ny; i++) {
    if (distances_tmp_buffer[i] < min_dis) {
      min_dis = distances_tmp_buffer[i];
      nearest_idx = i;
    }
  }

  return nearest_idx;
}

template <>
int fvec_madd_and_argmin<THE_SIMD_LEVEL>(size_t n, const float* a, float bf,
                                         const float* b, float* c) {
  float vmin = 1e20;
  int imin = -1;

  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] + bf * b[i];
    if (c[i] < vmin) {
      vmin = c[i];
      imin = i;
    }
  }
  return imin;
}

}  // namespace hypervec
