/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

// Included only by the isolated AVX-512 translation unit, never by callers.
#include <immintrin.h>
#include <utils/distances/distances.h>

namespace hypervec {
namespace avx512_detail {

template <bool L2>
__m512 AccumulateFP32(__m512 x, __m512 y, __m512 sum) {
  if constexpr (L2) {
    const __m512 diff = _mm512_sub_ps(x, y);
    return _mm512_fmadd_ps(diff, diff, sum);
  } else {
    return _mm512_fmadd_ps(x, y, sum);
  }
}

template <bool L2>
float DistanceFP32(const float* x, const float* y, size_t d) {
  // Short vectors (including typical PQ subvectors) retain the AVX2 kernel.
  if (d < 32) {
    if constexpr (L2) return fvec_L2sqr<SIMDLevel::AVX2>(x, y, d);
    return fvec_inner_product<SIMDLevel::AVX2>(x, y, d);
  }
  __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
  __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 64 <= d; i += 64) {
    s0 = AccumulateFP32<L2>(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i), s0);
    s1 = AccumulateFP32<L2>(_mm512_loadu_ps(x + i + 16),
                            _mm512_loadu_ps(y + i + 16), s1);
    s2 = AccumulateFP32<L2>(_mm512_loadu_ps(x + i + 32),
                            _mm512_loadu_ps(y + i + 32), s2);
    s3 = AccumulateFP32<L2>(_mm512_loadu_ps(x + i + 48),
                            _mm512_loadu_ps(y + i + 48), s3);
  }
  s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
  for (; i + 16 <= d; i += 16) {
    s0 = AccumulateFP32<L2>(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i), s0);
  }
  if (i < d) {
    const auto mask = static_cast<__mmask16>((1U << (d - i)) - 1U);
    s0 = AccumulateFP32<L2>(_mm512_maskz_loadu_ps(mask, x + i),
                            _mm512_maskz_loadu_ps(mask, y + i), s0);
  }
  return _mm512_reduce_add_ps(s0);
}

template <bool L2>
void DistanceBatchFP32(const float* x, const float* y0, const float* y1,
                       const float* y2, const float* y3, size_t d, float& dis0,
                       float& dis1, float& dis2, float& dis3) {
  if (d < 32) {
    if constexpr (L2) {
      fvec_L2sqr_batch_4<SIMDLevel::AVX2>(x, y0, y1, y2, y3, d, dis0, dis1,
                                          dis2, dis3);
    } else {
      fvec_inner_product_batch_4<SIMDLevel::AVX2>(x, y0, y1, y2, y3, d, dis0,
                                                  dis1, dis2, dis3);
    }
    return;
  }
  __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
  __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    const __m512 q = _mm512_loadu_ps(x + i);
    s0 = AccumulateFP32<L2>(q, _mm512_loadu_ps(y0 + i), s0);
    s1 = AccumulateFP32<L2>(q, _mm512_loadu_ps(y1 + i), s1);
    s2 = AccumulateFP32<L2>(q, _mm512_loadu_ps(y2 + i), s2);
    s3 = AccumulateFP32<L2>(q, _mm512_loadu_ps(y3 + i), s3);
  }
  if (i < d) {
    const auto mask = static_cast<__mmask16>((1U << (d - i)) - 1U);
    const __m512 q = _mm512_maskz_loadu_ps(mask, x + i);
    s0 = AccumulateFP32<L2>(q, _mm512_maskz_loadu_ps(mask, y0 + i), s0);
    s1 = AccumulateFP32<L2>(q, _mm512_maskz_loadu_ps(mask, y1 + i), s1);
    s2 = AccumulateFP32<L2>(q, _mm512_maskz_loadu_ps(mask, y2 + i), s2);
    s3 = AccumulateFP32<L2>(q, _mm512_maskz_loadu_ps(mask, y3 + i), s3);
  }
  dis0 = _mm512_reduce_add_ps(s0);
  dis1 = _mm512_reduce_add_ps(s1);
  dis2 = _mm512_reduce_add_ps(s2);
  dis3 = _mm512_reduce_add_ps(s3);
}

}  // namespace avx512_detail

template <>
float fvec_L2sqr<SIMDLevel::AVX512>(const float* x, const float* y, size_t d) {
  return avx512_detail::DistanceFP32<true>(x, y, d);
}

template <>
float fvec_inner_product<SIMDLevel::AVX512>(const float* x, const float* y,
                                            size_t d) {
  return avx512_detail::DistanceFP32<false>(x, y, d);
}

template <>
float fvec_norm_L2sqr<SIMDLevel::AVX512>(const float* x, size_t d) {
  return avx512_detail::DistanceFP32<false>(x, x, d);
}

template <>
void fvec_L2sqr_batch_4<SIMDLevel::AVX512>(const float* x, const float* y0,
                                           const float* y1, const float* y2,
                                           const float* y3, size_t d,
                                           float& dis0, float& dis1,
                                           float& dis2, float& dis3) {
  avx512_detail::DistanceBatchFP32<true>(x, y0, y1, y2, y3, d, dis0, dis1, dis2,
                                         dis3);
}

template <>
void fvec_inner_product_batch_4<SIMDLevel::AVX512>(
    const float* x, const float* y0, const float* y1, const float* y2,
    const float* y3, size_t d, float& dis0, float& dis1, float& dis2,
    float& dis3) {
  avx512_detail::DistanceBatchFP32<false>(x, y0, y1, y2, y3, d, dis0, dis1,
                                          dis2, dis3);
}

}  // namespace hypervec
