/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <immintrin.h>
#include <quantization/lvq/lvq.h>

#include <cstddef>
#include <cstring>

#include "quantization/lvq/lvq_distance_kernels.h"

namespace hypervec {

float LVQ8DistanceAVX2(const LocalVectorQuantizer& lvq, const float* query,
                       const uint8_t* code) {
  float lower, step;
  std::memcpy(&lower, code, sizeof(float));
  std::memcpy(&step, code + sizeof(float), sizeof(float));
  const uint8_t* packed = code + 2 * sizeof(float);
  const size_t d = static_cast<size_t>(lvq.d);
  const __m256 lower8 = _mm256_set1_ps(lower);
  const __m256 step8 = _mm256_set1_ps(step);
  const auto difference = [&](size_t j) {
    // Exactly eight bytes, including for the last complete vector block.
    const __m128i bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + j));
    const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
    return _mm256_sub_ps(_mm256_sub_ps(_mm256_loadu_ps(query + j), lower8),
                         _mm256_mul_ps(step8, values));
  };
  __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps();
  size_t j = 0;
  // Four independent accumulators avoid a long dependent FMA chain.
  for (; j + 32 <= d; j += 32) {
    const __m256 a = difference(j), b = difference(j + 8);
    const __m256 c = difference(j + 16), e = difference(j + 24);
    sum0 = _mm256_fmadd_ps(a, a, sum0);
    sum1 = _mm256_fmadd_ps(b, b, sum1);
    sum2 = _mm256_fmadd_ps(c, c, sum2);
    sum3 = _mm256_fmadd_ps(e, e, sum3);
  }
  sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
  for (; j + 8 <= d; j += 8) {
    const __m256 diff = difference(j);
    sum0 = _mm256_fmadd_ps(diff, diff, sum0);
  }
  __m128 sum =
      _mm_add_ps(_mm256_castps256_ps128(sum0), _mm256_extractf128_ps(sum0, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float squared = _mm_cvtss_f32(sum);
  for (; j < d; ++j) {
    const float diff = query[j] - lower - step * static_cast<float>(packed[j]);
    squared += diff * diff;
  }
  return squared;
}

}  // namespace hypervec
