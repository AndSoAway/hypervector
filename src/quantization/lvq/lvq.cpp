/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/lvq/lvq.h>
#include <utils/log/assert.h>
#include <utils/simd/simd_levels.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "quantization/lvq/lvq_distance_kernels.h"

namespace hypervec {
namespace {

unsigned ReadComponent(const uint8_t* packed, size_t bit, int width) {
  unsigned value = static_cast<unsigned>(packed[bit / 8]) >> (bit % 8);
  if (bit % 8 + width > 8) {
    value |= static_cast<unsigned>(packed[bit / 8 + 1]) << (8 - bit % 8);
  }
  return value & ((1u << width) - 1u);
}

void WriteComponent(uint8_t* packed, size_t bit, int width, unsigned value) {
  packed[bit / 8] |= static_cast<uint8_t>(value << (bit % 8));
  if (bit % 8 + width > 8) {
    packed[bit / 8 + 1] |= static_cast<uint8_t>(value >> (8 - bit % 8));
  }
}

}  // namespace

LocalVectorQuantizer::LocalVectorQuantizer(idx_t dimension, int bits)
    : d(dimension), nbits(bits) {
  SetDerivedValues();
}

void LocalVectorQuantizer::SetDerivedValues() {
  HYPERVEC_THROW_IF_NOT_MSG(d > 0, "LVQ: dimension must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(nbits >= 1 && nbits <= 8,
                            "LVQ: nbits must be in [1, 8]");
  const size_t bits = mul_no_overflow(
      static_cast<size_t>(d), static_cast<size_t>(nbits), "LVQ code bits");
  code_size =
      add_no_overflow(2 * sizeof(float),
                      add_no_overflow(bits, size_t{7}, "LVQ rounded bits") / 8,
                      "LVQ code bytes");
  mean.resize(static_cast<size_t>(d));
}

void LocalVectorQuantizer::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(n > 0 && x != nullptr,
                            "LVQ: training requires nonempty vectors");
  std::vector<double> totals(static_cast<size_t>(d), 0.0);
  for (idx_t i = 0; i < n; ++i) {
    for (idx_t j = 0; j < d; ++j) {
      const float value = x[i * d + j];
      HYPERVEC_THROW_IF_NOT_MSG(std::isfinite(value),
                                "LVQ: training vectors must be finite");
      totals[static_cast<size_t>(j)] += value;
    }
  }
  std::vector<float> trained_mean(static_cast<size_t>(d));
  for (idx_t j = 0; j < d; ++j) {
    trained_mean[static_cast<size_t>(j)] =
        static_cast<float>(totals[static_cast<size_t>(j)] / n);
    HYPERVEC_THROW_IF_NOT_MSG(
        std::isfinite(trained_mean[static_cast<size_t>(j)]),
        "LVQ: non-finite trained mean");
  }
  mean = std::move(trained_mean);
  is_trained = true;
}

void LocalVectorQuantizer::ComputeCode(const float* x, uint8_t* code) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr && code != nullptr,
                            "LVQ: encode requires non-null buffers");
  float lower = (std::numeric_limits<float>::infinity)();
  float upper = -(std::numeric_limits<float>::infinity)();
  for (idx_t j = 0; j < d; ++j) {
    const float centered = x[j] - mean[static_cast<size_t>(j)];
    HYPERVEC_THROW_IF_NOT_MSG(std::isfinite(centered),
                              "LVQ: non-finite centered component");
    lower = std::min(lower, centered);
    upper = std::max(upper, centered);
  }
  const float step = (upper - lower) / static_cast<float>((1u << nbits) - 1u);
  HYPERVEC_THROW_IF_NOT_MSG(std::isfinite(step),
                            "LVQ: non-finite quantization step");
  std::memset(code, 0, code_size);
  std::memcpy(code, &lower, sizeof(float));
  std::memcpy(code + sizeof(float), &step, sizeof(float));
  uint8_t* packed = code + 2 * sizeof(float);
  for (idx_t j = 0; j < d; ++j) {
    const float centered = x[j] - mean[static_cast<size_t>(j)];
    const unsigned value =
        step == 0.0f
            ? 0u
            : static_cast<unsigned>(std::min(
                  static_cast<float>((1u << nbits) - 1u),
                  std::max(0.0f, std::round((centered - lower) / step))));
    if (nbits == 8) {
      packed[j] = static_cast<uint8_t>(value);
    } else {
      WriteComponent(packed, static_cast<size_t>(j) * nbits, nbits, value);
    }
  }
}

void LocalVectorQuantizer::ComputeCodes(idx_t n, const float* x,
                                        uint8_t* codes) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
#pragma omp parallel for if (n > 1)
  for (idx_t i = 0; i < n; ++i) {
    ComputeCode(x + i * d, codes + static_cast<size_t>(i) * code_size);
  }
}

void LocalVectorQuantizer::Decode(const uint8_t* code, float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  float lower, step;
  std::memcpy(&lower, code, sizeof(float));
  std::memcpy(&step, code + sizeof(float), sizeof(float));
  const uint8_t* packed = code + 2 * sizeof(float);
  for (idx_t j = 0; j < d; ++j) {
    x[j] = mean[static_cast<size_t>(j)] + lower +
           step * static_cast<float>(
                      nbits == 8
                          ? packed[j]
                          : ReadComponent(
                                packed, static_cast<size_t>(j) * nbits, nbits));
  }
}

void LocalVectorQuantizer::DecodeBatch(idx_t n, const uint8_t* codes,
                                       float* x) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
#pragma omp parallel for if (n > 1)
  for (idx_t i = 0; i < n; ++i) {
    Decode(codes + static_cast<size_t>(i) * code_size, x + i * d);
  }
}

void LocalVectorQuantizer::ComputeDistanceTable(const float* x,
                                                float* query_buffer) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  for (idx_t j = 0; j < d; ++j) {
    query_buffer[j] = x[j] - mean[static_cast<size_t>(j)];
  }
}

namespace {

float ScalarDistance(const LocalVectorQuantizer& lvq, const float* query_buffer,
                     const uint8_t* code) {
  const idx_t d = lvq.d;
  const int nbits = lvq.nbits;
  float lower, step;
  std::memcpy(&lower, code, sizeof(float));
  std::memcpy(&step, code + sizeof(float), sizeof(float));
  float squared = 0.0f;
  const uint8_t* packed = code + 2 * sizeof(float);
  if (nbits == 8) {
    for (idx_t j = 0; j < d; ++j) {
      const float diff =
          query_buffer[j] - lower - step * static_cast<float>(packed[j]);
      squared += diff * diff;
    }
    return squared;
  }
  for (idx_t j = 0; j < d; ++j) {
    const float diff =
        query_buffer[j] - lower -
        step * static_cast<float>(ReadComponent(
                   packed, static_cast<size_t>(j) * nbits, nbits));
    squared += diff * diff;
  }
  return squared;
}

}  // namespace

auto LocalVectorQuantizer::GetDistanceKernel() const -> DistanceKernel {
#ifdef COMPILE_SIMD_AVX2
  if (nbits == 8 && SIMDConfig::get_level() == SIMDLevel::AVX2) {
    return LVQ8DistanceAVX2;
  }
#endif
  return ScalarDistance;
}

float LocalVectorQuantizer::ApplyDistanceTable(const float* query_buffer,
                                               const uint8_t* code) const {
  return GetDistanceKernel()(*this, query_buffer, code);
}

void LocalVectorQuantizer::SearchL2(idx_t nx, const float* x, idx_t ncodes,
                                    const uint8_t* codes, idx_t k,
                                    float* distances, idx_t* labels) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT_MSG(nx >= 0 && ncodes >= 0 && k > 0,
                            "LVQ: invalid search counts");
  (void)mul_no_overflow(static_cast<size_t>(nx), static_cast<size_t>(k),
                        "LVQ output count");
  (void)mul_no_overflow(static_cast<size_t>(ncodes), code_size,
                        "LVQ input code bytes");
  if (nx == 0) {
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr && distances != nullptr &&
                                labels != nullptr &&
                                (ncodes == 0 || codes != nullptr),
                            "LVQ: null search buffer");
  const auto distance_kernel = GetDistanceKernel();
#pragma omp parallel
  {
    std::vector<float> query(static_cast<size_t>(d));
#pragma omp for
    for (idx_t qi = 0; qi < nx; ++qi) {
      ComputeDistanceTable(x + qi * d, query.data());
      float* heap_dis = distances + qi * k;
      idx_t* heap_ids = labels + qi * k;
      heap_heapify<CMax<float, idx_t>>(k, heap_dis, heap_ids);
      float threshold = heap_dis[0];
      for (idx_t j = 0; j < ncodes; ++j) {
        const float dis = distance_kernel(
            *this, query.data(), codes + static_cast<size_t>(j) * code_size);
        if (CMax<float, idx_t>::cmp(threshold, dis)) {
          heap_replace_top<CMax<float, idx_t>>(k, heap_dis, heap_ids, dis, j);
          threshold = heap_dis[0];
        }
      }
      heap_reorder<CMax<float, idx_t>>(k, heap_dis, heap_ids);
    }
  }
}

}  // namespace hypervec
