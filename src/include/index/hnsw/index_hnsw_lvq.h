/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/hnsw/index_hnsw.h>
#include <quantization/lvq/index_lvq.h>

namespace hypervec {

struct IndexHNSWLVQ : IndexHNSW {
  Index* raw_storage = nullptr;

  IndexHNSWLVQ();
  IndexHNSWLVQ(int d, int nlocal, int nbits, int M_hnsw,
               MetricType metric = kMetricL2);
  ~IndexHNSWLVQ() override;

  void Train(idx_t n, const float* x) override;
  void Add(idx_t n, const float* x) override;
  void Reset() override;
  void Freeze();
  size_t SaCodeSize() const override;
  void SaEncode(idx_t n, const float* x, uint8_t* bytes) const override;
  void SaDecode(idx_t n, const uint8_t* bytes, float* x) const override;
  void Search1(const float* x, ResultHandler& handler,
               SearchParameters* params = nullptr) const override;
  void RangeSearch(idx_t n, const float* x, float radius,
                   RangeSearchResult* result,
                   const SearchParameters* params = nullptr) const override;
};

}  // namespace hypervec
