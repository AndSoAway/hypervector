/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>
#include <index/index.h>
#include <utils/common/range_search_result.h>
#include <utils/distances/distances.h>

#include <cstring>

namespace hypervec {

Index::~Index() = default;

void Index::Train(idx_t /*n*/, const float* /*x*/) {
  // does nothing by default
}

void Index::Train(idx_t /*n*/, const float* /*x*/, idx_t /*n_train_q*/,
                  const float* /*xq_train*/) {
  // does nothing by default
}

void Index::RangeSearch(idx_t, const float*, float, RangeSearchResult*,
                         const SearchParameters* /*params*/) const {
  HYPERVEC_THROW_MSG("range Search not implemented");
}

void Index::Assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
  std::vector<float> distances(n * k);
  Search(n, x, k, distances.data(), labels);
}

void Index::AddWithIds(idx_t /*n*/, const float* /*x*/,
                         const idx_t* /*xids*/) {
  HYPERVEC_THROW_MSG("AddWithIds not implemented for this type of index");
}

size_t Index::RemoveIds(const IDSelector& /*sel*/) {
  HYPERVEC_THROW_MSG("RemoveIds not implemented for this type of index");
  return -1;
}

void Index::Reconstruct(idx_t, float*) const {
  HYPERVEC_THROW_MSG("Reconstruct not implemented for this type of index");
}

void Index::ReconstructBatch(idx_t n, const idx_t* keys, float* recons) const {
  std::mutex exception_mutex;
  std::string exception_string;
#pragma omp parallel for if (n > 1000)
  for (idx_t i = 0; i < n; i++) {
    try {
      Reconstruct(keys[i], &recons[i * d]);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(exception_mutex);
      exception_string = e.what();
    }
  }
  if (!exception_string.empty()) {
    HYPERVEC_THROW_MSG(exception_string.c_str());
  }
}

void Index::ReconstructN(idx_t i0, idx_t ni, float* recons) const {
#pragma omp parallel for if (ni > 1000)
  for (idx_t i = 0; i < ni; i++) {
    Reconstruct(i0 + i, recons + i * d);
  }
}

void Index::SearchAndReconstruct(idx_t n, const float* x, idx_t k,
                                   float* distances, idx_t* labels,
                                   float* recons,
                                   const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(k > 0);

  Search(n, x, k, distances, labels, params);
  for (idx_t i = 0; i < n; ++i) {
    for (idx_t j = 0; j < k; ++j) {
      idx_t ij = i * k + j;
      idx_t key = labels[ij];
      float* reconstructed = recons + ij * d;
      if (key < 0) {
        // Fill with NaNs
        memset(reconstructed, -1, sizeof(*reconstructed) * d);
      } else {
        Reconstruct(key, reconstructed);
      }
    }
  }
}

void Index::SearchSubset(idx_t /*n*/, const float* /*x*/, idx_t /*k_base*/,
                          const idx_t* /*base_labels*/, idx_t /*k*/,
                          float* /*distances*/, idx_t* /*labels*/) const {
  HYPERVEC_THROW_MSG("SearchSubset not implemented for this type of index");
}

void Index::Search1(const float*, ResultHandler&, SearchParameters*) const {
  HYPERVEC_THROW_MSG("Search1 not implemented for this type of index");
}

void Index::ComputeResidual(const float* x, float* residual, idx_t key) const {
  Reconstruct(key, residual);
  for (size_t i = 0; i < d; i++) {
    residual[i] = x[i] - residual[i];
  }
}

void Index::ComputeResidualN(idx_t n, const float* xs, float* residuals,
                               const idx_t* keys) const {
#pragma omp parallel for
  for (idx_t i = 0; i < n; ++i) {
    ComputeResidual(&xs[i * d], &residuals[i * d], keys[i]);
  }
}

size_t Index::SaCodeSize() const {
  HYPERVEC_THROW_MSG("standalone codec not implemented for this type of index");
}

void Index::SaEncode(idx_t, const float*, uint8_t*) const {
  HYPERVEC_THROW_MSG("standalone codec not implemented for this type of index");
}

void Index::SaDecode(idx_t, const uint8_t*, float*) const {
  HYPERVEC_THROW_MSG("standalone codec not implemented for this type of index");
}

void Index::AddSaCodes(idx_t, const uint8_t*, const idx_t*) {
  HYPERVEC_THROW_MSG("AddSaCodes not implemented for this type of index");
}

namespace {

// storage that explicitly reconstructs vectors before computing distances
struct GenericDistanceComputer : DistanceComputer {
  size_t d;
  const Index& storage;
  std::vector<float> buf;
  const float* q;

  explicit GenericDistanceComputer(const Index& storage) : storage(storage) {
    d = storage.d;
    buf.resize(d * 2);
  }

  float operator()(idx_t i) override {
    storage.Reconstruct(i, buf.data());
    return fvec_L2sqr(q, buf.data(), d);
  }

  float symmetric_dis(idx_t i, idx_t j) override {
    storage.Reconstruct(i, buf.data());
    storage.Reconstruct(j, buf.data() + d);
    return fvec_L2sqr(buf.data() + d, buf.data(), d);
  }

  void SetQuery(const float* x) override {
    q = x;
  }
};

}  // namespace

DistanceComputer* Index::GetDistanceComputer() const {
  if (metric_type == kMetricL2) {
    return new GenericDistanceComputer(*this);
  } else {
    HYPERVEC_THROW_MSG("GetDistanceComputer() not implemented");
  }
}

void Index::MergeFrom(Index& /* otherIndex */, idx_t /* add_id */) {
  HYPERVEC_THROW_MSG("MergeFrom() not implemented");
}

void Index::CheckCompatibleForMerge(const Index& /* otherIndex */) const {
  HYPERVEC_THROW_MSG("CheckCompatibleForMerge() not implemented");
}

}  // namespace hypervec
