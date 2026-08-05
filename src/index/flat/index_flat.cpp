/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#include <utils/log/assert.h>
#include <utils/simd/simd_dispatch.h>
#include <index/flat/index_flat.h>
#include <omp.h>
#include <utils/common/range_search_result.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distances.h>
#include <utils/distances/extra_distances.h>
#include <utils/structures/heap.h>
#include <utils/structures/prefetch.h>
#include <utils/structures/sorting.h>

#include <cstring>

namespace hypervec {

IndexFlat::IndexFlat(idx_t d, MetricType metric)
  : IndexFlatCodes(sizeof(float) * d, d, metric) {}

void IndexFlat::Search(idx_t n, const float* x, idx_t k, float* distances,
                       idx_t* labels, const SearchParameters* params) const {
  IDSelector* sel = params ? params->sel : nullptr;
  HYPERVEC_THROW_IF_NOT(k > 0);

  // we see the distances and labels as heaps
  if (metric_type == kMetricInnerProduct) {
    float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
    knn_inner_product(x, GetXb(), d, n, n_total, &res, sel);
  } else if (metric_type == kMetricL2) {
    float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
    knn_L2sqr(x, GetXb(), d, n, n_total, &res, nullptr, sel);
  } else {
    knn_extra_metrics(x, GetXb(), d, n, n_total, metric_type, metric_arg, k,
                      distances, labels, sel);
  }
}

void IndexFlat::RangeSearch(idx_t n, const float* x, float radius,
                             RangeSearchResult* result,
                             const SearchParameters* params) const {
  IDSelector* sel = params ? params->sel : nullptr;

  switch (metric_type) {
    case kMetricInnerProduct:
      range_search_inner_product(x, GetXb(), d, n, n_total, radius, result,
                                 sel);
      break;
    case kMetricL2:
      range_search_L2sqr(x, GetXb(), d, n, n_total, radius, result, sel);
      break;
    default:
      HYPERVEC_THROW_MSG("metric type not supported");
  }
}

void IndexFlat::ComputeDistanceSubset(idx_t n, const float* x, idx_t k,
                                        float* distances,
                                        const idx_t* labels) const {
  switch (metric_type) {
    case kMetricInnerProduct:
      fvec_inner_products_by_idx(distances, x, GetXb(), labels, d, n, k);
      break;
    case kMetricL2:
      fvec_L2sqr_by_idx(distances, x, GetXb(), labels, d, n, k);
      break;
    default:
      HYPERVEC_THROW_MSG("metric type not supported");
  }
}

namespace {

template <SIMDLevel SL>
struct FlatL2Dis : FlatCodesDistanceComputer {
  size_t d;
  idx_t nb;
  const float* b;
  size_t ndis;
  size_t npartial_dot_products;

  float distance_to_code(const uint8_t* code) final {
    ndis++;
    return fvec_L2sqr<SL>(q, (float*)code, d);
  }

  float partial_dot_product(const idx_t i, const uint32_t offset,
                            const uint32_t num_components) final override {
    npartial_dot_products++;
    return fvec_inner_product<SL>(q + offset, b + i * d + offset,
                                  num_components);
  }

  float symmetric_dis(idx_t i, idx_t j) override {
    return fvec_L2sqr<SL>(b + j * d, b + i * d, d);
  }

  explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
    : FlatCodesDistanceComputer(storage.codes.data(), storage.code_size, q)
    , d(storage.d)
    , nb(storage.n_total)
    , b(storage.GetXb())
    , ndis(0)
    , npartial_dot_products(0) {}

  void SetQuery(const float* x) override {
    q = x;
  }

  // compute four distances
  void distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2,
                         const idx_t idx3, float& dis0, float& dis1,
                         float& dis2, float& dis3) final override {
    ndis += 4;

    // compute first, Assign next
    const float* __restrict y0 =
      reinterpret_cast<const float*>(codes + idx0 * code_size);
    const float* __restrict y1 =
      reinterpret_cast<const float*>(codes + idx1 * code_size);
    const float* __restrict y2 =
      reinterpret_cast<const float*>(codes + idx2 * code_size);
    const float* __restrict y3 =
      reinterpret_cast<const float*>(codes + idx3 * code_size);

    float dp0 = 0;
    float dp1 = 0;
    float dp2 = 0;
    float dp3 = 0;
    fvec_L2sqr_batch_4<SL>(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
    dis0 = dp0;
    dis1 = dp1;
    dis2 = dp2;
    dis3 = dp3;
  }

  void partial_dot_product_batch_4(
    const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3,
    float& dp0, float& dp1, float& dp2, float& dp3, const uint32_t offset,
    const uint32_t num_components) final override {
    npartial_dot_products += 4;

    // compute first, Assign next
    const float* __restrict y0 =
      reinterpret_cast<const float*>(codes + idx0 * code_size);
    const float* __restrict y1 =
      reinterpret_cast<const float*>(codes + idx1 * code_size);
    const float* __restrict y2 =
      reinterpret_cast<const float*>(codes + idx2 * code_size);
    const float* __restrict y3 =
      reinterpret_cast<const float*>(codes + idx3 * code_size);

    float dp0_ = 0;
    float dp1_ = 0;
    float dp2_ = 0;
    float dp3_ = 0;
    fvec_inner_product_batch_4<SL>(q + offset, y0 + offset, y1 + offset,
                                   y2 + offset, y3 + offset, num_components,
                                   dp0_, dp1_, dp2_, dp3_);
    dp0 = dp0_;
    dp1 = dp1_;
    dp2 = dp2_;
    dp3 = dp3_;
  }
};

template <SIMDLevel SL>
struct FlatIPDis : FlatCodesDistanceComputer {
  size_t d;
  idx_t nb;
  const float* q;
  const float* b;
  size_t ndis;

  float symmetric_dis(idx_t i, idx_t j) final override {
    return fvec_inner_product<SL>(b + j * d, b + i * d, d);
  }

  float distance_to_code(const uint8_t* code) final override {
    ndis++;
    return fvec_inner_product<SL>(q, (const float*)code, d);
  }

  explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
    : FlatCodesDistanceComputer(storage.codes.data(), storage.code_size)
    , d(storage.d)
    , nb(storage.n_total)
    , q(q)
    , b(storage.GetXb())
    , ndis(0) {}

  void SetQuery(const float* x) override {
    q = x;
  }

  // compute four distances
  void distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2,
                         const idx_t idx3, float& dis0, float& dis1,
                         float& dis2, float& dis3) final override {
    ndis += 4;

    // compute first, Assign next
    const float* __restrict y0 =
      reinterpret_cast<const float*>(codes + idx0 * code_size);
    const float* __restrict y1 =
      reinterpret_cast<const float*>(codes + idx1 * code_size);
    const float* __restrict y2 =
      reinterpret_cast<const float*>(codes + idx2 * code_size);
    const float* __restrict y3 =
      reinterpret_cast<const float*>(codes + idx3 * code_size);

    float dp0 = 0;
    float dp1 = 0;
    float dp2 = 0;
    float dp3 = 0;
    fvec_inner_product_batch_4<SL>(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
    dis0 = dp0;
    dis1 = dp1;
    dis2 = dp2;
    dis3 = dp3;
  }
};

}  // namespace

FlatCodesDistanceComputer* IndexFlat::GetFlatCodesDistanceComputer() const {
  FlatCodesDistanceComputer* dc = nullptr;
  if (metric_type == kMetricL2) {
    with_simd_level([&]<SIMDLevel SL>() { dc = new FlatL2Dis<SL>(*this); });
  } else if (metric_type == kMetricInnerProduct) {
    with_simd_level([&]<SIMDLevel SL>() { dc = new FlatIPDis<SL>(*this); });
  } else {
    dc = get_extra_distance_computer(d, metric_type, metric_arg, n_total,
                                     GetXb());
  }
  return dc;
}

void IndexFlat::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT(key < n_total);
  memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlat::SaEncode(idx_t n, const float* x, uint8_t* bytes) const {
  if (n > 0) {
    memcpy(bytes, x, sizeof(float) * d * n);
  }
}

void IndexFlat::SaDecode(idx_t n, const uint8_t* bytes, float* x) const {
  if (n > 0) {
    memcpy(x, bytes, sizeof(float) * d * n);
  }
}

/***************************************************
 * IndexFlatL2
 ***************************************************/

namespace {
template <SIMDLevel SL>
struct FlatL2WithNormsDis : FlatCodesDistanceComputer {
  size_t d;
  idx_t nb;
  const float* q;
  const float* b;
  size_t ndis;

  const float* l2norms;
  float query_l2norm;

  float distance_to_code(const uint8_t* code) final override {
    ndis++;
    return fvec_L2sqr<SL>(q, (float*)code, d);
  }

  float operator()(const idx_t i) final override {
    const float* __restrict y =
      reinterpret_cast<const float*>(codes + i * code_size);

    prefetch_L2(l2norms + i);
    const float dp0 = fvec_inner_product<SL>(q, y, d);
    return query_l2norm + l2norms[i] - 2 * dp0;
  }

  float symmetric_dis(idx_t i, idx_t j) final override {
    const float* __restrict yi =
      reinterpret_cast<const float*>(codes + i * code_size);
    const float* __restrict yj =
      reinterpret_cast<const float*>(codes + j * code_size);

    prefetch_L2(l2norms + i);
    prefetch_L2(l2norms + j);
    const float dp0 = fvec_inner_product<SL>(yi, yj, d);
    return l2norms[i] + l2norms[j] - 2 * dp0;
  }

  explicit FlatL2WithNormsDis(const IndexFlatL2& storage,
                              const float* q = nullptr)
    : FlatCodesDistanceComputer(storage.codes.data(), storage.code_size)
    , d(storage.d)
    , nb(storage.n_total)
    , q(q)
    , b(storage.GetXb())
    , ndis(0)
    , l2norms(storage.cached_l2norms.data())
    , query_l2norm(0) {}

  void SetQuery(const float* x) override {
    q = x;
    query_l2norm = fvec_norm_L2sqr<SL>(q, d);
  }

  // compute four distances
  void distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2,
                         const idx_t idx3, float& dis0, float& dis1,
                         float& dis2, float& dis3) final override {
    ndis += 4;

    // compute first, Assign next
    const float* __restrict y0 =
      reinterpret_cast<const float*>(codes + idx0 * code_size);
    const float* __restrict y1 =
      reinterpret_cast<const float*>(codes + idx1 * code_size);
    const float* __restrict y2 =
      reinterpret_cast<const float*>(codes + idx2 * code_size);
    const float* __restrict y3 =
      reinterpret_cast<const float*>(codes + idx3 * code_size);

    prefetch_L2(l2norms + idx0);
    prefetch_L2(l2norms + idx1);
    prefetch_L2(l2norms + idx2);
    prefetch_L2(l2norms + idx3);

    float dp0 = 0;
    float dp1 = 0;
    float dp2 = 0;
    float dp3 = 0;
    fvec_inner_product_batch_4<SL>(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
    dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
    dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
    dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
    dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
  }
};

}  // namespace

void IndexFlatL2::SyncL2Norms() {
  cached_l2norms.resize(n_total);
  fvec_norms_L2sqr(cached_l2norms.data(),
                   reinterpret_cast<const float*>(codes.data()), d, n_total);
}

void IndexFlatL2::ClearL2Norms() {
  cached_l2norms.clear();
  cached_l2norms.shrink_to_fit();
}

FlatCodesDistanceComputer* IndexFlatL2::GetFlatCodesDistanceComputer() const {
  if (metric_type == kMetricL2) {
    if (!cached_l2norms.empty()) {
      FlatCodesDistanceComputer* dc = nullptr;
      with_simd_level(
        [&]<SIMDLevel SL>() { dc = new FlatL2WithNormsDis<SL>(*this); });
      return dc;
    }
  }

  return IndexFlat::GetFlatCodesDistanceComputer();
}

/***************************************************
 * IndexFlat1D
 ***************************************************/

IndexFlat1D::IndexFlat1D(bool continuous_update)
  : IndexFlatL2(1), continuous_update(continuous_update) {}

/// if not continuous_update, call this between the last Add and
/// the first Search
void IndexFlat1D::UpdatePermutation() {
  perm.resize(n_total);
  if (n_total < 1000000) {
    fvec_argsort(n_total, GetXb(), (size_t*)perm.data());
  } else {
    fvec_argsort_parallel(n_total, GetXb(), (size_t*)perm.data());
  }
}

void IndexFlat1D::Add(idx_t n, const float* x) {
  IndexFlatL2::Add(n, x);
  if (continuous_update) {
    UpdatePermutation();
  }
}

void IndexFlat1D::Reset() {
  IndexFlatL2::Reset();
  perm.clear();
}

void IndexFlat1D::Search(idx_t n, const float* x, idx_t k, float* distances,
                         idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_MSG(!params,
                            "Search params not supported for this index");
  HYPERVEC_THROW_IF_NOT(k > 0);
  HYPERVEC_THROW_IF_NOT_MSG(perm.size() == n_total,
                            "Call UpdatePermutation before Search");
  const float* xb = GetXb();

#pragma omp parallel for if (n > 10000)
  for (idx_t i = 0; i < n; i++) {
    float q = x[i];  // query
    float* D = distances + i * k;
    idx_t* I = labels + i * k;

    // binary Search
    idx_t i0 = 0, i1 = n_total;
    idx_t wp = 0;

    if (n_total == 0) {
      for (idx_t j = 0; j < k; j++) {
        I[j] = -1;
        D[j] = HUGE_VAL;
      }
      goto done;
    }

    if (xb[perm[i0]] > q) {
      i1 = 0;
      goto finish_right;
    }

    if (xb[perm[i1 - 1]] <= q) {
      i0 = i1 - 1;
      goto finish_left;
    }

    while (i0 + 1 < i1) {
      idx_t imed = (i0 + i1) / 2;
      if (xb[perm[imed]] <= q) {
        i0 = imed;
      } else {
        i1 = imed;
      }
    }

    // query is between xb[perm[i0]] and xb[perm[i1]]
    // expand to nearest neighs

    while (wp < k) {
      float xleft = xb[perm[i0]];
      float xright = xb[perm[i1]];

      if (q - xleft < xright - q) {
        D[wp] = q - xleft;
        I[wp] = perm[i0];
        i0--;
        wp++;
        if (i0 < 0) {
          goto finish_right;
        }
      } else {
        D[wp] = xright - q;
        I[wp] = perm[i1];
        i1++;
        wp++;
        if (i1 >= n_total) {
          goto finish_left;
        }
      }
    }
    goto done;

  finish_right:
    // grow to the right from i1
    while (wp < k) {
      if (i1 < n_total) {
        D[wp] = xb[perm[i1]] - q;
        I[wp] = perm[i1];
        i1++;
      } else {
        D[wp] = std::numeric_limits<float>::infinity();
        I[wp] = -1;
      }
      wp++;
    }
    goto done;

  finish_left:
    // grow to the left from i0
    while (wp < k) {
      if (i0 >= 0) {
        D[wp] = q - xb[perm[i0]];
        I[wp] = perm[i0];
        i0--;
      } else {
        D[wp] = std::numeric_limits<float>::infinity();
        I[wp] = -1;
      }
      wp++;
    }
  done:;
  }
}

}  // namespace hypervec
