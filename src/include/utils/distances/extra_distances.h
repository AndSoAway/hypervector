/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

/** In this file are the implementations of extra metrics beyond L2
 *  and inner product */

#include <core/id_selector.h>
#include <core/metric_type.h>
#include <core/simd_dispatch.h>
#include <utils/structures/ordered_key_value.h>

#include <cstdint>

namespace hypervec {

struct FlatCodesDistanceComputer;

void pairwise_extra_distances(int64_t d, int64_t nq, const float* xq,
                              int64_t nb, const float* xb, MetricType mt,
                              float metric_arg, float* dis, int64_t ldq = -1,
                              int64_t ldb = -1, int64_t ldd = -1);

void knn_extra_metrics(const float* x, const float* y, size_t d, size_t nx,
                       size_t ny, MetricType mt, float metric_arg, size_t k,
                       float* distances, int64_t* indexes,
                       const IDSelector* sel = nullptr);

/** get a DistanceComputer that refers to this type of distance and
 *  indexes a flat array of size nb */
FlatCodesDistanceComputer* get_extra_distance_computer(size_t d, MetricType mt,
                                                       float metric_arg,
                                                       size_t nb,
                                                       const float* xb);

/// Dispatch to a lambda with MetricType as a compile-time constant.
/// This allows writing generic code that works with different metrics
/// while maintaining compile-time optimization.
///
/// Example usage:
///   auto result = with_metric_type(runtime_metric, [&](auto metric_tag) {
///       constexpr MetricType M = decltype(metric_tag)::value;
///       return compute_distance<M>(x, y);
///   });
#ifndef SWIG

template <typename LambdaType>
inline auto with_metric_type(MetricType metric, LambdaType&& action) {
  switch (metric) {
    case kMetricInnerProduct:
      return action.template operator()<kMetricInnerProduct>();
    case kMetricL2:
      return action.template operator()<kMetricL2>();
    case kMetricL1:
      return action.template operator()<kMetricL1>();
    case kMetricLinf:
      return action.template operator()<kMetricLinf>();
    case kMetricLp:
      return action.template operator()<kMetricLp>();
    case kMetricCanberra:
      return action.template operator()<kMetricCanberra>();
    case kMetricBrayCurtis:
      return action.template operator()<kMetricBrayCurtis>();
    case kMetricJensenShannon:
      return action.template operator()<kMetricJensenShannon>();
    case kMetricJaccard:
      return action.template operator()<kMetricJaccard>();
    case kMetricNaNEuclidean:
      return action.template operator()<kMetricNaNEuclidean>();
    case kMetricGower:
      return action.template operator()<kMetricGower>();
    default:
      HYPERVEC_THROW_FMT("with_metric_type called with unknown metric %d",
                         int(metric));
  }
}
#endif  // SWIG

#ifndef SWIG

/***************************************************************************
 * VectorDistance base class - contains common data members and type defs
 * VectorDistance struct template - specializations for each metric type
 **************************************************************************/

template <MetricType mt, SIMDLevel level>
struct VectorDistance {
  size_t d;
  float metric_arg;

  VectorDistance(size_t d, float metric_arg) : d(d), metric_arg(metric_arg) {}

  static constexpr MetricType metric = mt;
  static constexpr bool is_similarity = IsSimilarityMetric(mt);

  using C =
    typename std::conditional<IsSimilarityMetric(mt), CMin<float, int64_t>,
                              CMax<float, int64_t>>::type;

  float operator()(const float* x, const float* y) const;
};

/***************************************************************************
 * Dispatching function that takes a lambda directly.
 * The lambda should be templated on VectorDistance, eg.:
 *
 *   auto result = with_VectorDistance(
 *       metric, metric_arg, [&]<class VD>(VD vd) {
 *           return vd(x, y);
 *       });
 **************************************************************************/

template <typename LambdaType>
auto with_VectorDistance(size_t d, MetricType metric, float metric_arg,
                         LambdaType&& action) {
  auto dispatch_metric = [&]<MetricType mt>() {
    auto call = [&]<SIMDLevel level>() {
      VectorDistance<mt, level> vd = {d, metric_arg};
      return action(vd);
    };

    constexpr bool has_simd = mt == kMetricInnerProduct || mt == kMetricL2 ||
                              mt == kMetricL1 || mt == kMetricLinf;
    if constexpr (!has_simd) {
      return call.template operator()<SIMDLevel::NONE>();
    } else {
      DISPATCH_SIMDLevel(call.template operator());
    }
  };
  return with_metric_type(metric, dispatch_metric);
}

#endif  // SWIG

}  // namespace hypervec
