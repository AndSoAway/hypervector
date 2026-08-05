/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/algo/kmeans/kmeans.h>

#include <utils/distances/distances.h>
#include <utils/log/assert.h>
#include <utils/structures/heap.h>
#include <utils/structures/random.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace hypervec {

namespace {

/// Reject metrics whose mathematical centroid update is not yet implemented.
/// Adding support for a new metric means:
///   1. extend this whitelist
///   2. implement its assignment branch in AssignNearest()
///   3. implement its centroid update rule (e.g. median for L1)
///   4. ensure the objective sign convention (small = better) is preserved
void CheckMetricSupported(MetricType metric) {
  switch (metric) {
    case kMetricL2:
    case kMetricInnerProduct:
      return;
    default:
      HYPERVEC_THROW_FMT(
        "RunKMeans: metric %d is not supported yet. Currently supported: "
        "kMetricL2, kMetricInnerProduct. To add support for another metric, "
        "extend the dispatch in src/utils/algo/kmeans/kmeans.cpp.",
        static_cast<int>(metric));
  }
}

/// Assign each training vector to its nearest centroid under `metric`.
/// On return, assign[i] is the centroid index for x[i] and dis[i] is the
/// raw kernel score (L2sqr for L2, inner product for IP).
void AssignNearest(idx_t n, const float* x, idx_t d, idx_t k,
                   const float* centroids, MetricType metric, idx_t* assign,
                   float* dis) {
  if (metric == kMetricInnerProduct) {
    // knn_inner_product fills dis[] with the IP scores of the assigned
    // centroid (larger = better). We use a min-heap of size 1 because the
    // result handler keeps the largest score at top so it can be replaced
    // when a larger one comes in.
    float_minheap_array_t res = {static_cast<size_t>(n), 1, assign, dis};
    knn_inner_product(x, centroids, static_cast<size_t>(d),
                      static_cast<size_t>(n), static_cast<size_t>(k), &res);
  } else {
    // kMetricL2 (the only other case CheckMetricSupported lets through).
    float_maxheap_array_t res = {static_cast<size_t>(n), 1, assign, dis};
    knn_L2sqr(x, centroids, static_cast<size_t>(d), static_cast<size_t>(n),
              static_cast<size_t>(k), &res);
  }
}

/// Convert raw per-point kernel scores into a per-point loss whose sum is
/// minimized by good clusterings. For dissimilarity metrics (L2) the kernel
/// score is already a loss; for similarity metrics (IP) we negate.
float ScoreToLoss(float score, MetricType metric) {
  return IsSimilarityMetric(metric) ? -score : score;
}

/// Normalize each row of `centroids` (k rows of dim d) to unit L2 length.
/// Rows with zero norm are left untouched.
void NormalizeRows(float* centroids, idx_t k, idx_t d) {
  for (idx_t c = 0; c < k; c++) {
    float* row = centroids + c * d;
    float norm2 = 0.0f;
    for (idx_t j = 0; j < d; j++) {
      norm2 += row[j] * row[j];
    }
    if (norm2 > 0.0f) {
      const float inv = 1.0f / std::sqrt(norm2);
      for (idx_t j = 0; j < d; j++) {
        row[j] *= inv;
      }
    }
  }
}

/// One Lloyd run. Returns the total loss (smaller is better) under the chosen
/// metric, computed against the post-update centroids.
float LloydOnce(idx_t n, const float* x, idx_t d, idx_t k, float* centroids,
                const KMeansParameters& params, int seed) {
  const MetricType metric = params.metric;
  // IP requires unit-norm centroids; allow caller to set spherical
  // explicitly, otherwise force it on for IP.
  const bool spherical = params.spherical || (metric == kMetricInnerProduct);

  // ---------- init: sample k distinct vectors ----------
  RandomGenerator rng(seed);
  std::vector<idx_t> perm(static_cast<size_t>(n));
  std::iota(perm.begin(), perm.end(), idx_t{0});
  for (idx_t i = 0; i < k; i++) {
    const int range = static_cast<int>(n - i);
    const idx_t j = i + static_cast<idx_t>(rng.rand_int(range));
    std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(j)]);
  }
  for (idx_t i = 0; i < k; i++) {
    std::memcpy(centroids + i * d, x + perm[static_cast<size_t>(i)] * d,
                static_cast<size_t>(d) * sizeof(float));
  }
  if (spherical) {
    NormalizeRows(centroids, k, d);
  }

  // ---------- Lloyd iterations ----------
  std::vector<idx_t> assign(static_cast<size_t>(n));
  std::vector<float> dis(static_cast<size_t>(n));
  std::vector<float> new_centroids(static_cast<size_t>(k) * d);
  std::vector<idx_t> cnt(static_cast<size_t>(k));
  float loss = std::numeric_limits<float>::infinity();

  for (int it = 0; it < params.niter; it++) {
    AssignNearest(n, x, d, k, centroids, metric, assign.data(), dis.data());

    loss = 0.0f;
    for (idx_t i = 0; i < n; i++) {
      loss += ScoreToLoss(dis[static_cast<size_t>(i)], metric);
    }

    // Recompute centroids as mean of assigned vectors. Both supported
    // metrics use the arithmetic mean here: for L2 it's the loss minimizer;
    // for IP-spherical the mean is the unit-norm direction maximizing
    // Σ x_i · c, which we obtain by normalizing right after.
    std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
    std::fill(cnt.begin(), cnt.end(), idx_t{0});

    for (idx_t i = 0; i < n; i++) {
      const idx_t c = assign[static_cast<size_t>(i)];
      cnt[static_cast<size_t>(c)]++;
      const float* xi = x + i * d;
      float* nc = new_centroids.data() + c * d;
      for (idx_t j = 0; j < d; j++) {
        nc[j] += xi[j];
      }
    }

    // find the heaviest cluster (used to seed splits for empty ones)
    idx_t heaviest = 0;
    for (idx_t c = 1; c < k; c++) {
      if (cnt[static_cast<size_t>(c)] >
          cnt[static_cast<size_t>(heaviest)]) {
        heaviest = c;
      }
    }

    // commit means first so the heaviest centroid is fresh before splits
    // read it
    for (idx_t c = 0; c < k; c++) {
      if (cnt[static_cast<size_t>(c)] > 0) {
        float* cp = centroids + c * d;
        const float inv =
          1.0f / static_cast<float>(cnt[static_cast<size_t>(c)]);
        const float* nc = new_centroids.data() + c * d;
        for (idx_t j = 0; j < d; j++) {
          cp[j] = nc[j] * inv;
        }
      }
    }

    // Faiss-style multiplicative perturbation with an additive floor so the
    // split survives float32 precision for both small (PQ subspace centres
    // near 0) and large (raw vector centres) magnitudes.
    constexpr float kSplitFactor = 1.0f / 1024.0f;
    int empty_idx = 0;
    for (idx_t c = 0; c < k; c++) {
      if (cnt[static_cast<size_t>(c)] == 0 &&
          cnt[static_cast<size_t>(heaviest)] > 0 && heaviest != c) {
        float* cp = centroids + c * d;
        const float* src = centroids + heaviest * d;
        for (idx_t j = 0; j < d; j++) {
          const float sign = ((empty_idx + j) & 1) ? 1.0f : -1.0f;
          const float scale =
            std::max(std::fabs(src[j]), 1.0f) * kSplitFactor;
          const float mag = scale * static_cast<float>(empty_idx + 1);
          cp[j] = src[j] + sign * mag;
        }
        empty_idx++;
      }
    }

    if (spherical) {
      NormalizeRows(centroids, k, d);
    }

    if (params.verbose) {
      std::fprintf(stderr,
                   "  KMeans iter %2d/%d  loss = %.6g\n",
                   it + 1, params.niter, static_cast<double>(loss));
    }
  }

  // Final assignment so the returned loss reflects the post-update centroids
  // (otherwise it is one Lloyd step stale).
  AssignNearest(n, x, d, k, centroids, metric, assign.data(), dis.data());
  loss = 0.0f;
  for (idx_t i = 0; i < n; i++) {
    loss += ScoreToLoss(dis[static_cast<size_t>(i)], metric);
  }
  return loss;
}

}  // namespace

float RunKMeans(idx_t n, const float* x, idx_t d, idx_t k, float* centroids,
                const KMeansParameters& params) {
  CheckMetricSupported(params.metric);
  HYPERVEC_THROW_IF_NOT_FMT(
    n >= k,
    "KMeans: need at least k=%ld training vectors, got %ld",
    static_cast<long>(k), static_cast<long>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
    params.niter > 0, "KMeans: niter must be > 0, got %d", params.niter);
  HYPERVEC_THROW_IF_NOT_FMT(
    params.nredo > 0, "KMeans: nredo must be > 0, got %d", params.nredo);

  if (params.nredo == 1) {
    return LloydOnce(n, x, d, k, centroids, params, params.seed);
  }

  // nredo > 1: keep the run with the lowest loss
  std::vector<float> best_centroids(static_cast<size_t>(k) * d);
  float best_loss = std::numeric_limits<float>::infinity();

  for (int redo = 0; redo < params.nredo; redo++) {
    const float loss =
      LloydOnce(n, x, d, k, centroids, params, params.seed + redo);
    if (params.verbose) {
      std::fprintf(stderr, "KMeans redo %d/%d loss = %.6g\n",
                   redo + 1, params.nredo, static_cast<double>(loss));
    }
    if (loss < best_loss) {
      best_loss = loss;
      std::memcpy(best_centroids.data(), centroids,
                  static_cast<size_t>(k) * d * sizeof(float));
    }
  }

  std::memcpy(centroids, best_centroids.data(),
              static_cast<size_t>(k) * d * sizeof(float));
  return best_loss;
}

}  // namespace hypervec
