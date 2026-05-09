/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <utils/algo/kmeans/kmeans.h>
#include <utils/distances/distances.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <cmath>
#include <vector>

namespace {

// Helper: assign each x[i] to its nearest centroid and return total L2sqr
// objective. Used to cross-check the value returned by RunKMeans.
float ComputeObjective(hypervec::idx_t n, const float* x, hypervec::idx_t d,
                       hypervec::idx_t k, const float* centroids) {
  float total = 0.0f;
  for (hypervec::idx_t i = 0; i < n; i++) {
    float best = std::numeric_limits<float>::infinity();
    for (hypervec::idx_t c = 0; c < k; c++) {
      const float dist = hypervec::fvec_L2sqr(
        x + i * d, centroids + c * d, static_cast<size_t>(d));
      if (dist < best) {
        best = dist;
      }
    }
    total += best;
  }
  return total;
}

// Helper: build a synthetic dataset of `k_true` well-separated clusters,
// each with `pts_per_cluster` points sampled from a unit-variance gaussian
// around a centre.
std::vector<float> MakeClusteredData(hypervec::idx_t d, hypervec::idx_t k_true,
                                     hypervec::idx_t pts_per_cluster,
                                     int64_t seed) {
  hypervec::RandomGenerator rng(seed);
  std::vector<float> x(static_cast<size_t>(k_true) * pts_per_cluster * d);

  // pick well-separated centre points on a hypercube vertex
  std::vector<float> centres(static_cast<size_t>(k_true) * d);
  for (hypervec::idx_t c = 0; c < k_true; c++) {
    for (hypervec::idx_t j = 0; j < d; j++) {
      const float vertex = (((c >> j) & 1) ? 10.0f : -10.0f);
      centres[c * d + j] = vertex;
    }
  }

  for (hypervec::idx_t c = 0; c < k_true; c++) {
    for (hypervec::idx_t i = 0; i < pts_per_cluster; i++) {
      const hypervec::idx_t row = c * pts_per_cluster + i;
      for (hypervec::idx_t j = 0; j < d; j++) {
        // small jitter (uniform in [-1,1]) around the cluster centre
        const float jitter = 2.0f * rng.rand_float() - 1.0f;
        x[row * d + j] = centres[c * d + j] + jitter;
      }
    }
  }
  return x;
}

}  // namespace

TEST(KMeans, ConvergesOnWellSeparatedClusters) {
  const hypervec::idx_t d = 4;
  const hypervec::idx_t k = 4;
  const hypervec::idx_t pts_per_cluster = 50;
  const hypervec::idx_t n = k * pts_per_cluster;

  const std::vector<float> x = MakeClusteredData(d, k, pts_per_cluster, 7);
  std::vector<float> centroids(static_cast<size_t>(k) * d);

  // Use a few redos; a single random init can land in a local minimum where
  // two true clusters share one centroid. PQ training will typically use
  // nredo>1 for the same reason.
  hypervec::KMeansParameters params;
  params.nredo = 5;
  const float objective =
    hypervec::RunKMeans(n, x.data(), d, k, centroids.data(), params);

  // Returned objective must match what we recompute from the centroids.
  const float verify = ComputeObjective(n, x.data(), d, k, centroids.data());
  ASSERT_NEAR(objective, verify, 1.0f);

  // With well-separated clusters and uniform jitter in [-1,1], the per-point
  // squared distance to its true centroid is bounded by d * 1 = 4. Allow
  // some slack for the centroid estimate.
  EXPECT_LT(objective, static_cast<float>(n) * d * 2.0f);
}

TEST(KMeans, NRedoBeatsSingleRun) {
  const hypervec::idx_t d = 8;
  const hypervec::idx_t k = 6;
  const hypervec::idx_t pts_per_cluster = 30;
  const hypervec::idx_t n = k * pts_per_cluster;

  const std::vector<float> x = MakeClusteredData(d, k, pts_per_cluster, 11);

  std::vector<float> c1(static_cast<size_t>(k) * d);
  hypervec::KMeansParameters single;
  single.nredo = 1;
  const float obj_single =
    hypervec::RunKMeans(n, x.data(), d, k, c1.data(), single);

  std::vector<float> c5(static_cast<size_t>(k) * d);
  hypervec::KMeansParameters multi;
  multi.nredo = 5;
  const float obj_multi =
    hypervec::RunKMeans(n, x.data(), d, k, c5.data(), multi);

  // The best of 5 redos must not be worse than a single run with the same
  // base seed (redo=0 of the multi run uses the same seed as `single`).
  EXPECT_LE(obj_multi, obj_single + 1e-4f);
}

TEST(KMeans, NEqualsKEachClusterHoldsOnePoint) {
  const hypervec::idx_t d = 3;
  const hypervec::idx_t k = 5;
  const hypervec::idx_t n = k;

  // Five distinct points; the only zero-objective solution is one centroid
  // per point.
  const std::vector<float> x = {
    1.0f,  0.0f,  0.0f,
    0.0f,  1.0f,  0.0f,
    0.0f,  0.0f,  1.0f,
    -1.0f, 0.0f,  0.0f,
    0.0f,  -1.0f, 0.0f,
  };
  std::vector<float> centroids(static_cast<size_t>(k) * d);

  const float objective =
    hypervec::RunKMeans(n, x.data(), d, k, centroids.data());

  // With n == k and distinct inputs, k-means should drive the objective to 0.
  EXPECT_NEAR(objective, 0.0f, 1e-4f);
}

TEST(KMeans, RejectsTooFewTrainingVectors) {
  const hypervec::idx_t d = 2;
  const hypervec::idx_t k = 8;
  const hypervec::idx_t n = 4;  // intentionally < k
  const std::vector<float> x(static_cast<size_t>(n) * d, 0.0f);
  std::vector<float> centroids(static_cast<size_t>(k) * d);

  EXPECT_THROW(
    hypervec::RunKMeans(n, x.data(), d, k, centroids.data()),
    hypervec::HypervecException);
}

TEST(KMeans, RejectsZeroOrNegativeNiter) {
  const hypervec::idx_t d = 2;
  const hypervec::idx_t k = 2;
  const hypervec::idx_t n = 10;
  const std::vector<float> x(static_cast<size_t>(n) * d, 0.0f);
  std::vector<float> centroids(static_cast<size_t>(k) * d);

  hypervec::KMeansParameters bad;
  bad.niter = 0;
  EXPECT_THROW(
    hypervec::RunKMeans(n, x.data(), d, k, centroids.data(), bad),
    hypervec::HypervecException);
}

TEST(KMeans, SphericalProducesUnitNormCentroids) {
  const hypervec::idx_t d = 4;
  const hypervec::idx_t k = 3;
  const hypervec::idx_t pts_per_cluster = 20;
  const hypervec::idx_t n = k * pts_per_cluster;

  const std::vector<float> x = MakeClusteredData(d, k, pts_per_cluster, 13);
  std::vector<float> centroids(static_cast<size_t>(k) * d);

  hypervec::KMeansParameters params;
  params.spherical = true;
  hypervec::RunKMeans(n, x.data(), d, k, centroids.data(), params);

  for (hypervec::idx_t c = 0; c < k; c++) {
    float norm2 = 0.0f;
    for (hypervec::idx_t j = 0; j < d; j++) {
      const float v = centroids[c * d + j];
      norm2 += v * v;
    }
    EXPECT_NEAR(std::sqrt(norm2), 1.0f, 1e-4f)
      << "centroid " << c << " not unit norm";
  }
}

TEST(KMeans, InnerProductMetricGroupsCosineSimilarPoints) {
  // Two well-separated cosine clusters: one near (+1, 0, ...) direction,
  // one near (-1, 0, ...). Magnitudes vary so naive L2 would group by
  // magnitude, not direction; IP-spherical k-means should group by direction.
  const hypervec::idx_t d = 4;
  const hypervec::idx_t k = 2;
  const hypervec::idx_t pts_per_cluster = 30;
  const hypervec::idx_t n = k * pts_per_cluster;

  hypervec::RandomGenerator rng(99);
  std::vector<float> x(static_cast<size_t>(n) * d);
  for (hypervec::idx_t i = 0; i < pts_per_cluster; i++) {
    // varied-magnitude vectors pointing roughly +x
    const float scale = 1.0f + 9.0f * rng.rand_float();
    x[i * d + 0] = scale * (1.0f + 0.05f * rng.rand_float());
    x[i * d + 1] = scale * 0.05f * (rng.rand_float() - 0.5f);
    x[i * d + 2] = scale * 0.05f * (rng.rand_float() - 0.5f);
    x[i * d + 3] = scale * 0.05f * (rng.rand_float() - 0.5f);
  }
  for (hypervec::idx_t i = 0; i < pts_per_cluster; i++) {
    // varied-magnitude vectors pointing roughly -x
    const float scale = 1.0f + 9.0f * rng.rand_float();
    const hypervec::idx_t row = pts_per_cluster + i;
    x[row * d + 0] = scale * (-1.0f + 0.05f * rng.rand_float());
    x[row * d + 1] = scale * 0.05f * (rng.rand_float() - 0.5f);
    x[row * d + 2] = scale * 0.05f * (rng.rand_float() - 0.5f);
    x[row * d + 3] = scale * 0.05f * (rng.rand_float() - 0.5f);
  }

  std::vector<float> centroids(static_cast<size_t>(k) * d);
  hypervec::KMeansParameters params;
  params.metric = hypervec::kMetricInnerProduct;
  params.nredo = 5;  // IP training with random init benefits from redos
  hypervec::RunKMeans(n, x.data(), d, k, centroids.data(), params);

  // IP forces spherical: every centroid must be unit norm.
  for (hypervec::idx_t c = 0; c < k; c++) {
    float norm2 = 0.0f;
    for (hypervec::idx_t j = 0; j < d; j++) {
      const float v = centroids[c * d + j];
      norm2 += v * v;
    }
    EXPECT_NEAR(std::sqrt(norm2), 1.0f, 1e-4f)
      << "centroid " << c << " not unit norm under kMetricInnerProduct";
  }

  // The two centroids should point in opposite directions (cos(theta) ~ -1).
  // Exact value depends on random init, but it must be < 0.
  float dot = 0.0f;
  for (hypervec::idx_t j = 0; j < d; j++) {
    dot += centroids[0 * d + j] * centroids[1 * d + j];
  }
  EXPECT_LT(dot, 0.0f)
    << "IP k-means failed to separate opposite-direction clusters";
}

TEST(KMeans, RejectsUnsupportedMetric) {
  const hypervec::idx_t d = 2;
  const hypervec::idx_t k = 3;
  const hypervec::idx_t n = 20;
  const std::vector<float> x(static_cast<size_t>(n) * d, 1.0f);
  std::vector<float> centroids(static_cast<size_t>(k) * d);

  // L1 needs a median update rule, which is not implemented yet. The kmeans
  // module must refuse rather than silently fall back to L2 mean update.
  hypervec::KMeansParameters params;
  params.metric = hypervec::kMetricL1;
  EXPECT_THROW(
    hypervec::RunKMeans(n, x.data(), d, k, centroids.data(), params),
    hypervec::HypervecException);

  // Same for Lp / Canberra etc. — pick one representative.
  params.metric = hypervec::kMetricLp;
  params.metric_arg = 3.0f;
  EXPECT_THROW(
    hypervec::RunKMeans(n, x.data(), d, k, centroids.data(), params),
    hypervec::HypervecException);
}

TEST(KMeans, EmptyClusterSplitProducesDistinctCentroids) {
  // Construct a degenerate dataset where naive Lloyd would leave clusters
  // empty: only 2 distinct points but k=4 requested. The empty-cluster
  // split should yield 4 distinct centroids (modulo the tiny epsilon
  // perturbation), so no two centroids should coincide exactly.
  const hypervec::idx_t d = 2;
  const hypervec::idx_t k = 4;
  const hypervec::idx_t n = 8;
  const std::vector<float> x = {
    0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    5.0f, 5.0f,  5.0f, 5.0f,  5.0f, 5.0f,  5.0f, 5.0f,
  };
  std::vector<float> centroids(static_cast<size_t>(k) * d);

  hypervec::RunKMeans(n, x.data(), d, k, centroids.data());

  // Pairwise: no two centroids should be byte-identical.
  for (hypervec::idx_t a = 0; a < k; a++) {
    for (hypervec::idx_t b = a + 1; b < k; b++) {
      bool identical = true;
      for (hypervec::idx_t j = 0; j < d; j++) {
        if (centroids[a * d + j] != centroids[b * d + j]) {
          identical = false;
          break;
        }
      }
      EXPECT_FALSE(identical)
        << "centroids " << a << " and " << b << " collapsed to identical";
    }
  }
}
