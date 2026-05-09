/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <utils/distances/metric_type.h>

namespace hypervec {

/// Project-wide default seed for k-means random initialization. Exposed as a
/// macro so callers (IVF training, PQ subquantizer training, tests) can refer
/// to a single source of truth instead of repeating a literal.
#define HYPERVEC_KMEANS_DEFAULT_SEED 1234

/// Default Lloyd iteration count, sufficient for typical IVF nlist (1k-65k)
/// and PQ ksub=256 training sets.
#define HYPERVEC_KMEANS_DEFAULT_NITER 25

/// Tunables for RunKMeans. Defaults preserve historical IVF behaviour
/// (L2 metric, mean-update, niter=25, seed=1234, nredo=1).
struct KMeansParameters {
  /// Number of Lloyd iterations per redo.
  int niter = HYPERVEC_KMEANS_DEFAULT_NITER;

  /// RNG seed for centroid initialization. Successive redos use seed + redo_idx.
  int seed = HYPERVEC_KMEANS_DEFAULT_SEED;

  /// Number of independent random restarts; the run with the lowest objective
  /// is returned. PQ training typically benefits from nredo > 1.
  int nredo = 1;

  /// Print per-iteration objective and redo summaries to stderr.
  bool verbose = false;

  /// Normalize centroids to unit length after each iteration. Required when
  /// metric == kMetricInnerProduct (otherwise centroid magnitude diverges);
  /// optional otherwise (e.g. L2 training on already-normalized data).
  bool spherical = false;

  /// Metric used for both data-to-centroid assignment and the objective.
  ///
  /// Currently supported values (others throw HypervecException at training
  /// time):
  ///   - kMetricL2 (default): mean update, L2sqr assignment via knn_L2sqr.
  ///   - kMetricInnerProduct: spherical (unit-norm) centroids; spherical is
  ///     forced true. Assignment via knn_inner_product. Useful for PQ-IP and
  ///     cosine training.
  ///
  /// The dispatch in kmeans.cpp is structured so adding support for a new
  /// metric (e.g. L1 with median update) is "add one case". See the
  /// `Lloyd*` helpers and the assignment switch.
  MetricType metric = kMetricL2;

  /// Optional argument for parameterized metrics (e.g. p in L_p). Reserved
  /// for future metric implementations; unused by the L2 / IP paths.
  float metric_arg = 0.0f;
};

/** Lloyd's k-means with random init (sample without replacement). The
 *  assignment kernel and centroid update rule are selected by
 *  `params.metric`; see KMeansParameters::metric for supported values.
 *
 *  Empty clusters are split from the heaviest cluster with a small
 *  per-dimension perturbation so the returned centroid table is free of
 *  degenerate duplicates.
 *
 *  The function is pure (no global / static mutable state) and intentionally
 *  does **not** parallelize internally: outer callers (e.g. PQ training M
 *  subquantizers) parallelize across kmeans instances instead.
 *
 *  @param n         number of training vectors (must be >= k)
 *  @param x         training vectors, size n * d, row-major
 *  @param d         vector dimension
 *  @param k         number of centroids
 *  @param centroids output centroids, size k * d, row-major
 *  @param params    tunables; default-constructed value reproduces historical
 *                   IVF behaviour (L2, niter=25, seed=1234, nredo=1)
 *  @return          final objective value, lower is better. For dissimilarity
 *                   metrics (kMetricL2) this is the sum of squared distances;
 *                   for similarity metrics (kMetricInnerProduct) the sign is
 *                   flipped (-Σ similarity) so `nredo` selection always picks
 *                   the run with the smallest returned value.
 *  @throws HypervecException if params.metric is not supported, n < k,
 *                            niter <= 0, or nredo <= 0.
 */
float RunKMeans(idx_t n, const float* x, idx_t d, idx_t k, float* centroids,
                const KMeansParameters& params = {});

}  // namespace hypervec
