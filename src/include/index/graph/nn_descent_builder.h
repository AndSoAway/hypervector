/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>
#include <utils/distances/distance_computer.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace hypervec {

struct NNDescentOptions {
  size_t max_degree = 32;
  size_t max_iterations = 30;
  double convergence_threshold = 0.001;
  uint64_t random_seed = 0x9E3779B97F4A7C15ULL;
  double sample_rate = 0.2;
  /** Parallel distance evaluations; the graph is still updated in order. */
  size_t build_threads = 1;
};

struct NNDescentStats {
  size_t iterations = 0;
  size_t initial_distance_computations = 0;
  size_t refinement_distance_computations = 0;
  size_t neighbor_updates = 0;
  size_t sampled_old_neighbors = 0;
  size_t sampled_new_neighbors = 0;
  size_t sampled_neighbors_trimmed = 0;
  size_t peak_sampled_neighbors = 0;
  bool converged = false;

  void Reset() noexcept;
  void Combine(const NNDescentStats& other) noexcept;
};

/** Batch builder for an approximate directed k-nearest-neighbor graph.
 *
 * The builder starts from deterministic random neighbors. Every iteration
 * samples edges into old and new sets, joins new-new and new-old pairs, and
 * retains the closest candidates. Sample sets include reverse observations
 * and are bounded by a constant multiple of max_degree to prevent
 * high-indegree hubs from causing unbounded candidate growth.
 * DistanceComputer must expose all vectors through symmetric_dis(), with
 * smaller values meaning closer neighbors.
 */
class NNDescentBuilder {
 public:
  using DistanceFactory = std::function<std::unique_ptr<DistanceComputer>()>;

  explicit NNDescentBuilder(NNDescentOptions options = {});

  MutableBoundedGraph Build(DistanceComputer& distance, size_t node_count,
                            NNDescentStats* stats = nullptr) const;

  MutableBoundedGraph Build(DistanceComputer& distance, size_t node_count,
                            NNDescentStats* stats,
                            const DistanceFactory& distance_factory) const;

  const NNDescentOptions& Options() const noexcept { return options_; }

 private:
  NNDescentOptions options_;
};

}  // namespace hypervec
