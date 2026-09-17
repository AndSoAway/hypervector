/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>

#include <cstddef>
#include <span>
#include <vector>

namespace hypervec {

struct DistanceComputer;

struct NeighborCandidate {
  GraphId id = kInvalidGraphId;
  float distance = 0.0f;
};

struct GraphPruneStats {
  size_t candidates_examined = 0;
  size_t distance_computations = 0;
  size_t accepted = 0;
  size_t rejected = 0;

  void Reset() noexcept { *this = {}; }
  void Combine(const GraphPruneStats& other) noexcept;
};

/** Candidate reduction strategy used by graph builders. */
class NeighborPruner {
 public:
  virtual ~NeighborPruner() = default;

  /** Candidates must be ordered from nearest to farthest. */
  virtual std::vector<NeighborCandidate> Prune(
      std::span<const NeighborCandidate> candidates, size_t max_degree,
      DistanceComputer& distance, GraphPruneStats* stats = nullptr) const = 0;
};

/** Keep the closest candidates up to the requested degree.
 *
 * This is the connection rule used by the incremental NSW builder. It does
 * not perform extra pairwise distance computations.
 */
class NearestNeighborPruner final : public NeighborPruner {
 public:
  std::vector<NeighborCandidate> Prune(
      std::span<const NeighborCandidate> candidates, size_t max_degree,
      DistanceComputer& distance,
      GraphPruneStats* stats = nullptr) const override;
};

/** HNSW diversity heuristic extracted from HNSW::ShrinkNeighborList. */
class HnswHeuristicPruner final : public NeighborPruner {
 public:
  explicit HnswHeuristicPruner(bool fill_to_max_degree = false)
      : fill_to_max_degree_(fill_to_max_degree) {}

  std::vector<NeighborCandidate> Prune(
      std::span<const NeighborCandidate> candidates, size_t max_degree,
      DistanceComputer& distance,
      GraphPruneStats* stats = nullptr) const override;

 private:
  bool fill_to_max_degree_;
};

}  // namespace hypervec
