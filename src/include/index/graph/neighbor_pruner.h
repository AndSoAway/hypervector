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
#include <span>
#include <vector>

namespace hypervec {

struct NeighborCandidate {
  GraphId id;
  float distance;
};

struct GraphPruneStats {
  size_t candidates_examined = 0;
  size_t distance_computations = 0;
  size_t accepted = 0;
  size_t rejected = 0;
  size_t filled = 0;

  void Reset() noexcept;
  void Combine(const GraphPruneStats& other) noexcept;
};

class NeighborPruner {
 public:
  virtual ~NeighborPruner() = default;

  virtual std::vector<NeighborCandidate> Prune(
      std::span<const NeighborCandidate> candidates, size_t max_neighbors,
      DistanceComputer& distance, GraphPruneStats* stats = nullptr) const = 0;
};

/** HNSW relative-neighborhood heuristic.
 *
 * Candidates are considered nearest first. A candidate is retained only when
 * it is not closer to an already retained node than to the source query.
 */
class HnswHeuristicPruner final : public NeighborPruner {
 public:
  explicit HnswHeuristicPruner(bool fill_to_capacity = false)
      : fill_to_capacity_(fill_to_capacity) {}

  std::vector<NeighborCandidate> Prune(
      std::span<const NeighborCandidate> candidates, size_t max_neighbors,
      DistanceComputer& distance,
      GraphPruneStats* stats = nullptr) const override;

 private:
  bool fill_to_capacity_;
};

/** Alpha-scaled relative-neighborhood pruning used by Vamana graphs.
 *
 * A candidate is rejected when an already selected neighbor provides an
 * alpha-scaled shorter route. Alpha must be finite and at least one; larger
 * values retain more edges.
 */
class VamanaRobustPruner final : public NeighborPruner {
 public:
  explicit VamanaRobustPruner(float alpha);

  std::vector<NeighborCandidate> Prune(
      std::span<const NeighborCandidate> candidates, size_t max_neighbors,
      DistanceComputer& distance,
      GraphPruneStats* stats = nullptr) const override;

  float Alpha() const noexcept { return alpha_; }

 private:
  float alpha_;
};

}  // namespace hypervec
