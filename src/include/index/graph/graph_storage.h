/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace hypervec {

/** Compact node identifier shared by in-memory and persisted graph layouts.
 *
 * Graph algorithms use 32-bit internal row identifiers. Public/external IDs
 * remain idx_t values and are translated by the owning index or IDMap.
 */
using GraphId = int32_t;
constexpr GraphId kInvalidGraphId = -1;
using GraphNeighborView = std::span<const GraphId>;

/** Read-only adjacency interface consumed by graph search algorithms. */
class GraphStorage {
 public:
  virtual ~GraphStorage() = default;

  virtual size_t NodeCount() const noexcept = 0;
  virtual size_t MaxDegree() const noexcept = 0;
  virtual GraphNeighborView Neighbors(GraphId node) const = 0;

  /** Best-effort prefetch of the adjacency data for node. */
  virtual void Prefetch(GraphId node) const noexcept;
};

/** Mutable adjacency interface used by incremental and batch builders. */
class MutableGraphStorage : public GraphStorage {
 public:
  virtual void Resize(size_t node_count) = 0;
  virtual void SetNeighbors(GraphId node, GraphNeighborView neighbors) = 0;
};

/** Builder-friendly graph backed by one vector per node. */
class MutableBoundedGraph final : public MutableGraphStorage {
 public:
  explicit MutableBoundedGraph(size_t max_degree);
  MutableBoundedGraph(size_t node_count, size_t max_degree);

  size_t NodeCount() const noexcept override;
  size_t MaxDegree() const noexcept override;
  GraphNeighborView Neighbors(GraphId node) const override;
  void Prefetch(GraphId node) const noexcept override;

  void Resize(size_t node_count) override;
  void SetNeighbors(GraphId node, GraphNeighborView neighbors) override;

  /** Add one directed edge. Returns false when it already exists. */
  bool AddNeighbor(GraphId node, GraphId neighbor);
  /** Remove one directed edge. Returns false when it was absent. */
  bool RemoveNeighbor(GraphId node, GraphId neighbor);

 private:
  size_t max_degree_;
  std::vector<std::vector<GraphId>> adjacency_;
};

/** Contiguous fixed-stride graph with an explicit degree per node. */
class FixedDegreeGraph final : public MutableGraphStorage {
 public:
  explicit FixedDegreeGraph(size_t max_degree);
  FixedDegreeGraph(size_t node_count, size_t max_degree);

  size_t NodeCount() const noexcept override;
  size_t MaxDegree() const noexcept override;
  GraphNeighborView Neighbors(GraphId node) const override;
  void Prefetch(GraphId node) const noexcept override;

  void Resize(size_t node_count) override;
  void SetNeighbors(GraphId node, GraphNeighborView neighbors) override;

  const std::vector<GraphId>& Data() const noexcept { return data_; }
  const std::vector<uint32_t>& Degrees() const noexcept { return degrees_; }

 private:
  size_t max_degree_;
  std::vector<GraphId> data_;
  std::vector<uint32_t> degrees_;
};

/** Immutable compressed-sparse-row graph for static indexes. */
class CsrGraph final : public GraphStorage {
 public:
  CsrGraph(std::vector<size_t> offsets, std::vector<GraphId> edges);
  explicit CsrGraph(const GraphStorage& graph);

  size_t NodeCount() const noexcept override;
  size_t MaxDegree() const noexcept override;
  GraphNeighborView Neighbors(GraphId node) const override;
  void Prefetch(GraphId node) const noexcept override;

  const std::vector<size_t>& Offsets() const noexcept { return offsets_; }
  const std::vector<GraphId>& Edges() const noexcept { return edges_; }

 private:
  void ValidateAndSetMaxDegree();

  std::vector<size_t> offsets_;
  std::vector<GraphId> edges_;
  size_t max_degree_ = 0;
};

}  // namespace hypervec
