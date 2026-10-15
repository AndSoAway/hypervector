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

/** Internal node identifier shared by in-memory graph implementations. */
using GraphId = int32_t;
constexpr GraphId kInvalidGraphId = -1;
using GraphNeighborView = std::span<const GraphId>;

/** Read-only adjacency contract consumed by graph search algorithms. */
class GraphStorage {
 public:
  virtual ~GraphStorage() = default;

  virtual size_t NodeCount() const noexcept = 0;
  virtual size_t MaxDegree() const noexcept = 0;
  virtual GraphNeighborView Neighbors(GraphId node) const = 0;

  /** Best-effort prefetch. Invalid node identifiers are ignored. */
  virtual void Prefetch(GraphId node) const noexcept;
};

/** Adjacency contract used by incremental and batch graph builders. */
class MutableGraphStorage : public GraphStorage {
 public:
  virtual void Resize(size_t node_count) = 0;
  virtual void SetNeighbors(GraphId node, GraphNeighborView neighbors) = 0;
};

/** Builder-friendly bounded graph backed by one vector per node. */
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

  /** Add one directed edge. Returns false if the edge already exists. */
  bool AddNeighbor(GraphId node, GraphId neighbor);
  /** Remove one directed edge. Returns false if the edge does not exist. */
  bool RemoveNeighbor(GraphId node, GraphId neighbor);

 private:
  size_t max_degree_;
  std::vector<std::vector<GraphId>> adjacency_;
};

/** Cache-friendly mutable graph with a fixed stride for every node. */
class FixedDegreeGraph final : public MutableGraphStorage {
 public:
  explicit FixedDegreeGraph(size_t max_degree);
  FixedDegreeGraph(size_t node_count, size_t max_degree);
  explicit FixedDegreeGraph(const GraphStorage& graph);

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

/** Immutable compressed-sparse-row layout for static graph indexes. */
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
