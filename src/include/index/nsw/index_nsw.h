/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/flat/index_flat.h>
#include <index/graph/graph_storage.h>
#include <index/index.h>

namespace hypervec {

/** Per-query controls for NSW graph traversal. */
struct SearchParametersNSW : SearchParameters {
  int ef_search = 16;

  ~SearchParametersNSW() override = default;
};

/** Single-layer navigable small-world graph over exact float vectors. */
struct IndexNSWFlat : Index {
  int graph_degree = 32;
  int ef_construction = 40;
  int ef_search = 16;
  GraphId entry_point = kInvalidGraphId;
  IndexFlat storage;
  MutableBoundedGraph graph;

  IndexNSWFlat();
  explicit IndexNSWFlat(int d, int graph_degree = 32,
                        MetricType metric = kMetricL2);

  std::string_view TypeName() const noexcept override { return "nsw_flat"; }

  IndexCapabilities Capabilities() const noexcept override;

  void Add(idx_t n, const float* x) override;
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Search1(const float* x, ResultHandler& handler,
               SearchParameters* params = nullptr) const override;
  void Reset() override;
  void Reconstruct(idx_t key, float* recons) const override;
  DistanceComputer* GetDistanceComputer() const override;

  /** Validate wrapper, vector storage, and graph invariants. */
  void CheckConsistency() const;
};

}  // namespace hypervec
