/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>
#include <index/graph/vamana_builder.h>
#include <index/index.h>
#include <quantization/quantizer.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace hypervec {

struct SearchParametersVamana : SearchParameters {
  size_t search_width = 64;
  bool check_relative_distance = true;

  ~SearchParametersVamana() override = default;
};

struct VamanaIndexOptions {
  size_t max_degree = 32;
  size_t build_search_width = 64;
  size_t candidate_pool_size = 200;
  float alpha = 1.2F;
  size_t build_passes = 2;
  uint64_t random_seed = 0x9E3779B97F4A7C15ULL;
  size_t search_width = 64;
  bool check_relative_distance = true;
  /** Build-time scheduling hint; persisted indexes do not need to store it. */
  size_t build_threads = 1;
};

/** Static in-memory Vamana index used to validate graph quality.
 *
 * Build encodes the complete dataset, selects a centroid-nearest navigation
 * point, constructs a Vamana graph, and publishes a fixed-degree layout only
 * after the full pipeline succeeds. The initial implementation accepts L2
 * quantizers; similarity transformations belong to the later DiskANN layer.
 */
class IndexVamana : public Index {
 public:
  IndexVamana(std::unique_ptr<Quantizer> quantizer,
              VamanaIndexOptions options = {});
  ~IndexVamana() override = default;

  IndexVamana(const IndexVamana&) = delete;
  IndexVamana& operator=(const IndexVamana&) = delete;

  IndexCapabilities GetCapabilities() const override;
  void Train(idx_t n, const float* x) override;
  void Build(idx_t n, const float* x) override;
  void Build(idx_t n, const float* x, idx_t n_train_q,
             const float* xq_train) override;
  void Add(idx_t n, const float* x) override;
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Reset() override;
  void Reconstruct(idx_t key, float* recons) const override;
  DistanceComputer* GetDistanceComputer() const override;

  /** Replace vectors and graph after external decoding. */
  void RestoreState(InMemoryCodeStore code_store, FixedDegreeGraph graph,
                    GraphId entry_point);

  const VamanaIndexOptions& Options() const noexcept { return options_; }
  const Quantizer& QuantizerModel() const noexcept { return *quantizer_; }
  const InMemoryCodeStore& CodeStore() const noexcept { return code_store_; }
  const FixedDegreeGraph& Graph() const noexcept { return graph_; }
  GraphId EntryPoint() const noexcept { return entry_point_; }
  const VamanaBuildStats& BuildStats() const noexcept { return build_stats_; }

 private:
  void ValidateAlignedState() const;

  VamanaIndexOptions options_;
  std::unique_ptr<Quantizer> quantizer_;
  InMemoryCodeStore code_store_;
  FixedDegreeGraph graph_;
  VamanaBuilder builder_;
  GraphId entry_point_ = kInvalidGraphId;
  VamanaBuildStats build_stats_;
};

/** Lossless float32 L2 Vamana index. */
class IndexVamanaFlat final : public IndexVamana {
 public:
  IndexVamanaFlat(idx_t dimension, MetricType metric = kMetricL2,
                  VamanaIndexOptions options = {});
};

}  // namespace hypervec
