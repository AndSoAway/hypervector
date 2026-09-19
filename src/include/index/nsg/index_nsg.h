/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>
#include <index/graph/nn_descent_builder.h>
#include <index/graph/nsg_builder.h>
#include <index/index.h>
#include <quantization/quantizer.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace hypervec {

struct SearchParametersNSG : SearchParameters {
  size_t ef_search = 40;
  bool check_relative_distance = true;

  ~SearchParametersNSG() override = default;
};

struct NSGIndexOptions {
  size_t knn_degree = 64;
  size_t nn_descent_iterations = 10;
  double nn_descent_convergence_threshold = 0.001;
  uint64_t random_seed = 0x9E3779B97F4A7C15ULL;
  size_t max_degree = 32;
  size_t build_search_width = 40;
  size_t candidate_pool_size = 200;
  size_t ef_search = 40;
  bool check_relative_distance = true;
};

struct NSGIndexBuildStats {
  NNDescentStats candidate_graph;
  NSGBuildStats nsg;

  void Reset() noexcept;
  void Combine(const NSGIndexBuildStats& other) noexcept;
};

/** Static NSG index composed from the shared graph and quantizer protocols.
 *
 * Build publishes the encoded vectors and final graph together after the
 * complete NN-Descent and NSG pipeline succeeds. Incremental Add is not part
 * of the index contract; Reset followed by Build creates a replacement.
 */
class IndexNSG : public Index {
 public:
  IndexNSG(std::unique_ptr<Quantizer> quantizer, NSGIndexOptions options = {},
           float metric_arg = 0.0F);
  ~IndexNSG() override = default;

  IndexNSG(const IndexNSG&) = delete;
  IndexNSG& operator=(const IndexNSG&) = delete;

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

  const NSGIndexOptions& Options() const noexcept { return options_; }
  const Quantizer& QuantizerModel() const noexcept { return *quantizer_; }
  const InMemoryCodeStore& CodeStore() const noexcept { return code_store_; }
  const GraphStorage& Graph() const noexcept { return graph_; }
  GraphId EntryPoint() const noexcept { return entry_point_; }
  const NSGIndexBuildStats& BuildStats() const noexcept { return build_stats_; }

 private:
  void ValidateAlignedState() const;

  NSGIndexOptions options_;
  std::unique_ptr<Quantizer> quantizer_;
  InMemoryCodeStore code_store_;
  MutableBoundedGraph graph_;
  NNDescentBuilder candidate_builder_;
  NSGBuilder nsg_builder_;
  GraphId entry_point_ = kInvalidGraphId;
  NSGIndexBuildStats build_stats_;
};

/** Lossless float32 NSG index. */
class IndexNSGFlat final : public IndexNSG {
 public:
  IndexNSGFlat(idx_t dimension, MetricType metric = kMetricL2,
               NSGIndexOptions options = {}, float metric_arg = 0.0F);
};

}  // namespace hypervec
