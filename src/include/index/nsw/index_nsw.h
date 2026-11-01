/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/graph/graph_storage.h>
#include <index/graph/nsw_builder.h>
#include <index/index.h>
#include <quantization/quantizer.h>

#include <cstddef>
#include <memory>

namespace hypervec {

struct SearchParametersNSW : SearchParameters {
  size_t ef_search = 16;
  bool check_relative_distance = true;

  ~SearchParametersNSW() override = default;
};

struct NSWIndexOptions {
  size_t max_degree = 32;
  size_t ef_construction = 64;
  size_t ef_search = 16;
  bool check_relative_distance = true;
  bool fill_to_max_degree = true;
};

/** Single-layer NSW index composed from the shared graph and codec protocols.
 *
 * Add is transactional for the index state: vectors and graph edges are built
 * in temporary stores and published together only after the whole batch has
 * succeeded.
 */
class IndexNSW : public Index {
 public:
  IndexNSW(std::unique_ptr<Quantizer> quantizer, NSWIndexOptions options = {},
           float metric_arg = 0.0F);
  ~IndexNSW() override = default;

  IndexNSW(const IndexNSW&) = delete;
  IndexNSW& operator=(const IndexNSW&) = delete;

  IndexCapabilities GetCapabilities() const override;
  void Train(idx_t n, const float* x) override;
  void Add(idx_t n, const float* x) override;
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Reset() override;
  void Reconstruct(idx_t key, float* recons) const override;
  DistanceComputer* GetDistanceComputer() const override;

  /** Replace the stored vectors and graph after external decoding.
   *
   * All inputs are validated before the live index state is changed. This is
   * intended for persistence adapters and other trusted state loaders.
   * Diagnostic build counters restart at zero.
   */
  void RestoreState(InMemoryCodeStore code_store, MutableBoundedGraph graph,
                    GraphId entry_point);

  const NSWIndexOptions& Options() const noexcept { return options_; }
  const Quantizer& QuantizerModel() const noexcept { return *quantizer_; }
  const InMemoryCodeStore& CodeStore() const noexcept { return code_store_; }
  const GraphStorage& Graph() const noexcept { return graph_; }
  GraphId EntryPoint() const noexcept { return entry_point_; }
  const NSWBuildStats& BuildStats() const noexcept { return build_stats_; }

 private:
  void ValidateAlignedState() const;

  NSWIndexOptions options_;
  std::unique_ptr<Quantizer> quantizer_;
  InMemoryCodeStore code_store_;
  MutableBoundedGraph graph_;
  NSWIncrementalBuilder builder_;
  GraphId entry_point_ = kInvalidGraphId;
  NSWBuildStats build_stats_;
};

/** Lossless float32 NSW index. */
class IndexNSWFlat final : public IndexNSW {
 public:
  IndexNSWFlat(idx_t dimension, MetricType metric = kMetricL2,
               NSWIndexOptions options = {}, float metric_arg = 0.0F);
};

}  // namespace hypervec
