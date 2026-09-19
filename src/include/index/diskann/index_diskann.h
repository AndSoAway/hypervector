/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/diskann/diskann_searcher.h>
#include <index/graph/paged_graph_storage.h>
#include <index/graph/vamana_builder.h>
#include <index/index.h>
#include <persistence/page_cache.h>
#include <persistence/random_access_io.h>
#include <quantization/quantizer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace hypervec {

struct SearchParametersDiskANN : SearchParameters {
  size_t search_width = 64;
  bool check_relative_distance = true;

  ~SearchParametersDiskANN() override = default;
};

struct DiskAnnIndexOptions {
  size_t max_degree = 32;
  size_t build_search_width = 64;
  size_t candidate_pool_size = 200;
  float alpha = 1.2F;
  size_t build_passes = 2;
  uint64_t random_seed = 0x9E3779B97F4A7C15ULL;
  size_t search_width = 64;
  bool check_relative_distance = true;
  size_t page_size = 4096;
  size_t cache_capacity_pages = 1024;
};

/** Static DiskANN index composed from Vamana, paged storage, and reranking.
 *
 * Build first validates a full-precision in-memory Vamana graph, then emits
 * page-aligned node records. Build owns an embedded random-access backing
 * store; BuildToFile and RestoreState use caller-owned durable node files.
 */
class IndexDiskANN : public Index {
 public:
  IndexDiskANN(std::unique_ptr<Quantizer> quantizer,
               DiskAnnIndexOptions options = {});
  ~IndexDiskANN() override = default;

  IndexDiskANN(const IndexDiskANN&) = delete;
  IndexDiskANN& operator=(const IndexDiskANN&) = delete;

  IndexCapabilities GetCapabilities() const override;
  void Train(idx_t n, const float* x) override;
  void Build(idx_t n, const float* x) override;
  void Build(idx_t n, const float* x, idx_t n_train_q,
             const float* xq_train) override;
  void BuildToFile(idx_t n, const float* x, const std::string& filename);
  void Add(idx_t n, const float* x) override;
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Reset() override;
  void Reconstruct(idx_t key, float* recons) const override;

  /** Replace live node data and traversal codes after external decoding.
   *
   * The reader is retained through shared ownership. Inputs are validated
   * before live state is changed. Diagnostic I/O counters and the page cache
   * start cold after validation.
   */
  void RestoreState(InMemoryCodeStore code_store,
                    std::shared_ptr<RandomAccessReader> reader,
                    GraphId entry_point);

  const DiskAnnIndexOptions& Options() const noexcept { return options_; }
  const Quantizer& QuantizerModel() const noexcept { return *quantizer_; }
  const InMemoryCodeStore& CodeStore() const noexcept { return code_store_; }
  const DiskAnnNodeLayout* Layout() const noexcept {
    return layout_ ? &*layout_ : nullptr;
  }
  const GraphStorage* Graph() const noexcept { return graph_.get(); }
  GraphId EntryPoint() const noexcept { return entry_point_; }
  const VamanaBuildStats& BuildStats() const noexcept { return build_stats_; }
  PageCacheStats CacheStats() const noexcept;
  RandomAccessReadStats ReadStats() const noexcept;
  void ResetIOStats() const noexcept;

 private:
  void BuildImpl(idx_t n, const float* x, const std::string* filename);
  void ValidateAlignedState() const;

  DiskAnnIndexOptions options_;
  std::unique_ptr<Quantizer> quantizer_;
  InMemoryCodeStore code_store_;
  std::shared_ptr<RandomAccessReader> reader_;
  std::shared_ptr<PageCache> cache_;
  std::unique_ptr<PagedGraphStorage> graph_;
  std::unique_ptr<PagedVectorStorage> raw_vectors_;
  std::optional<DiskAnnNodeLayout> layout_;
  GraphId entry_point_ = kInvalidGraphId;
  VamanaBuildStats build_stats_;
};

/** Lossless traversal-code variant used by the first CPU implementation. */
class IndexDiskANNFlat final : public IndexDiskANN {
 public:
  IndexDiskANNFlat(idx_t dimension, MetricType metric = kMetricL2,
                   DiskAnnIndexOptions options = {});
};

}  // namespace hypervec
