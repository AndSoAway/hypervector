/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/diskann/index_diskann.h>
#include <index/graph/graph_validation.h>
#include <index/vamana/index_vamana.h>
#include <persistence/io.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

size_t QuantizerCodeSize(const std::unique_ptr<Quantizer>& quantizer) {
  HYPERVEC_THROW_IF_NOT_MSG(quantizer != nullptr,
                            "IndexDiskANN: quantizer must not be null");
  return quantizer->CodeSize();
}

VamanaIndexOptions GraphBuildOptions(const DiskAnnIndexOptions& options) {
  VamanaIndexOptions graph_options;
  graph_options.max_degree = options.max_degree;
  graph_options.build_search_width = options.build_search_width;
  graph_options.candidate_pool_size = options.candidate_pool_size;
  graph_options.alpha = options.alpha;
  graph_options.build_passes = options.build_passes;
  graph_options.random_seed = options.random_seed;
  graph_options.search_width = options.search_width;
  graph_options.check_relative_distance = options.check_relative_distance;
  return graph_options;
}

std::unique_ptr<Quantizer> MakeFlat(idx_t dimension, MetricType metric) {
  return std::make_unique<FlatQuantizer>(dimension, metric);
}

}  // namespace

IndexDiskANN::IndexDiskANN(std::unique_ptr<Quantizer> quantizer,
                           DiskAnnIndexOptions options)
    : Index(),
      options_(options),
      quantizer_(std::move(quantizer)),
      code_store_(QuantizerCodeSize(quantizer_)) {
  HYPERVEC_THROW_IF_NOT_FMT(
      quantizer_->Dimension() > 0 &&
          quantizer_->Dimension() <= std::numeric_limits<int>::max(),
      "IndexDiskANN: dimension must be in [1, %d]",
      std::numeric_limits<int>::max());
  HYPERVEC_THROW_IF_NOT_MSG(quantizer_->Metric() == kMetricL2,
                            "IndexDiskANN: only kMetricL2 is supported");
  HYPERVEC_THROW_IF_NOT_MSG(options_.search_width > 0,
                            "IndexDiskANN: search_width must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.cache_capacity_pages > 0,
      "IndexDiskANN: cache_capacity_pages must be positive");
  const DiskAnnNodeLayout validated_layout(
      0, static_cast<size_t>(quantizer_->Dimension()), options_.max_degree,
      options_.page_size);
  (void)validated_layout;
  const IndexVamanaFlat validated_builder(quantizer_->Dimension(), kMetricL2,
                                          GraphBuildOptions(options_));
  (void)validated_builder;

  d = static_cast<int>(quantizer_->Dimension());
  metric_type = quantizer_->Metric();
  is_trained = quantizer_->IsTrained();
}

IndexCapabilities IndexDiskANN::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.requires_training = quantizer_->NeedsTraining();
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexDiskANN::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0,
      "IndexDiskANN::Train: a populated index cannot be retrained");
  quantizer_->Train(n, x);
  is_trained = quantizer_->IsTrained();
  HYPERVEC_THROW_IF_NOT_MSG(
      is_trained, "IndexDiskANN::Train: quantizer did not become trained");
}

void IndexDiskANN::Build(idx_t n, const float* x) {
  const std::string* filename =
      options_.node_data_path.empty() ? nullptr : &options_.node_data_path;
  BuildImpl(n, x, filename);
}

void IndexDiskANN::BuildToFile(idx_t n, const float* x,
                               const std::string& filename) {
  HYPERVEC_THROW_IF_NOT_MSG(
      !filename.empty(),
      "IndexDiskANN::BuildToFile: filename must not be empty");
  BuildImpl(n, x, &filename);
}

void IndexDiskANN::BuildImpl(idx_t n, const float* x,
                             const std::string* filename) {
  HYPERVEC_THROW_IF_NOT_MSG(n_total == 0,
                            "IndexDiskANN::Build: index must be empty");
  HYPERVEC_THROW_IF_NOT_FMT(
      n > 0, "IndexDiskANN::Build: n must be positive, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr,
                            "IndexDiskANN::Build: x must not be null");
  constexpr idx_t kMaxNodeCount =
      static_cast<idx_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(
      n <= kMaxNodeCount,
      "IndexDiskANN::Build: vector count exceeds GraphId capacity");

  IndexVamanaFlat scaffold(d, kMetricL2, GraphBuildOptions(options_));
  scaffold.Build(n, x);
  if (!quantizer_->IsTrained()) {
    Train(n, x);
  }
  HYPERVEC_THROW_IF_NOT_MSG(is_trained && quantizer_->IsTrained(),
                            "IndexDiskANN::Build: quantizer is not trained");

  const size_t encoded_size =
      mul_no_overflow(static_cast<size_t>(n), quantizer_->CodeSize(),
                      "IndexDiskANN::Build encoded dataset size");
  std::vector<uint8_t> encoded(encoded_size);
  quantizer_->Encode(n, x, encoded.data());
  InMemoryCodeStore staged_codes(quantizer_->CodeSize());
  staged_codes.Append(n, encoded.data());

  DiskAnnNodeLayout staged_layout(static_cast<size_t>(n),
                                  static_cast<size_t>(d), options_.max_degree,
                                  options_.page_size);
  std::shared_ptr<RandomAccessReader> staged_reader;
  if (filename == nullptr) {
    VectorIOWriter writer;
    WriteDiskAnnNodes(staged_layout, x, scaffold.Graph(), &writer);
    staged_reader =
        std::make_shared<VectorRandomAccessReader>(std::move(writer.data));
  } else {
    WriteDiskAnnNodesToFile(staged_layout, x, scaffold.Graph(), *filename);
    staged_reader = std::make_shared<FileRandomAccessReader>(*filename);
  }
  auto staged_cache = std::make_shared<PageCache>(
      staged_reader, options_.page_size, options_.cache_capacity_pages);
  auto staged_graph = std::make_unique<PagedGraphStorage>(
      staged_cache, static_cast<size_t>(n), options_.max_degree,
      staged_layout.RecordSize(), staged_layout.DegreeOffset());
  auto staged_vectors =
      std::make_unique<PagedVectorStorage>(staged_cache, staged_layout);
  const DiskAnnSearcher validated_searcher(*staged_graph, *staged_vectors,
                                           *quantizer_, staged_codes.View(),
                                           scaffold.EntryPoint());
  (void)validated_searcher;

  code_store_ = std::move(staged_codes);
  reader_ = std::move(staged_reader);
  cache_ = std::move(staged_cache);
  graph_ = std::move(staged_graph);
  raw_vectors_ = std::move(staged_vectors);
  layout_ = staged_layout;
  entry_point_ = scaffold.EntryPoint();
  build_stats_ = scaffold.BuildStats();
  n_total = n;
  ValidateAlignedState();
}

void IndexDiskANN::Build(idx_t n, const float* x, idx_t n_train_q,
                         const float* xq_train) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q >= 0,
      "IndexDiskANN::Build: query training count must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q == 0 || xq_train != nullptr,
      "IndexDiskANN::Build: query training data must not be null");
  Build(n, x);
}

void IndexDiskANN::Add(idx_t /*n*/, const float* /*x*/) {
  HYPERVEC_THROW_MSG(
      "IndexDiskANN::Add: incremental insertion is not supported; use Build");
}

void IndexDiskANN::Search(idx_t n, const float* x, idx_t k, float* distances,
                          idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexDiskANN::Search: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexDiskANN::Search: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexDiskANN::Search: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexDiskANN::Search: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexDiskANN::Search: labels must not be null when n is positive");
  ValidateAlignedState();
  if (n == 0) {
    return;
  }

  const size_t output_size =
      mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                      "IndexDiskANN::Search output size");
  std::fill_n(distances, output_size, (std::numeric_limits<float>::infinity)());
  std::fill_n(labels, output_size, static_cast<idx_t>(-1));
  if (n_total == 0) {
    return;
  }

  size_t search_width = options_.search_width;
  bool check_relative_distance = options_.check_relative_distance;
  if (const auto* diskann_params =
          dynamic_cast<const SearchParametersDiskANN*>(params)) {
    search_width = diskann_params->search_width;
    check_relative_distance = diskann_params->check_relative_distance;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      search_width > 0, "IndexDiskANN::Search: search_width must be positive");
  const IDSelector* selector = params == nullptr ? nullptr : params->sel;
  const DiskAnnSearcher searcher(*graph_, *raw_vectors_, *quantizer_,
                                 code_store_.View(), entry_point_);
  for (idx_t query = 0; query < n; ++query) {
    const std::vector<GraphSearchResult> results = searcher.Search(
        x + static_cast<size_t>(query) * static_cast<size_t>(d),
        static_cast<size_t>(k),
        DiskAnnSearchOptions{search_width, check_relative_distance, selector});
    const size_t output_offset =
        static_cast<size_t>(query) * static_cast<size_t>(k);
    for (size_t result = 0; result < results.size(); ++result) {
      distances[output_offset + result] = results[result].distance;
      labels[output_offset + result] = results[result].id;
    }
  }
}

void IndexDiskANN::Reset() {
  code_store_.Reset();
  raw_vectors_.reset();
  graph_.reset();
  cache_.reset();
  reader_.reset();
  layout_.reset();
  entry_point_ = kInvalidGraphId;
  build_stats_.Reset();
  n_total = 0;
}

void IndexDiskANN::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      key >= 0 && key < n_total,
      "IndexDiskANN::Reconstruct: key is outside [0, n_total)");
  HYPERVEC_THROW_IF_NOT_MSG(
      recons != nullptr, "IndexDiskANN::Reconstruct: output must not be null");
  ValidateAlignedState();
  const DiskAnnVectorView vector =
      raw_vectors_->Vector(static_cast<GraphId>(key));
  std::memcpy(recons, vector.data(), vector.size() * sizeof(float));
}

void IndexDiskANN::RestoreState(InMemoryCodeStore code_store,
                                std::shared_ptr<RandomAccessReader> reader,
                                GraphId entry_point) {
  const idx_t restored_total = code_store.Size();
  HYPERVEC_THROW_IF_NOT_MSG(
      code_store.CodeSize() == quantizer_->CodeSize(),
      "IndexDiskANN::RestoreState: code size does not match the quantizer");
  HYPERVEC_THROW_IF_NOT_MSG(
      quantizer_->IsTrained(),
      "IndexDiskANN::RestoreState: quantizer must be trained");
  if (restored_total == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        reader == nullptr && entry_point == kInvalidGraphId,
        "IndexDiskANN::RestoreState: empty state requires no reader or entry "
        "point");
    Reset();
    code_store_ = std::move(code_store);
    return;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      reader != nullptr,
      "IndexDiskANN::RestoreState: node reader must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      entry_point >= 0 && entry_point < restored_total,
      "IndexDiskANN::RestoreState: entry point is outside the index");

  DiskAnnNodeLayout staged_layout(static_cast<size_t>(restored_total),
                                  static_cast<size_t>(d), options_.max_degree,
                                  options_.page_size);
  auto staged_cache = std::make_shared<PageCache>(
      reader, options_.page_size, options_.cache_capacity_pages);
  auto staged_graph = std::make_unique<PagedGraphStorage>(
      staged_cache, static_cast<size_t>(restored_total), options_.max_degree,
      staged_layout.RecordSize(), staged_layout.DegreeOffset());
  auto staged_vectors =
      std::make_unique<PagedVectorStorage>(staged_cache, staged_layout);
  const GraphValidationReport report =
      ValidateGraph(*staged_graph, entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(restored_total),
      "IndexDiskANN::RestoreState: graph is invalid or unreachable");
  const DiskAnnSearcher validated_searcher(*staged_graph, *staged_vectors,
                                           *quantizer_, code_store.View(),
                                           entry_point);
  (void)validated_searcher;
  staged_cache->Clear();
  staged_cache->ResetStats();
  reader->ResetStats();

  code_store_ = std::move(code_store);
  reader_ = std::move(reader);
  cache_ = std::move(staged_cache);
  graph_ = std::move(staged_graph);
  raw_vectors_ = std::move(staged_vectors);
  layout_ = staged_layout;
  entry_point_ = entry_point;
  build_stats_.Reset();
  n_total = restored_total;
  is_trained = quantizer_->IsTrained();
  ValidateAlignedState();
}

PageCacheStats IndexDiskANN::CacheStats() const noexcept {
  return cache_ == nullptr ? PageCacheStats{} : cache_->Stats();
}

RandomAccessReadStats IndexDiskANN::ReadStats() const noexcept {
  return reader_ == nullptr ? RandomAccessReadStats{} : reader_->Stats();
}

void IndexDiskANN::ResetIOStats() const noexcept {
  if (cache_ != nullptr) {
    cache_->ResetStats();
  }
  if (reader_ != nullptr) {
    reader_->ResetStats();
  }
}

void IndexDiskANN::ValidateAlignedState() const {
  const bool empty = n_total == 0;
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total >= 0 && code_store_.Size() == n_total &&
          (empty == (reader_ == nullptr)) && (empty == (cache_ == nullptr)) &&
          (empty == (graph_ == nullptr)) &&
          (empty == (raw_vectors_ == nullptr)) &&
          (empty == !layout_.has_value()),
      "IndexDiskANN: storage components are inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      empty || (graph_->NodeCount() == static_cast<size_t>(n_total) &&
                raw_vectors_->NodeCount() == static_cast<size_t>(n_total) &&
                layout_->NodeCount() == static_cast<size_t>(n_total)),
      "IndexDiskANN: storage counts are inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      (empty && entry_point_ == kInvalidGraphId) ||
          (!empty && entry_point_ >= 0 && entry_point_ < n_total),
      "IndexDiskANN: entry point is inconsistent with the index size");
}

IndexDiskANNFlat::IndexDiskANNFlat(idx_t dimension, MetricType metric,
                                   DiskAnnIndexOptions options)
    : IndexDiskANN(MakeFlat(dimension, metric), options) {}

}  // namespace hypervec
