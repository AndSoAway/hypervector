/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <index/graph/graph_validation.h>
#include <index/graph/visited_table.h>
#include <index/vamana/index_vamana.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

size_t QuantizerCodeSize(const std::unique_ptr<Quantizer>& quantizer) {
  HYPERVEC_THROW_IF_NOT_MSG(quantizer != nullptr,
                            "IndexVamana: quantizer must not be null");
  return quantizer->CodeSize();
}

VamanaBuildOptions GraphBuildOptions(const VamanaIndexOptions& options) {
  VamanaBuildOptions builder_options;
  builder_options.max_degree = options.max_degree;
  builder_options.search_width = options.build_search_width;
  builder_options.candidate_pool_size = options.candidate_pool_size;
  builder_options.alpha = options.alpha;
  builder_options.build_passes = options.build_passes;
  builder_options.build_threads = options.build_threads;
  builder_options.random_seed = options.random_seed;
  return builder_options;
}

std::unique_ptr<Quantizer> MakeFlat(idx_t dimension, MetricType metric) {
  return std::make_unique<FlatQuantizer>(dimension, metric);
}

GraphId SelectNavigationPoint(idx_t count, idx_t dimension, const float* data,
                              DistanceComputer* distance) {
  std::vector<double> sums(static_cast<size_t>(dimension), 0.0);
  for (idx_t node = 0; node < count; ++node) {
    for (idx_t component = 0; component < dimension; ++component) {
      sums[static_cast<size_t>(component)] +=
          data[static_cast<size_t>(node) * static_cast<size_t>(dimension) +
               static_cast<size_t>(component)];
    }
  }
  std::vector<float> centroid(static_cast<size_t>(dimension));
  for (idx_t component = 0; component < dimension; ++component) {
    centroid[static_cast<size_t>(component)] = static_cast<float>(
        sums[static_cast<size_t>(component)] / static_cast<double>(count));
  }

  distance->SetQuery(centroid.data());
  GraphId best = kInvalidGraphId;
  float best_distance = (std::numeric_limits<float>::infinity)();
  for (idx_t node = 0; node < count; ++node) {
    const float candidate_distance = (*distance)(node);
    HYPERVEC_THROW_IF_NOT_MSG(
        !std::isnan(candidate_distance),
        "IndexVamana::Build: navigation distance returned NaN");
    if (best == kInvalidGraphId || candidate_distance < best_distance ||
        (candidate_distance == best_distance && node < best)) {
      best = static_cast<GraphId>(node);
      best_distance = candidate_distance;
    }
  }
  return best;
}

}  // namespace

IndexVamana::IndexVamana(std::unique_ptr<Quantizer> quantizer,
                         VamanaIndexOptions options)
    : Index(),
      options_(options),
      quantizer_(std::move(quantizer)),
      code_store_(QuantizerCodeSize(quantizer_)),
      graph_(options_.max_degree),
      builder_(GraphBuildOptions(options_)) {
  HYPERVEC_THROW_IF_NOT_FMT(
      quantizer_->Dimension() > 0 &&
          quantizer_->Dimension() <= std::numeric_limits<int>::max(),
      "IndexVamana: dimension must be in [1, %d]",
      std::numeric_limits<int>::max());
  HYPERVEC_THROW_IF_NOT_MSG(quantizer_->Metric() == kMetricL2,
                            "IndexVamana: only kMetricL2 is supported");
  HYPERVEC_THROW_IF_NOT_MSG(options_.search_width > 0,
                            "IndexVamana: search_width must be positive");

  d = static_cast<int>(quantizer_->Dimension());
  metric_type = quantizer_->Metric();
  is_trained = quantizer_->IsTrained();
}

IndexCapabilities IndexVamana::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.requires_training = quantizer_->NeedsTraining();
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexVamana::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0,
      "IndexVamana::Train: a populated index cannot be retrained");
  quantizer_->Train(n, x);
  is_trained = quantizer_->IsTrained();
  HYPERVEC_THROW_IF_NOT_MSG(
      is_trained, "IndexVamana::Train: quantizer did not become trained");
}

void IndexVamana::Build(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(n_total == 0,
                            "IndexVamana::Build: index must be empty");
  HYPERVEC_THROW_IF_NOT_FMT(
      n > 0, "IndexVamana::Build: n must be positive, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr,
                            "IndexVamana::Build: x must not be null");
  constexpr idx_t kMaxNodeCount =
      static_cast<idx_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(
      n <= kMaxNodeCount,
      "IndexVamana::Build: vector count exceeds GraphId capacity");

  if (!quantizer_->IsTrained()) {
    Train(n, x);
  }
  HYPERVEC_THROW_IF_NOT_MSG(is_trained && quantizer_->IsTrained(),
                            "IndexVamana::Build: quantizer is not trained");

  const size_t encoded_size =
      mul_no_overflow(static_cast<size_t>(n), quantizer_->CodeSize(),
                      "IndexVamana::Build encoded dataset size");
  std::vector<uint8_t> encoded(encoded_size);
  quantizer_->Encode(n, x, encoded.data());
  InMemoryCodeStore staged_store(quantizer_->CodeSize());
  staged_store.Append(n, encoded.data());
  std::unique_ptr<DistanceComputer> distance =
      quantizer_->CreateDistanceComputer(staged_store.View());
  const GraphId staged_entry_point =
      SelectNavigationPoint(n, d, x, distance.get());
  VamanaBuildStats staged_stats;
  MutableBoundedGraph staged_mutable = builder_.Build(
      *distance, static_cast<size_t>(n), staged_entry_point, &staged_stats,
      [this, &staged_store] {
        return quantizer_->CreateDistanceComputer(staged_store.View());
      });
  FixedDegreeGraph staged_graph(staged_mutable);
  const GraphValidationReport report =
      ValidateGraph(staged_graph, staged_entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(n),
      "IndexVamana::Build: final graph is invalid or unreachable");

  code_store_ = std::move(staged_store);
  graph_ = std::move(staged_graph);
  entry_point_ = staged_entry_point;
  build_stats_ = staged_stats;
  n_total = n;
  ValidateAlignedState();
}

void IndexVamana::Build(idx_t n, const float* x, idx_t n_train_q,
                        const float* xq_train) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q >= 0,
      "IndexVamana::Build: query training count must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q == 0 || xq_train != nullptr,
      "IndexVamana::Build: query training data must not be null");
  Build(n, x);
}

void IndexVamana::Add(idx_t /*n*/, const float* /*x*/) {
  HYPERVEC_THROW_MSG(
      "IndexVamana::Add: incremental insertion is not supported; use Build");
}

void IndexVamana::Search(idx_t n, const float* x, idx_t k, float* distances,
                         idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexVamana::Search: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexVamana::Search: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexVamana::Search: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexVamana::Search: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexVamana::Search: labels must not be null when n is positive");
  ValidateAlignedState();
  if (n == 0) {
    return;
  }

  const size_t output_size =
      mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                      "IndexVamana::Search output size");
  std::fill_n(distances, output_size, (std::numeric_limits<float>::infinity)());
  std::fill_n(labels, output_size, static_cast<idx_t>(-1));
  if (n_total == 0) {
    return;
  }

  size_t search_width = options_.search_width;
  bool check_relative_distance = options_.check_relative_distance;
  if (const auto* vamana_params =
          dynamic_cast<const SearchParametersVamana*>(params)) {
    search_width = vamana_params->search_width;
    check_relative_distance = vamana_params->check_relative_distance;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      search_width > 0, "IndexVamana::Search: search_width must be positive");
  search_width = std::max(search_width, static_cast<size_t>(k));

  std::unique_ptr<DistanceComputer> distance =
      quantizer_->CreateDistanceComputer(code_store_.View());
  const GraphSearcher searcher(graph_);
  VisitedTable visited(graph_.NodeCount(), false);
  const GraphId entry_points[] = {entry_point_};
  const IDSelector* selector = params == nullptr ? nullptr : params->sel;
  for (idx_t query = 0; query < n; ++query) {
    distance->SetQuery(x + static_cast<size_t>(query) * d);
    const std::vector<GraphSearchResult> results = searcher.Search(
        *distance, entry_points,
        GraphSearchOptions{search_width, check_relative_distance, selector},
        &visited);
    const size_t result_count =
        std::min(results.size(), static_cast<size_t>(k));
    const size_t output_offset =
        static_cast<size_t>(query) * static_cast<size_t>(k);
    for (size_t result = 0; result < result_count; ++result) {
      distances[output_offset + result] = results[result].distance;
      labels[output_offset + result] = results[result].id;
    }
  }
}

void IndexVamana::Reset() {
  code_store_.Reset();
  graph_.Resize(0);
  entry_point_ = kInvalidGraphId;
  n_total = 0;
  build_stats_.Reset();
}

void IndexVamana::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      key >= 0 && key < n_total,
      "IndexVamana::Reconstruct: key is outside [0, n_total)");
  HYPERVEC_THROW_IF_NOT_MSG(
      recons != nullptr, "IndexVamana::Reconstruct: output must not be null");
  quantizer_->Decode(1, code_store_.Code(key), recons);
}

DistanceComputer* IndexVamana::GetDistanceComputer() const {
  return quantizer_->CreateDistanceComputer(code_store_.View()).release();
}

void IndexVamana::RestoreState(InMemoryCodeStore code_store,
                               FixedDegreeGraph graph, GraphId entry_point) {
  const idx_t restored_total = code_store.Size();
  HYPERVEC_THROW_IF_NOT_MSG(
      code_store.CodeSize() == quantizer_->CodeSize(),
      "IndexVamana::RestoreState: code size does not match the quantizer");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph.MaxDegree() == options_.max_degree,
      "IndexVamana::RestoreState: graph max degree does not match the options");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph.NodeCount() == static_cast<size_t>(restored_total),
      "IndexVamana::RestoreState: graph and code counts do not match");
  HYPERVEC_THROW_IF_NOT_MSG(
      (restored_total == 0 && entry_point == kInvalidGraphId) ||
          (restored_total > 0 && entry_point >= 0 &&
           entry_point < restored_total),
      "IndexVamana::RestoreState: entry point is inconsistent with the index");
  const GraphValidationReport report = ValidateGraph(graph, entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(restored_total),
      "IndexVamana::RestoreState: graph is invalid or unreachable");

  code_store_ = std::move(code_store);
  graph_ = std::move(graph);
  entry_point_ = entry_point;
  build_stats_.Reset();
  n_total = restored_total;
  ValidateAlignedState();
}

void IndexVamana::ValidateAlignedState() const {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total >= 0 && code_store_.Size() == n_total &&
          graph_.NodeCount() == static_cast<size_t>(n_total),
      "IndexVamana: code store, graph, and index counts are inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      (n_total == 0 && entry_point_ == kInvalidGraphId) ||
          (n_total > 0 && entry_point_ >= 0 && entry_point_ < n_total),
      "IndexVamana: entry point is inconsistent with the index size");
}

IndexVamanaFlat::IndexVamanaFlat(idx_t dimension, MetricType metric,
                                 VamanaIndexOptions options)
    : IndexVamana(MakeFlat(dimension, metric), options) {}

}  // namespace hypervec
