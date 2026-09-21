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
#include <index/nsg/index_nsg.h>
#include <utils/distances/distance_computer.h>
#include <utils/distances/metric_type.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

size_t QuantizerCodeSize(const std::unique_ptr<Quantizer>& quantizer) {
  HYPERVEC_THROW_IF_NOT_MSG(quantizer != nullptr,
                            "IndexNSG: quantizer must not be null");
  return quantizer->CodeSize();
}

size_t InitialGraphCapacity(const NSGIndexOptions& options) {
  return add_no_overflow(options.max_degree, size_t{1},
                         "IndexNSG graph capacity");
}

size_t GraphCapacity(const NSGIndexOptions& options, size_t node_count) {
  if (node_count == 0) {
    return InitialGraphCapacity(options);
  }
  if (node_count == 1) {
    return 1;
  }
  return std::min(node_count - 1, InitialGraphCapacity(options));
}

NNDescentOptions CandidateBuildOptions(const NSGIndexOptions& options) {
  NNDescentOptions builder_options;
  builder_options.max_degree = options.knn_degree;
  builder_options.max_iterations = options.nn_descent_iterations;
  builder_options.build_threads = options.build_threads;
  builder_options.convergence_threshold =
      options.nn_descent_convergence_threshold;
  builder_options.random_seed = options.random_seed;
  return builder_options;
}

NSGBuildOptions GraphBuildOptions(const NSGIndexOptions& options) {
  NSGBuildOptions builder_options;
  builder_options.max_degree = options.max_degree;
  builder_options.build_threads = options.build_threads;
  builder_options.search_width = options.build_search_width;
  builder_options.candidate_pool_size = options.candidate_pool_size;
  builder_options.check_relative_distance = options.check_relative_distance;
  return builder_options;
}

std::unique_ptr<DistanceComputer> MakeTraversalDistance(
    const Quantizer& quantizer, EncodedVectorView store) {
  std::unique_ptr<DistanceComputer> distance =
      quantizer.CreateDistanceComputer(store);
  if (IsSimilarityMetric(quantizer.Metric())) {
    return std::make_unique<NegativeDistanceComputer>(distance.release());
  }
  return distance;
}

float ExternalDistance(float traversal_distance, MetricType metric) {
  return IsSimilarityMetric(metric) ? -traversal_distance : traversal_distance;
}

std::unique_ptr<Quantizer> MakeFlat(idx_t dimension, MetricType metric,
                                    float metric_arg) {
  return std::make_unique<FlatQuantizer>(dimension, metric, metric_arg);
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
        "IndexNSG::Build: navigation distance returned NaN");
    if (best == kInvalidGraphId || candidate_distance < best_distance ||
        (candidate_distance == best_distance && node < best)) {
      best = static_cast<GraphId>(node);
      best_distance = candidate_distance;
    }
  }
  return best;
}

}  // namespace

void NSGIndexBuildStats::Reset() noexcept { *this = {}; }

void NSGIndexBuildStats::Combine(const NSGIndexBuildStats& other) noexcept {
  candidate_graph.Combine(other.candidate_graph);
  nsg.Combine(other.nsg);
}

IndexNSG::IndexNSG(std::unique_ptr<Quantizer> quantizer,
                   NSGIndexOptions options, float metric_arg)
    : Index(),
      options_(options),
      quantizer_(std::move(quantizer)),
      code_store_(QuantizerCodeSize(quantizer_)),
      graph_(InitialGraphCapacity(options_)),
      candidate_builder_(CandidateBuildOptions(options_)),
      nsg_builder_(GraphBuildOptions(options_)) {
  HYPERVEC_THROW_IF_NOT_FMT(
      quantizer_->Dimension() > 0 &&
          quantizer_->Dimension() <= std::numeric_limits<int>::max(),
      "IndexNSG: dimension must be in [1, %d]",
      std::numeric_limits<int>::max());
  HYPERVEC_THROW_IF_NOT_MSG(options_.ef_search > 0,
                            "IndexNSG: ef_search must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      quantizer_->Metric() != kMetricLp ||
          (std::isfinite(metric_arg) && metric_arg > 0.0F),
      "IndexNSG: kMetricLp requires a finite, positive metric_arg");

  d = static_cast<int>(quantizer_->Dimension());
  metric_type = quantizer_->Metric();
  this->metric_arg = metric_arg;
  is_trained = quantizer_->IsTrained();
}

IndexCapabilities IndexNSG::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.requires_training = quantizer_->NeedsTraining();
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexNSG::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0, "IndexNSG::Train: a populated index cannot be retrained");
  quantizer_->Train(n, x);
  is_trained = quantizer_->IsTrained();
  HYPERVEC_THROW_IF_NOT_MSG(
      is_trained, "IndexNSG::Train: quantizer did not become trained");
}

void IndexNSG::Build(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(n_total == 0,
                            "IndexNSG::Build: index must be empty");
  HYPERVEC_THROW_IF_NOT_FMT(n > 0,
                            "IndexNSG::Build: n must be positive, got %" PRId64,
                            static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr,
                            "IndexNSG::Build: x must not be null");
  constexpr idx_t kMaxNodeCount =
      static_cast<idx_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(n <= kMaxNodeCount,
                            "IndexNSG::Build: vector count exceeds GraphId "
                            "capacity");

  if (!quantizer_->IsTrained()) {
    Train(n, x);
  }
  HYPERVEC_THROW_IF_NOT_MSG(is_trained && quantizer_->IsTrained(),
                            "IndexNSG::Build: quantizer is not trained");

  const size_t encoded_size =
      mul_no_overflow(static_cast<size_t>(n), quantizer_->CodeSize(),
                      "IndexNSG::Build encoded dataset size");
  std::vector<uint8_t> encoded(encoded_size);
  quantizer_->Encode(n, x, encoded.data());
  InMemoryCodeStore staged_store(quantizer_->CodeSize());
  staged_store.Append(n, encoded.data());

  std::unique_ptr<DistanceComputer> distance =
      MakeTraversalDistance(*quantizer_, staged_store.View());
  NSGIndexBuildStats staged_stats;
  MutableBoundedGraph candidate_graph = candidate_builder_.Build(
      *distance, static_cast<size_t>(n), &staged_stats.candidate_graph,
      [this, &staged_store] {
        return MakeTraversalDistance(*quantizer_, staged_store.View());
      });
  const GraphId staged_entry_point =
      SelectNavigationPoint(n, d, x, distance.get());
  MutableBoundedGraph staged_graph = nsg_builder_.Build(
      candidate_graph, *distance, staged_entry_point, &staged_stats.nsg,
      [this, &staged_store] {
        return MakeTraversalDistance(*quantizer_, staged_store.View());
      });
  const GraphValidationReport report =
      ValidateGraph(staged_graph, staged_entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(n),
      "IndexNSG::Build: final graph is invalid or unreachable");

  code_store_ = std::move(staged_store);
  graph_ = std::move(staged_graph);
  entry_point_ = staged_entry_point;
  build_stats_ = staged_stats;
  n_total = n;
  ValidateAlignedState();
}

void IndexNSG::Build(idx_t n, const float* x, idx_t n_train_q,
                     const float* xq_train) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q >= 0,
      "IndexNSG::Build: query training count must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      n_train_q == 0 || xq_train != nullptr,
      "IndexNSG::Build: query training data must not be null");
  Build(n, x);
}

void IndexNSG::Add(idx_t /*n*/, const float* /*x*/) {
  HYPERVEC_THROW_MSG(
      "IndexNSG::Add: incremental insertion is not supported; use Build");
}

void IndexNSG::Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexNSG::Search: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexNSG::Search: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexNSG::Search: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexNSG::Search: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexNSG::Search: labels must not be null when n is positive");
  ValidateAlignedState();
  if (n == 0) {
    return;
  }

  const bool similarity = IsSimilarityMetric(metric_type);
  const float padding = similarity ? -(std::numeric_limits<float>::infinity)()
                                   : (std::numeric_limits<float>::infinity)();
  const size_t output_size =
      mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                      "IndexNSG::Search output size");
  std::fill_n(distances, output_size, padding);
  std::fill_n(labels, output_size, static_cast<idx_t>(-1));
  if (n_total == 0) {
    return;
  }

  size_t ef_search = options_.ef_search;
  bool check_relative_distance = options_.check_relative_distance;
  if (const auto* nsg_params =
          dynamic_cast<const SearchParametersNSG*>(params)) {
    ef_search = nsg_params->ef_search;
    check_relative_distance = nsg_params->check_relative_distance;
  }
  HYPERVEC_THROW_IF_NOT_MSG(ef_search > 0,
                            "IndexNSG::Search: ef_search must be positive");
  ef_search = std::max(ef_search, static_cast<size_t>(k));

  std::unique_ptr<DistanceComputer> distance =
      MakeTraversalDistance(*quantizer_, code_store_.View());
  const GraphSearcher searcher(graph_);
  VisitedTable visited(graph_.NodeCount(), false);
  const GraphId entry_points[] = {entry_point_};
  const IDSelector* selector = params == nullptr ? nullptr : params->sel;
  for (idx_t query = 0; query < n; ++query) {
    distance->SetQuery(x + static_cast<size_t>(query) * d);
    const std::vector<GraphSearchResult> results = searcher.Search(
        *distance, entry_points,
        GraphSearchOptions{ef_search, check_relative_distance, selector},
        &visited);
    const size_t result_count =
        std::min(results.size(), static_cast<size_t>(k));
    const size_t output_offset =
        static_cast<size_t>(query) * static_cast<size_t>(k);
    for (size_t result = 0; result < result_count; ++result) {
      distances[output_offset + result] =
          ExternalDistance(results[result].distance, metric_type);
      labels[output_offset + result] = results[result].id;
    }
  }
}

void IndexNSG::Reset() {
  code_store_.Reset();
  graph_.Resize(0);
  entry_point_ = kInvalidGraphId;
  n_total = 0;
  build_stats_.Reset();
}

void IndexNSG::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      key >= 0 && key < n_total,
      "IndexNSG::Reconstruct: key is outside [0, n_total)");
  HYPERVEC_THROW_IF_NOT_MSG(recons != nullptr,
                            "IndexNSG::Reconstruct: output must not be null");
  quantizer_->Decode(1, code_store_.Code(key), recons);
}

DistanceComputer* IndexNSG::GetDistanceComputer() const {
  return quantizer_->CreateDistanceComputer(code_store_.View()).release();
}

void IndexNSG::RestoreState(InMemoryCodeStore code_store,
                            MutableBoundedGraph graph, GraphId entry_point) {
  const idx_t restored_total = code_store.Size();
  HYPERVEC_THROW_IF_NOT_MSG(
      code_store.CodeSize() == quantizer_->CodeSize(),
      "IndexNSG::RestoreState: code size does not match the quantizer");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph.MaxDegree() ==
          GraphCapacity(options_, static_cast<size_t>(restored_total)),
      "IndexNSG::RestoreState: graph max degree does not match the options");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph.NodeCount() == static_cast<size_t>(restored_total),
      "IndexNSG::RestoreState: graph and code counts do not match");
  HYPERVEC_THROW_IF_NOT_MSG(
      (restored_total == 0 && entry_point == kInvalidGraphId) ||
          (restored_total > 0 && entry_point >= 0 &&
           entry_point < restored_total),
      "IndexNSG::RestoreState: entry point is inconsistent with the index");
  const GraphValidationReport report = ValidateGraph(graph, entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(restored_total),
      "IndexNSG::RestoreState: graph is invalid or unreachable");

  code_store_ = std::move(code_store);
  graph_ = std::move(graph);
  entry_point_ = entry_point;
  build_stats_.Reset();
  n_total = restored_total;
  ValidateAlignedState();
}

void IndexNSG::ValidateAlignedState() const {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total >= 0 && code_store_.Size() == n_total &&
          graph_.NodeCount() == static_cast<size_t>(n_total),
      "IndexNSG: code store, graph, and index counts are inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      (n_total == 0 && entry_point_ == kInvalidGraphId) ||
          (n_total > 0 && entry_point_ >= 0 && entry_point_ < n_total),
      "IndexNSG: entry point is inconsistent with the index size");
}

IndexNSGFlat::IndexNSGFlat(idx_t dimension, MetricType metric,
                           NSGIndexOptions options, float metric_arg)
    : IndexNSG(MakeFlat(dimension, metric, metric_arg), options, metric_arg) {}

}  // namespace hypervec
