/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <index/graph/visited_table.h>
#include <index/nsw/index_nsw.h>
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
                            "IndexNSW: quantizer must not be null");
  return quantizer->CodeSize();
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

}  // namespace

IndexNSW::IndexNSW(std::unique_ptr<Quantizer> quantizer,
                   NSWIndexOptions options, float metric_arg)
    : Index(),
      options_(options),
      quantizer_(std::move(quantizer)),
      code_store_(QuantizerCodeSize(quantizer_)),
      graph_(options.max_degree),
      builder_(NSWBuildOptions{options.ef_construction,
                               options.check_relative_distance,
                               options.fill_to_max_degree}) {
  HYPERVEC_THROW_IF_NOT_FMT(
      quantizer_->Dimension() > 0 &&
          quantizer_->Dimension() <= std::numeric_limits<int>::max(),
      "IndexNSW: dimension must be in [1, %d]",
      std::numeric_limits<int>::max());
  HYPERVEC_THROW_IF_NOT_MSG(options_.ef_construction >= options_.max_degree,
                            "IndexNSW: ef_construction must cover max_degree");
  HYPERVEC_THROW_IF_NOT_MSG(options_.ef_search > 0,
                            "IndexNSW: ef_search must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      quantizer_->Metric() != kMetricLp ||
          (std::isfinite(metric_arg) && metric_arg > 0.0F),
      "IndexNSW: kMetricLp requires a finite, positive metric_arg");

  d = static_cast<int>(quantizer_->Dimension());
  metric_type = quantizer_->Metric();
  this->metric_arg = metric_arg;
  is_trained = quantizer_->IsTrained();
}

IndexCapabilities IndexNSW::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.requires_training = quantizer_->NeedsTraining();
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexNSW::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0, "IndexNSW::Train: a populated index cannot be retrained");
  quantizer_->Train(n, x);
  is_trained = quantizer_->IsTrained();
  HYPERVEC_THROW_IF_NOT_MSG(
      is_trained, "IndexNSW::Train: quantizer did not become trained");
}

void IndexNSW::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexNSW::Add: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexNSW::Add: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained && quantizer_->IsTrained(),
                            "IndexNSW::Add: call Train before Add");
  ValidateAlignedState();
  if (n == 0) {
    return;
  }

  constexpr idx_t kMaxNodeCount =
      static_cast<idx_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total <= kMaxNodeCount && n <= kMaxNodeCount - n_total,
      "IndexNSW::Add: vector count exceeds GraphId capacity");
  const idx_t new_total = n_total + n;

  const size_t encoded_size =
      mul_no_overflow(static_cast<size_t>(n), quantizer_->CodeSize(),
                      "IndexNSW::Add encoded batch size");
  std::vector<uint8_t> encoded(encoded_size);
  quantizer_->Encode(n, x, encoded.data());

  InMemoryCodeStore staged_store = code_store_;
  staged_store.Append(n, encoded.data());
  MutableBoundedGraph staged_graph = graph_;
  GraphId staged_entry_point = entry_point_;
  NSWBuildStats staged_stats;
  std::unique_ptr<DistanceComputer> distance =
      MakeTraversalDistance(*quantizer_, staged_store.View());

  for (idx_t node = n_total; node < new_total; ++node) {
    distance->SetQuery(x + static_cast<size_t>(node - n_total) * d);
    const NSWInsertionResult result = builder_.AddNode(
        *distance, staged_graph, staged_entry_point, &staged_stats);
    HYPERVEC_THROW_IF_NOT_MSG(
        result.node_id == node,
        "IndexNSW::Add: graph and vector identifiers diverged");
    staged_entry_point = result.entry_point;
  }

  code_store_ = std::move(staged_store);
  graph_ = std::move(staged_graph);
  entry_point_ = staged_entry_point;
  n_total = new_total;
  build_stats_.Combine(staged_stats);
}

void IndexNSW::Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexNSW::Search: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexNSW::Search: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexNSW::Search: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexNSW::Search: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexNSW::Search: labels must not be null when n is positive");
  ValidateAlignedState();
  if (n == 0) {
    return;
  }

  const bool similarity = IsSimilarityMetric(metric_type);
  const float padding = similarity ? -(std::numeric_limits<float>::infinity)()
                                   : (std::numeric_limits<float>::infinity)();
  const size_t output_size =
      mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                      "IndexNSW::Search output size");
  std::fill_n(distances, output_size, padding);
  std::fill_n(labels, output_size, static_cast<idx_t>(-1));
  if (n_total == 0) {
    return;
  }

  size_t ef_search = options_.ef_search;
  bool check_relative_distance = options_.check_relative_distance;
  if (const auto* nsw_params =
          dynamic_cast<const SearchParametersNSW*>(params)) {
    ef_search = nsw_params->ef_search;
    check_relative_distance = nsw_params->check_relative_distance;
  }
  HYPERVEC_THROW_IF_NOT_MSG(ef_search > 0,
                            "IndexNSW::Search: ef_search must be positive");
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

void IndexNSW::Reset() {
  code_store_.Reset();
  graph_.Resize(0);
  entry_point_ = kInvalidGraphId;
  n_total = 0;
  build_stats_.Reset();
}

void IndexNSW::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      key >= 0 && key < n_total,
      "IndexNSW::Reconstruct: key is outside [0, n_total)");
  HYPERVEC_THROW_IF_NOT_MSG(recons != nullptr,
                            "IndexNSW::Reconstruct: output must not be null");
  quantizer_->Decode(1, code_store_.Code(key), recons);
}

DistanceComputer* IndexNSW::GetDistanceComputer() const {
  return quantizer_->CreateDistanceComputer(code_store_.View()).release();
}

void IndexNSW::ValidateAlignedState() const {
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total >= 0 && code_store_.Size() == n_total &&
          graph_.NodeCount() == static_cast<size_t>(n_total),
      "IndexNSW: code store, graph, and index counts are inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      (n_total == 0 && entry_point_ == kInvalidGraphId) ||
          (n_total > 0 && entry_point_ >= 0 && entry_point_ < n_total),
      "IndexNSW: entry point is inconsistent with the index size");
}

IndexNSWFlat::IndexNSWFlat(idx_t dimension, MetricType metric,
                           NSWIndexOptions options, float metric_arg)
    : IndexNSW(MakeFlat(dimension, metric, metric_arg), options, metric_arg) {}

}  // namespace hypervec
