/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/diskann/diskann_searcher.h>
#include <index/graph/visited_table.h>
#include <utils/distances/distance_computer.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

class NaNCheckingDistanceComputer final : public DistanceComputer {
 public:
  explicit NaNCheckingDistanceComputer(std::unique_ptr<DistanceComputer> inner)
      : inner_(std::move(inner)) {
    HYPERVEC_THROW_IF_NOT_MSG(
        inner_ != nullptr,
        "DiskAnnSearcher: quantizer returned a null DistanceComputer");
  }

  void SetQuery(const float* query) override { inner_->SetQuery(query); }

  float operator()(idx_t index) override { return Check((*inner_)(index)); }

  float symmetric_dis(idx_t lhs, idx_t rhs) override {
    return Check(inner_->symmetric_dis(lhs, rhs));
  }

 private:
  static float Check(float distance) {
    HYPERVEC_THROW_IF_NOT_MSG(
        !std::isnan(distance),
        "DiskAnnSearcher: approximate distance returned NaN");
    return distance;
  }

  std::unique_ptr<DistanceComputer> inner_;
};

bool ExactResultOrder(const GraphSearchResult& lhs,
                      const GraphSearchResult& rhs) noexcept {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

}  // namespace

void DiskAnnSearchStats::Reset() noexcept { *this = {}; }

void DiskAnnSearchStats::Combine(const DiskAnnSearchStats& other) noexcept {
  graph.Combine(other.graph);
  exact_distance_computations += other.exact_distance_computations;
}

DiskAnnSearcher::DiskAnnSearcher(const GraphStorage& graph,
                                 const PagedVectorStorage& raw_vectors,
                                 const Quantizer& quantizer,
                                 EncodedVectorView encoded_vectors,
                                 GraphId entry_point)
    : graph_(graph),
      raw_vectors_(raw_vectors),
      quantizer_(quantizer),
      encoded_vectors_(encoded_vectors),
      entry_point_(entry_point) {
  HYPERVEC_THROW_IF_NOT_MSG(graph_.NodeCount() > 0,
                            "DiskAnnSearcher: graph must not be empty");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph_.NodeCount() == raw_vectors_.NodeCount(),
      "DiskAnnSearcher: graph and raw vector counts do not match");
  HYPERVEC_THROW_IF_NOT_MSG(
      graph_.NodeCount() == static_cast<size_t>(encoded_vectors_.Size()),
      "DiskAnnSearcher: graph and encoded vector counts do not match");
  HYPERVEC_THROW_IF_NOT_MSG(
      quantizer_.Dimension() > 0 &&
          static_cast<size_t>(quantizer_.Dimension()) ==
              raw_vectors_.Dimension(),
      "DiskAnnSearcher: raw vector dimension does not match the quantizer");
  HYPERVEC_THROW_IF_NOT_MSG(
      quantizer_.Metric() == kMetricL2,
      "DiskAnnSearcher: only squared L2 distance is supported");
  HYPERVEC_THROW_IF_NOT_MSG(quantizer_.IsTrained(),
                            "DiskAnnSearcher: quantizer must be trained");
  HYPERVEC_THROW_IF_NOT_MSG(
      quantizer_.CodeSize() == encoded_vectors_.CodeSize(),
      "DiskAnnSearcher: encoded vector size does not match the quantizer");
  HYPERVEC_THROW_IF_NOT_MSG(
      entry_point_ >= 0 &&
          static_cast<size_t>(entry_point_) < graph_.NodeCount(),
      "DiskAnnSearcher: entry point is outside the graph");
}

std::vector<GraphSearchResult> DiskAnnSearcher::Search(
    const float* query, size_t result_count,
    const DiskAnnSearchOptions& options, DiskAnnSearchStats* stats) const {
  HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                            "DiskAnnSearcher::Search: query must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      result_count > 0,
      "DiskAnnSearcher::Search: result_count must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options.search_width > 0,
      "DiskAnnSearcher::Search: search_width must be positive");

  NaNCheckingDistanceComputer approximate(
      quantizer_.CreateDistanceComputer(encoded_vectors_));
  approximate.SetQuery(query);
  VisitedTable visited(graph_.NodeCount());
  const GraphSearcher graph_searcher(graph_);
  const std::array<GraphId, 1> entry_points = {entry_point_};
  DiskAnnSearchStats local_stats;
  std::vector<GraphSearchResult> candidates = graph_searcher.Search(
      approximate, entry_points,
      GraphSearchOptions{std::max(options.search_width, result_count),
                         options.check_relative_distance, options.selector},
      &visited, &local_stats.graph);

  for (const GraphSearchResult& candidate : candidates) {
    raw_vectors_.Prefetch(candidate.id);
  }
  for (GraphSearchResult& candidate : candidates) {
    const DiskAnnVectorView vector = raw_vectors_.Vector(candidate.id);
    candidate.distance = fvec_L2sqr(query, vector.data(), vector.size());
    HYPERVEC_THROW_IF_NOT_MSG(!std::isnan(candidate.distance),
                              "DiskAnnSearcher: exact distance returned NaN");
    ++local_stats.exact_distance_computations;
  }
  std::sort(candidates.begin(), candidates.end(), ExactResultOrder);
  if (candidates.size() > result_count) {
    candidates.resize(result_count);
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return candidates;
}

}  // namespace hypervec
