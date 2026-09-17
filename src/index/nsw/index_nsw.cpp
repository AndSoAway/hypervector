/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <index/graph/graph_validation.h>
#include <index/graph/nsw_builder.h>
#include <index/graph/visited_table.h>
#include <index/nsw/index_nsw.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace hypervec {
namespace {

size_t CheckedMaxDegree(int graph_degree) {
  HYPERVEC_THROW_IF_NOT_MSG(
      graph_degree >= 2 && graph_degree <= INT32_MAX / 2,
      "IndexNSWFlat: graph_degree must be in [2, INT32_MAX / 2]");
  return static_cast<size_t>(graph_degree) * 2;
}

std::unique_ptr<DistanceComputer> TraversalDistance(const IndexFlat& storage) {
  DistanceComputer* distance = storage.GetDistanceComputer();
  if (IsSimilarityMetric(storage.metric_type)) {
    distance = new NegativeDistanceComputer(distance);
  }
  return std::unique_ptr<DistanceComputer>(distance);
}

int ResolveEfSearch(const IndexNSWFlat& index, const SearchParameters* params) {
  int ef_search = index.ef_search;
  if (const auto* nsw_params =
          dynamic_cast<const SearchParametersNSW*>(params)) {
    ef_search = nsw_params->ef_search;
  }
  HYPERVEC_THROW_IF_NOT_MSG(ef_search > 0,
                            "IndexNSWFlat::Search: ef_search must be positive");
  return ef_search;
}

void CheckBasicConsistency(const IndexNSWFlat& index) {
  HYPERVEC_THROW_IF_NOT_MSG(
      index.d > 0 && index.is_trained && index.graph_degree >= 2 &&
          index.graph_degree <= INT32_MAX / 2 && index.ef_construction > 0 &&
          index.ef_search > 0,
      "IndexNSWFlat: invalid dimensions or graph parameters");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.n_total >= 0 && index.storage.d == index.d &&
          index.storage.n_total == index.n_total &&
          index.storage.is_trained == index.is_trained &&
          index.storage.metric_type == index.metric_type &&
          index.storage.metric_arg == index.metric_arg,
      "IndexNSWFlat: vector storage metadata is inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.storage.code_size == static_cast<size_t>(index.d) * sizeof(float) &&
          static_cast<size_t>(index.n_total) <=
              (std::numeric_limits<size_t>::max)() / index.storage.code_size &&
          index.storage.codes.size() ==
              static_cast<size_t>(index.n_total) * index.storage.code_size,
      "IndexNSWFlat: vector storage size is inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.graph.MaxDegree() == CheckedMaxDegree(index.graph_degree) &&
          index.graph.NodeCount() == static_cast<size_t>(index.n_total),
      "IndexNSWFlat: graph shape is inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      (index.n_total == 0 && index.entry_point == kInvalidGraphId) ||
          (index.n_total > 0 && index.entry_point >= 0 &&
           index.entry_point < index.n_total),
      "IndexNSWFlat: entry point is inconsistent with vector count");
}

void ValidateSearchArguments(const IndexNSWFlat& index, idx_t n, const float* x,
                             idx_t k, float* distances, idx_t* labels) {
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexNSWFlat::Search: n must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(k > 0, "IndexNSWFlat::Search: k must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      k <= static_cast<idx_t>(INT32_MAX),
      "IndexNSWFlat::Search: k exceeds supported graph-search capacity");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexNSWFlat::Search: queries must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || (distances != nullptr && labels != nullptr),
      "IndexNSWFlat::Search: outputs must not be null when n is positive");
  CheckBasicConsistency(index);
}

class SimilarityResultHandler final : public ResultHandler {
 public:
  explicit SimilarityResultHandler(ResultHandler& target) : target_(target) {
    threshold = -target_.threshold;
  }

  bool AddResult(float distance, idx_t id) override {
    const bool updated = target_.AddResult(-distance, id);
    threshold = -target_.threshold;
    return updated;
  }

 private:
  ResultHandler& target_;
};

void SearchOne(const IndexNSWFlat& index, DistanceComputer& distance,
               ResultHandler& results, GraphCandidateQueue& candidates,
               VisitedTable& visited, int ef_search,
               const IDSelector* selector) {
  if (index.n_total == 0) {
    return;
  }
  candidates.clear();
  candidates.push(index.entry_point, distance(index.entry_point));
  GraphSearchOptions options;
  options.ef_search = ef_search;
  options.selector = selector;
  GraphSearcher(index.graph)
      .Search(distance, results, candidates, visited, options, nullptr);
  visited.advance();
}

}  // namespace

IndexNSWFlat::IndexNSWFlat() : Index(), storage(), graph(1) {}

IndexNSWFlat::IndexNSWFlat(int d, int degree, MetricType metric)
    : Index(d, metric),
      graph_degree(degree),
      storage(d, metric),
      graph(CheckedMaxDegree(degree)) {
  HYPERVEC_THROW_IF_NOT_MSG(d > 0, "IndexNSWFlat: dimension must be positive");
}

IndexCapabilities IndexNSWFlat::Capabilities() const noexcept {
  return IndexCapability::BatchBuild | IndexCapability::IncrementalAdd |
         IndexCapability::KnnSearch | IndexCapability::Filter |
         IndexCapability::ConcurrentSearch | IndexCapability::Serialize;
}

void IndexNSWFlat::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(n >= 0,
                            "IndexNSWFlat::Add: n must be non-negative");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexNSWFlat::Add: vectors must not be null when n is positive");
  if (n == 0) {
    return;
  }
  if (n_total == 0) {
    storage.metric_arg = metric_arg;
  }
  CheckBasicConsistency(*this);
  HYPERVEC_THROW_IF_NOT_MSG(n_total <= (std::numeric_limits<idx_t>::max)() - n,
                            "IndexNSWFlat::Add: vector count overflow");
  const idx_t new_total = n_total + n;
  HYPERVEC_THROW_IF_NOT_MSG(
      new_total <= static_cast<idx_t>((std::numeric_limits<GraphId>::max)()),
      "IndexNSWFlat::Add: vector count exceeds GraphId capacity");

  const idx_t old_total = n_total;
  const size_t old_code_size = storage.codes.size();
  MutableBoundedGraph new_graph = graph;
  new_graph.Resize(static_cast<size_t>(new_total));

  try {
    storage.Add(n, x);
    std::unique_ptr<DistanceComputer> distance = TraversalDistance(storage);
    NSWIncrementalBuilder builder(
        new_graph, {static_cast<size_t>(graph_degree), ef_construction});
    GraphId new_entry_point = entry_point;
    for (idx_t id = old_total; id < new_total; ++id) {
      if (id == 0) {
        new_entry_point = 0;
        continue;
      }
      distance->SetQuery(x + (id - old_total) * d);
      builder.AddNode(static_cast<GraphId>(id), new_entry_point, *distance);
    }

    graph = std::move(new_graph);
    entry_point = new_entry_point;
    n_total = new_total;
  } catch (...) {
    storage.codes.resize(old_code_size);
    storage.n_total = old_total;
    throw;
  }
}

void IndexNSWFlat::Search(idx_t n, const float* x, idx_t k, float* distances,
                          idx_t* labels, const SearchParameters* params) const {
  ValidateSearchArguments(*this, n, x, k, distances, labels);
  if (n == 0) {
    return;
  }
  const int ef_search =
      std::max(ResolveEfSearch(*this, params), static_cast<int>(k));
  const IDSelector* selector = params == nullptr ? nullptr : params->sel;
  std::unique_ptr<DistanceComputer> distance = TraversalDistance(storage);

  using Handler = HeapBlockResultHandler<CMax<float, idx_t>>;
  Handler block(static_cast<size_t>(n), distances, labels,
                static_cast<size_t>(k));
  Handler::SingleResultHandler results(block);
  GraphCandidateQueue candidates(ef_search);
  VisitedTable visited(static_cast<size_t>(n_total));
  for (idx_t query = 0; query < n; ++query) {
    results.begin(static_cast<size_t>(query));
    if (n_total > 0) {
      distance->SetQuery(x + query * d);
      SearchOne(*this, *distance, results, candidates, visited, ef_search,
                selector);
    }
    results.end();
  }

  if (IsSimilarityMetric(metric_type)) {
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(k);
    for (size_t i = 0; i < count; ++i) {
      distances[i] = -distances[i];
    }
  }
}

void IndexNSWFlat::Search1(const float* x, ResultHandler& handler,
                           SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr,
                            "IndexNSWFlat::Search1: query must not be null");
  ValidateSearchArguments(*this, 0, nullptr, 1, nullptr, nullptr);
  if (n_total == 0) {
    return;
  }
  const int ef_search = ResolveEfSearch(*this, params);
  const IDSelector* selector = params == nullptr ? nullptr : params->sel;
  std::unique_ptr<DistanceComputer> distance = TraversalDistance(storage);
  distance->SetQuery(x);
  GraphCandidateQueue candidates(ef_search);
  VisitedTable visited(static_cast<size_t>(n_total));
  if (IsSimilarityMetric(metric_type)) {
    SimilarityResultHandler traversal_results(handler);
    SearchOne(*this, *distance, traversal_results, candidates, visited,
              ef_search, selector);
  } else {
    SearchOne(*this, *distance, handler, candidates, visited, ef_search,
              selector);
  }
}

void IndexNSWFlat::Reset() {
  storage.Reset();
  graph = MutableBoundedGraph(CheckedMaxDegree(graph_degree));
  n_total = 0;
  entry_point = kInvalidGraphId;
}

void IndexNSWFlat::Reconstruct(idx_t key, float* recons) const {
  storage.Reconstruct(key, recons);
}

DistanceComputer* IndexNSWFlat::GetDistanceComputer() const {
  return storage.GetDistanceComputer();
}

void IndexNSWFlat::CheckConsistency() const {
  CheckBasicConsistency(*this);
  const GraphValidationReport report = ValidateGraph(graph, entry_point);
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(n_total),
      "IndexNSWFlat: graph is invalid or not fully reachable");
}

}  // namespace hypervec
