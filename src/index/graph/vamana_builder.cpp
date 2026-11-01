/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/vamana_builder.h>
#include <index/graph/visited_table.h>
#include <omp.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

using NeighborList = std::vector<NeighborCandidate>;

bool CandidateOrder(const NeighborCandidate& lhs,
                    const NeighborCandidate& rhs) noexcept {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

float ValidateDistance(float distance) {
  HYPERVEC_THROW_IF_NOT_MSG(!std::isnan(distance),
                            "VamanaBuilder: distance computation returned NaN");
  return distance;
}

class NodeQueryDistanceComputer final : public DistanceComputer {
 public:
  NodeQueryDistanceComputer(DistanceComputer& distance, GraphId query)
      : distance_(distance), query_(query) {}

  void SetQuery(const float* /*query*/) override {}

  float operator()(idx_t index) override {
    return ValidateDistance(distance_.symmetric_dis(query_, index));
  }

  float symmetric_dis(idx_t lhs, idx_t rhs) override {
    return ValidateDistance(distance_.symmetric_dis(lhs, rhs));
  }

 private:
  DistanceComputer& distance_;
  GraphId query_;
};

bool Contains(std::span<const GraphId> neighbors, GraphId candidate) {
  return std::find(neighbors.begin(), neighbors.end(), candidate) !=
         neighbors.end();
}

std::vector<GraphId> NeighborIds(const NeighborList& neighbors) {
  std::vector<GraphId> result;
  result.reserve(neighbors.size());
  for (const NeighborCandidate& neighbor : neighbors) {
    result.push_back(neighbor.id);
  }
  return result;
}

NeighborList CollectCandidates(
    const MutableBoundedGraph& graph, GraphId source,
    std::span<const GraphSearchResult> search_results,
    NodeQueryDistanceComputer& query_distance, VisitedTable* pooled,
    size_t candidate_pool_size, VamanaBuildStats* stats) {
  NeighborList candidates;
  const GraphNeighborList previous = graph.Neighbors(source);
  candidates.reserve(add_no_overflow(search_results.size(), previous.size(),
                                     "VamanaBuilder candidate pool"));
  pooled->set(static_cast<size_t>(source));
  for (const GraphSearchResult& result : search_results) {
    if (pooled->set(static_cast<size_t>(result.id))) {
      candidates.push_back({result.id, ValidateDistance(result.distance)});
    }
  }
  for (GraphId neighbor : previous) {
    if (pooled->set(static_cast<size_t>(neighbor))) {
      candidates.push_back({neighbor, query_distance(neighbor)});
      ++stats->candidate_distance_computations;
    }
  }
  std::sort(candidates.begin(), candidates.end(), CandidateOrder);
  if (candidates.size() > candidate_pool_size) {
    candidates.resize(candidate_pool_size);
  }
  pooled->advance();
  return candidates;
}

void InsertReciprocal(MutableBoundedGraph* graph, GraphId source,
                      GraphId neighbor, DistanceComputer& distance,
                      const VamanaRobustPruner& pruner, size_t max_degree,
                      VamanaBuildStats* stats) {
  const GraphNeighborList current = graph->Neighbors(neighbor);
  if (Contains(current, source)) {
    return;
  }
  if (current.size() < max_degree) {
    graph->AddNeighbor(neighbor, source);
    ++stats->reciprocal_edges_added;
    return;
  }

  NeighborList candidates;
  candidates.reserve(current.size() + 1);
  for (GraphId candidate : current) {
    candidates.push_back({candidate, ValidateDistance(distance.symmetric_dis(
                                         neighbor, candidate))});
    ++stats->candidate_distance_computations;
  }
  candidates.push_back(
      {source, ValidateDistance(distance.symmetric_dis(neighbor, source))});
  ++stats->candidate_distance_computations;
  NodeQueryDistanceComputer query_distance(distance, neighbor);
  const NeighborList selected =
      pruner.Prune(candidates, max_degree, query_distance, &stats->pruning);
  const std::vector<GraphId> ids = NeighborIds(selected);
  graph->SetNeighbors(neighbor, ids);
  ++stats->reciprocal_edges_repruned;
  if (Contains(graph->Neighbors(neighbor), source)) {
    ++stats->reciprocal_edges_added;
  } else {
    ++stats->reciprocal_edges_rejected;
  }
}

struct Reachability {
  std::vector<bool> reachable;
  std::vector<GraphId> parent;
  size_t count = 0;
};

void MarkReachable(const GraphStorage& graph, GraphId entry_point,
                   GraphId parent, Reachability* result) {
  std::queue<GraphId> pending;
  result->reachable[static_cast<size_t>(entry_point)] = true;
  result->parent[static_cast<size_t>(entry_point)] = parent;
  pending.push(entry_point);
  while (!pending.empty()) {
    const GraphId source = pending.front();
    pending.pop();
    ++result->count;
    for (GraphId neighbor : graph.Neighbors(source)) {
      if (!result->reachable[static_cast<size_t>(neighbor)]) {
        result->reachable[static_cast<size_t>(neighbor)] = true;
        result->parent[static_cast<size_t>(neighbor)] = source;
        pending.push(neighbor);
      }
    }
  }
}

Reachability FindReachable(const GraphStorage& graph, GraphId entry_point) {
  Reachability result;
  result.reachable.assign(graph.NodeCount(), false);
  result.parent.assign(graph.NodeCount(), kInvalidGraphId);
  MarkReachable(graph, entry_point, kInvalidGraphId, &result);
  return result;
}

bool HasRepairCapacity(const MutableBoundedGraph& graph, GraphId source,
                       size_t max_degree, std::span<const GraphId> parent) {
  const GraphNeighborList neighbors = graph.Neighbors(source);
  if (neighbors.size() < max_degree) {
    return true;
  }
  return std::any_of(neighbors.begin(), neighbors.end(), [&](GraphId neighbor) {
    return parent[static_cast<size_t>(neighbor)] != source;
  });
}

GraphId SelectReplacement(const MutableBoundedGraph& graph, GraphId source,
                          std::span<const GraphId> parent,
                          DistanceComputer& distance, VamanaBuildStats* stats) {
  GraphId replacement = kInvalidGraphId;
  float farthest = -(std::numeric_limits<float>::infinity)();
  for (GraphId neighbor : graph.Neighbors(source)) {
    if (parent[static_cast<size_t>(neighbor)] == source) {
      continue;
    }
    const float candidate_distance =
        ValidateDistance(distance.symmetric_dis(source, neighbor));
    ++stats->connectivity_distance_computations;
    if (replacement == kInvalidGraphId || candidate_distance > farthest ||
        (candidate_distance == farthest && neighbor > replacement)) {
      replacement = neighbor;
      farthest = candidate_distance;
    }
  }
  return replacement;
}

void RepairConnectivity(MutableBoundedGraph* graph, GraphId entry_point,
                        size_t max_degree, size_t search_width,
                        DistanceComputer& distance, VamanaBuildStats* stats) {
  Reachability reachability = FindReachable(*graph, entry_point);
  size_t next_unreachable = 0;
  const GraphSearcher searcher(*graph);
  VisitedTable searched(graph->NodeCount(), false);
  while (reachability.count < graph->NodeCount()) {
    while (reachability.reachable[next_unreachable]) {
      ++next_unreachable;
    }
    const GraphId target = static_cast<GraphId>(next_unreachable);
    GraphId best_source = kInvalidGraphId;
    float best_distance = (std::numeric_limits<float>::infinity)();
    const auto consider_source = [&](GraphId source) {
      if (!reachability.reachable[static_cast<size_t>(source)] ||
          !HasRepairCapacity(*graph, source, max_degree, reachability.parent)) {
        return;
      }
      const float candidate_distance =
          ValidateDistance(distance.symmetric_dis(source, target));
      ++stats->connectivity_distance_computations;
      if (best_source == kInvalidGraphId ||
          candidate_distance < best_distance ||
          (candidate_distance == best_distance && source < best_source)) {
        best_source = source;
        best_distance = candidate_distance;
      }
    };
    for (GraphId source : graph->Neighbors(target)) {
      consider_source(source);
    }
    if (best_source == kInvalidGraphId) {
      NodeQueryDistanceComputer query_distance(distance, target);
      const std::array<GraphId, 1> entry_points = {entry_point};
      const auto nearby = searcher.Search(
          query_distance, entry_points,
          GraphSearchOptions{std::max(search_width, max_degree * 4), false,
                             nullptr},
          &searched, &stats->search);
      for (const GraphSearchResult& candidate : nearby) {
        consider_source(candidate.id);
      }
    }
    if (best_source == kInvalidGraphId) {
      for (size_t source = 0; source < graph->NodeCount(); ++source) {
        if (!reachability.reachable[source]) {
          continue;
        }
        consider_source(static_cast<GraphId>(source));
      }
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        best_source != kInvalidGraphId,
        "VamanaBuilder: no bounded edge is available for connectivity repair");

    if (graph->Neighbors(best_source).size() == max_degree) {
      const GraphId replacement = SelectReplacement(
          *graph, best_source, reachability.parent, distance, stats);
      HYPERVEC_THROW_IF_NOT_MSG(
          replacement != kInvalidGraphId,
          "VamanaBuilder: connectivity repair would remove a tree edge");
      HYPERVEC_THROW_IF_NOT_MSG(
          graph->RemoveNeighbor(best_source, replacement),
          "VamanaBuilder: connectivity replacement edge disappeared");
      ++stats->connectivity_edges_replaced;
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        graph->AddNeighbor(best_source, target),
        "VamanaBuilder: connectivity edge already exists");
    ++stats->connectivity_edges_added;
    // Removing only non-tree edges preserves the existing reachable set.
    // Expand from the newly connected target instead of re-walking the entire
    // graph after each repaired component.
    MarkReachable(*graph, target, best_source, &reachability);
  }
}

}  // namespace

void VamanaBuildStats::Reset() noexcept { *this = {}; }

void VamanaBuildStats::Combine(const VamanaBuildStats& other) noexcept {
  passes_completed += other.passes_completed;
  nodes_processed += other.nodes_processed;
  candidate_distance_computations += other.candidate_distance_computations;
  reciprocal_edges_added += other.reciprocal_edges_added;
  reciprocal_edges_repruned += other.reciprocal_edges_repruned;
  reciprocal_edges_rejected += other.reciprocal_edges_rejected;
  connectivity_distance_computations +=
      other.connectivity_distance_computations;
  connectivity_edges_added += other.connectivity_edges_added;
  connectivity_edges_replaced += other.connectivity_edges_replaced;
  search.Combine(other.search);
  pruning.Combine(other.pruning);
}

VamanaBuilder::VamanaBuilder(VamanaBuildOptions options) : options_(options) {
  HYPERVEC_THROW_IF_NOT_MSG(options_.max_degree > 0,
                            "VamanaBuilder: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.search_width >= options_.max_degree,
      "VamanaBuilder: search_width must cover max_degree");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.candidate_pool_size >= options_.search_width,
      "VamanaBuilder: candidate_pool_size must cover search_width");
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(options_.alpha) && options_.alpha >= 1.0F,
      "VamanaBuilder: alpha must be finite and at least one");
  HYPERVEC_THROW_IF_NOT_MSG(options_.build_passes > 0,
                            "VamanaBuilder: build_passes must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.build_threads > 0 &&
          options_.build_threads <=
              static_cast<size_t>((std::numeric_limits<int>::max)()),
      "VamanaBuilder: build_threads must be in [1, INT_MAX]");
}

MutableBoundedGraph VamanaBuilder::Build(DistanceComputer& distance,
                                         size_t node_count,
                                         GraphId navigation_point,
                                         VamanaBuildStats* stats) const {
  return Build(distance, node_count, navigation_point, stats, {});
}

MutableBoundedGraph VamanaBuilder::Build(
    DistanceComputer& distance, size_t node_count, GraphId navigation_point,
    VamanaBuildStats* stats, const DistanceFactory& distance_factory) const {
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(
      node_count <= kGraphCapacity,
      "VamanaBuilder: node count exceeds GraphId capacity");
  if (node_count == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        navigation_point == kInvalidGraphId,
        "VamanaBuilder: an empty graph requires an invalid navigation point");
    return MutableBoundedGraph(options_.max_degree);
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      navigation_point >= 0 &&
          static_cast<size_t>(navigation_point) < node_count,
      "VamanaBuilder: navigation point is outside the graph");

  MutableBoundedGraph graph(node_count, options_.max_degree);
  VamanaBuildStats local_stats;
  if (node_count == 1) {
    if (stats != nullptr) {
      stats->Combine(local_stats);
    }
    return graph;
  }

  const size_t degree = std::min(options_.max_degree, node_count - 1);
  const GraphSearcher searcher(graph);
  VisitedTable searched(node_count, false);
  VisitedTable pooled(node_count, false);
  std::vector<GraphId> order(node_count);
  for (size_t node = 0; node < node_count; ++node) {
    order[node] = static_cast<GraphId>(node);
  }
  std::mt19937_64 random(options_.random_seed);
  const std::array<GraphId, 1> entry_points = {navigation_point};

  const size_t workers = std::min(options_.build_threads, node_count);
  std::vector<std::unique_ptr<DistanceComputer>> distances;
  std::vector<std::unique_ptr<VisitedTable>> worker_searched;
  std::vector<std::unique_ptr<VisitedTable>> worker_pooled;
  std::vector<VamanaBuildStats> worker_stats(workers);
  if (workers > 1) {
    HYPERVEC_THROW_IF_NOT_MSG(
        static_cast<bool>(distance_factory),
        "VamanaBuilder: parallel build requires a distance factory");
    for (size_t worker = 0; worker < workers; ++worker) {
      distances.push_back(distance_factory());
      HYPERVEC_THROW_IF_NOT_MSG(
          distances.back() != nullptr,
          "VamanaBuilder: distance factory returned null");
      worker_searched.push_back(
          std::make_unique<VisitedTable>(node_count, false));
      worker_pooled.push_back(
          std::make_unique<VisitedTable>(node_count, false));
    }
  }

  for (size_t pass = 0; pass < options_.build_passes; ++pass) {
    std::shuffle(order.begin(), order.end(), random);
    const bool bootstrap_pass = options_.build_passes > 1 && pass == 0;
    const VamanaRobustPruner pruner(bootstrap_pass ? 1.0F : options_.alpha);
    // Start with a live, connected bootstrap; thereafter each batch reads a
    // fixed graph snapshot while workers propose outgoing edges. Only the
    // ordered commit stage mutates adjacency and reciprocal edges.
    const size_t bootstrap =
        workers == 1 || pass > 0 ? 0 : std::min(node_count, size_t{1024});
    const size_t batch_size = workers == 1 ? 1 : 1024;
    for (size_t begin = 0; begin < node_count; begin += batch_size) {
      const size_t end = std::min(node_count, begin + batch_size);
      std::vector<NeighborList> selected(end - begin);
      const size_t serial_end =
          begin < bootstrap ? std::min(end, bootstrap) : begin;
      const auto propose = [&](size_t position,
                               DistanceComputer& worker_distance,
                               VisitedTable* worker_visited,
                               VisitedTable* worker_pool,
                               VamanaBuildStats* partial) {
        const GraphId node = order[position];
        NodeQueryDistanceComputer query_distance(worker_distance, node);
        const std::vector<GraphSearchResult> search_results = searcher.Search(
            query_distance, entry_points,
            GraphSearchOptions{options_.search_width, false, nullptr},
            worker_visited, &partial->search);
        const NeighborList candidates = CollectCandidates(
            graph, node, search_results, query_distance, worker_pool,
            options_.candidate_pool_size, partial);
        selected[position - begin] =
            pruner.Prune(candidates, degree, query_distance, &partial->pruning);
        ++partial->nodes_processed;
      };
      const auto commit = [&](size_t position) {
        const GraphId node = order[position];
        const NeighborList& neighbors = selected[position - begin];
        graph.SetNeighbors(node, NeighborIds(neighbors));
        for (const NeighborCandidate& neighbor : neighbors) {
          InsertReciprocal(&graph, node, neighbor.id, distance, pruner, degree,
                           &local_stats);
        }
      };
      for (size_t position = begin; position < serial_end; ++position) {
        propose(position, distance, &searched, &pooled, &local_stats);
        commit(position);
      }
      if (serial_end == end) {
        continue;
      }
      if (workers == 1) {
        propose(begin, distance, &searched, &pooled, &local_stats);
      } else {
        std::exception_ptr error;
        std::mutex error_mutex;
        std::atomic<bool> failed{false};
#pragma omp parallel for num_threads(static_cast<int>(workers)) \
    schedule(dynamic, 8)
        for (std::ptrdiff_t position = static_cast<std::ptrdiff_t>(serial_end);
             position < static_cast<std::ptrdiff_t>(end); ++position) {
          if (failed.load()) {
            continue;
          }
          const size_t worker = static_cast<size_t>(omp_get_thread_num());
          try {
            propose(static_cast<size_t>(position), *distances[worker],
                    worker_searched[worker].get(), worker_pooled[worker].get(),
                    &worker_stats[worker]);
          } catch (...) {
            std::lock_guard<std::mutex> guard(error_mutex);
            if (error == nullptr) {
              error = std::current_exception();
            }
            failed.store(true);
          }
        }
        if (error != nullptr) {
          std::rethrow_exception(error);
        }
      }
      for (size_t position = serial_end; position < end; ++position) {
        commit(position);
      }
    }
    ++local_stats.passes_completed;
  }
  for (const VamanaBuildStats& partial : worker_stats) {
    local_stats.Combine(partial);
  }
  RepairConnectivity(&graph, navigation_point, degree, options_.search_width,
                     distance, &local_stats);

  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return graph;
}

}  // namespace hypervec
