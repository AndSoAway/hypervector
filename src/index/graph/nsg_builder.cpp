/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_validation.h>
#include <index/graph/nsg_builder.h>
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

struct CandidateGreater {
  bool operator()(const NeighborCandidate& lhs,
                  const NeighborCandidate& rhs) const noexcept {
    return CandidateOrder(rhs, lhs);
  }
};

float ValidateDistance(float distance) {
  HYPERVEC_THROW_IF_NOT_MSG(!std::isnan(distance),
                            "NSGBuilder: distance computation returned NaN");
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

bool Contains(const NeighborList& neighbors, GraphId candidate) {
  return std::any_of(
      neighbors.begin(), neighbors.end(),
      [&](const auto& neighbor) { return neighbor.id == candidate; });
}

void AddCandidate(const NeighborCandidate& candidate, VisitedTable* pooled,
                  NeighborList* candidates) {
  if (pooled->set(static_cast<size_t>(candidate.id))) {
    candidates->push_back(candidate);
  }
}

void BuildDirectedNeighbors(
    const GraphStorage& candidate_graph, const GraphSearcher& searcher,
    GraphId navigation_point, size_t node, size_t pruned_degree,
    const NSGBuildOptions& options, const HnswHeuristicPruner& pruner,
    DistanceComputer& distance, VisitedTable* searched, VisitedTable* pooled,
    NeighborList* neighbors, NSGBuildStats* stats) {
  NodeQueryDistanceComputer query_distance(distance,
                                           static_cast<GraphId>(node));
  const std::array<GraphId, 1> entry_points = {navigation_point};
  const std::vector<GraphSearchResult> search_results = searcher.Search(
      query_distance, entry_points,
      GraphSearchOptions{options.search_width, options.check_relative_distance,
                         nullptr},
      searched, &stats->search);

  const GraphNeighborList original_neighbors =
      candidate_graph.Neighbors(static_cast<GraphId>(node));
  NeighborList candidates;
  candidates.reserve(add_no_overflow(search_results.size(),
                                     original_neighbors.size(),
                                     "NSGBuilder candidate pool"));
  pooled->set(node);
  for (const GraphSearchResult& result : search_results) {
    AddCandidate({result.id, ValidateDistance(result.distance)}, pooled,
                 &candidates);
  }
  for (GraphId neighbor : original_neighbors) {
    if (pooled->set(static_cast<size_t>(neighbor))) {
      candidates.push_back({neighbor, query_distance(neighbor)});
      ++stats->candidate_distance_computations;
    }
  }
  std::sort(candidates.begin(), candidates.end(), CandidateOrder);
  if (candidates.size() > options.candidate_pool_size) {
    candidates.resize(options.candidate_pool_size);
  }
  *neighbors =
      pruner.Prune(candidates, pruned_degree, query_distance, &stats->pruning);
  ++stats->pruned_nodes;
  pooled->advance();
}

void MarkReachable(const std::vector<NeighborList>& neighborhoods,
                   GraphId entry_point, std::vector<bool>* reachable,
                   size_t* reachable_count,
                   std::vector<GraphId>* newly_reachable) {
  if ((*reachable)[static_cast<size_t>(entry_point)]) {
    return;
  }
  std::queue<GraphId> pending;
  (*reachable)[static_cast<size_t>(entry_point)] = true;
  pending.push(entry_point);
  while (!pending.empty()) {
    const GraphId node = pending.front();
    pending.pop();
    ++*reachable_count;
    newly_reachable->push_back(node);
    for (const NeighborCandidate& neighbor :
         neighborhoods[static_cast<size_t>(node)]) {
      if (!(*reachable)[static_cast<size_t>(neighbor.id)]) {
        (*reachable)[static_cast<size_t>(neighbor.id)] = true;
        pending.push(neighbor.id);
      }
    }
  }
}

std::vector<GraphId> NeighborIds(const NeighborList& neighbors) {
  std::vector<GraphId> result;
  result.reserve(neighbors.size());
  for (const NeighborCandidate& neighbor : neighbors) {
    result.push_back(neighbor.id);
  }
  return result;
}

}  // namespace

void NSGBuildStats::Reset() noexcept { *this = {}; }

void NSGBuildStats::Combine(const NSGBuildStats& other) noexcept {
  pruned_nodes += other.pruned_nodes;
  candidate_distance_computations += other.candidate_distance_computations;
  reciprocal_edges_added += other.reciprocal_edges_added;
  reciprocal_edges_rejected += other.reciprocal_edges_rejected;
  connectivity_distance_computations +=
      other.connectivity_distance_computations;
  connectivity_edges_added += other.connectivity_edges_added;
  search.Combine(other.search);
  pruning.Combine(other.pruning);
}

NSGBuilder::NSGBuilder(NSGBuildOptions options)
    : options_(options), pruner_(false) {
  HYPERVEC_THROW_IF_NOT_MSG(options_.max_degree > 0,
                            "NSGBuilder: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(options_.search_width > 0,
                            "NSGBuilder: search_width must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.candidate_pool_size >= options_.max_degree,
      "NSGBuilder: candidate_pool_size must cover max_degree");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.build_threads > 0 &&
          options_.build_threads <=
              static_cast<size_t>(std::numeric_limits<int>::max()),
      "NSGBuilder: build_threads must be in [1, INT_MAX]");
}

MutableBoundedGraph NSGBuilder::Build(const GraphStorage& candidate_graph,
                                      DistanceComputer& distance,
                                      GraphId navigation_point,
                                      NSGBuildStats* stats) const {
  return Build(candidate_graph, distance, navigation_point, stats, {});
}

MutableBoundedGraph NSGBuilder::Build(
    const GraphStorage& candidate_graph, DistanceComputer& distance,
    GraphId navigation_point, NSGBuildStats* stats,
    const DistanceFactory& distance_factory) const {
  const size_t node_count = candidate_graph.NodeCount();
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(node_count <= kGraphCapacity,
                            "NSGBuilder: node count exceeds GraphId capacity");

  if (node_count == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        navigation_point == kInvalidGraphId,
        "NSGBuilder: an empty graph requires an invalid navigation point");
    return MutableBoundedGraph(1);
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      navigation_point >= 0 &&
          static_cast<size_t>(navigation_point) < node_count,
      "NSGBuilder: navigation point is outside the graph");
  const GraphValidationReport candidate_report = ValidateGraph(candidate_graph);
  HYPERVEC_THROW_IF_NOT_MSG(
      candidate_report.IsStructurallyValid(),
      "NSGBuilder: candidate graph is structurally invalid");

  const size_t pruned_degree = std::min(options_.max_degree, node_count - 1);
  const size_t graph_capacity =
      node_count == 1 ? 1 : std::min(node_count - 1, pruned_degree + 1);
  MutableBoundedGraph graph(node_count, graph_capacity);
  NSGBuildStats local_stats;
  if (node_count == 1) {
    if (stats != nullptr) {
      stats->Combine(local_stats);
    }
    return graph;
  }

  std::vector<NeighborList> neighborhoods(node_count);
  const GraphSearcher searcher(candidate_graph);
  const size_t workers = std::min(options_.build_threads, node_count);
  if (workers == 1) {
    VisitedTable searched(node_count, false);
    VisitedTable pooled(node_count, false);
    for (size_t node = 0; node < node_count; ++node) {
      BuildDirectedNeighbors(candidate_graph, searcher, navigation_point, node,
                             pruned_degree, options_, pruner_, distance,
                             &searched, &pooled, &neighborhoods[node],
                             &local_stats);
    }
  } else {
    HYPERVEC_THROW_IF_NOT_MSG(
        static_cast<bool>(distance_factory),
        "NSGBuilder: parallel build requires a distance factory");
    std::vector<std::unique_ptr<DistanceComputer>> distances;
    std::vector<std::unique_ptr<VisitedTable>> searched;
    std::vector<std::unique_ptr<VisitedTable>> pooled;
    distances.reserve(workers);
    searched.reserve(workers);
    pooled.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
      distances.push_back(distance_factory());
      HYPERVEC_THROW_IF_NOT_MSG(distances.back() != nullptr,
                                "NSGBuilder: distance factory returned null");
      searched.push_back(std::make_unique<VisitedTable>(node_count, false));
      pooled.push_back(std::make_unique<VisitedTable>(node_count, false));
    }
    std::vector<NSGBuildStats> worker_stats(workers);
    std::exception_ptr error;
    std::mutex error_mutex;
    std::atomic<bool> failed{false};
#pragma omp parallel for num_threads(static_cast<int>(workers)) \
    schedule(dynamic, 8)
    for (std::ptrdiff_t node = 0;
         node < static_cast<std::ptrdiff_t>(node_count); ++node) {
      if (failed.load()) {
        continue;
      }
      const size_t worker = static_cast<size_t>(omp_get_thread_num());
      try {
        BuildDirectedNeighbors(
            candidate_graph, searcher, navigation_point,
            static_cast<size_t>(node), pruned_degree, options_, pruner_,
            *distances[worker], searched[worker].get(), pooled[worker].get(),
            &neighborhoods[static_cast<size_t>(node)], &worker_stats[worker]);
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
    for (const NSGBuildStats& per_worker : worker_stats) {
      local_stats.Combine(per_worker);
    }
  }

  const std::vector<NeighborList> directed = neighborhoods;
  const auto add_reciprocals = [&](size_t node,
                                   DistanceComputer& worker_distance,
                                   NSGBuildStats* worker_stats,
                                   std::vector<std::mutex>* locks) {
    for (const NeighborCandidate& neighbor : directed[node]) {
      const auto insert = [&] {
        NeighborList& reciprocal =
            neighborhoods[static_cast<size_t>(neighbor.id)];
        const GraphId source = static_cast<GraphId>(node);
        if (Contains(reciprocal, source)) {
          return;
        }
        const NeighborCandidate proposal{source, neighbor.distance};
        if (reciprocal.size() < pruned_degree) {
          reciprocal.push_back(proposal);
          ++worker_stats->reciprocal_edges_added;
          return;
        }
        NeighborList candidates = reciprocal;
        candidates.push_back(proposal);
        NodeQueryDistanceComputer query_distance(worker_distance, neighbor.id);
        reciprocal = pruner_.Prune(candidates, pruned_degree, query_distance,
                                   &worker_stats->pruning);
        if (Contains(reciprocal, source)) {
          ++worker_stats->reciprocal_edges_added;
        } else {
          ++worker_stats->reciprocal_edges_rejected;
        }
      };
      if (locks != nullptr) {
        std::lock_guard<std::mutex> lock(
            (*locks)[static_cast<size_t>(neighbor.id)]);
        insert();
      } else {
        insert();
      }
    }
  };
  if (workers == 1) {
    for (size_t node = 0; node < node_count; ++node) {
      add_reciprocals(node, distance, &local_stats, nullptr);
    }
  } else {
    std::vector<std::mutex> node_locks(node_count);
    std::vector<std::unique_ptr<DistanceComputer>> distances;
    distances.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
      distances.push_back(distance_factory());
      HYPERVEC_THROW_IF_NOT_MSG(distances.back() != nullptr,
                                "NSGBuilder: distance factory returned null");
    }
    std::vector<NSGBuildStats> worker_stats(workers);
    std::exception_ptr error;
    std::mutex error_mutex;
    std::atomic<bool> failed{false};
#pragma omp parallel for num_threads(static_cast<int>(workers)) \
    schedule(dynamic, 8)
    for (std::ptrdiff_t node = 0;
         node < static_cast<std::ptrdiff_t>(node_count); ++node) {
      if (failed.load()) {
        continue;
      }
      const size_t worker = static_cast<size_t>(omp_get_thread_num());
      try {
        add_reciprocals(static_cast<size_t>(node), *distances[worker],
                        &worker_stats[worker], &node_locks);
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
    for (const auto& worker : worker_stats) {
      local_stats.Combine(worker);
    }
  }

  size_t reachable_count = 0;
  std::vector<bool> reachable(node_count, false);
  std::vector<GraphId> newly_reachable;
  MarkReachable(neighborhoods, navigation_point, &reachable, &reachable_count,
                &newly_reachable);
  constexpr size_t kNoSourcePosition = (std::numeric_limits<size_t>::max)();
  std::vector<GraphId> repair_sources;
  repair_sources.reserve(node_count);
  std::vector<size_t> source_positions(node_count, kNoSourcePosition);
  const auto add_sources = [&](const std::vector<GraphId>& nodes) {
    for (GraphId node : nodes) {
      const size_t position = static_cast<size_t>(node);
      if (neighborhoods[position].size() < graph_capacity) {
        source_positions[position] = repair_sources.size();
        repair_sources.push_back(node);
      }
    }
  };
  const auto remove_source = [&](GraphId node) {
    const size_t node_position = static_cast<size_t>(node);
    const size_t source_position = source_positions[node_position];
    HYPERVEC_THROW_IF_NOT_MSG(source_position != kNoSourcePosition,
                              "NSGBuilder: repair source is not active");
    const GraphId replacement = repair_sources.back();
    repair_sources[source_position] = replacement;
    source_positions[static_cast<size_t>(replacement)] = source_position;
    repair_sources.pop_back();
    source_positions[node_position] = kNoSourcePosition;
  };
  add_sources(newly_reachable);
  VisitedTable repair_visited(node_count, false);
  size_t next_unreachable = 0;
  while (reachable_count < node_count) {
    while (reachable[next_unreachable]) {
      ++next_unreachable;
    }
    const GraphId target = static_cast<GraphId>(next_unreachable);
    GraphId best_source = kInvalidGraphId;
    float best_distance = (std::numeric_limits<float>::infinity)();
    const auto consider_source = [&](GraphId source, float source_distance) {
      if (source < 0 ||
          source_positions[static_cast<size_t>(source)] == kNoSourcePosition) {
        return;
      }
      if (best_source == kInvalidGraphId || source_distance < best_distance ||
          (source_distance == best_distance && source < best_source)) {
        best_source = source;
        best_distance = source_distance;
      }
    };
    const auto evaluate_source = [&](GraphId source) {
      const float source_distance =
          ValidateDistance(distance.symmetric_dis(source, target));
      ++local_stats.connectivity_distance_computations;
      consider_source(source, source_distance);
    };
    for (GraphId source : candidate_graph.Neighbors(target)) {
      if (reachable[static_cast<size_t>(source)]) {
        evaluate_source(source);
      }
    }
    for (const NeighborCandidate& source :
         neighborhoods[static_cast<size_t>(target)]) {
      if (reachable[static_cast<size_t>(source.id)]) {
        consider_source(source.id, source.distance);
      }
    }

    std::priority_queue<NeighborCandidate, std::vector<NeighborCandidate>,
                        CandidateGreater>
        repair_candidates;
    const auto enqueue = [&](GraphId source) {
      if (!reachable[static_cast<size_t>(source)] ||
          !repair_visited.set(static_cast<size_t>(source))) {
        return;
      }
      const float source_distance =
          ValidateDistance(distance.symmetric_dis(source, target));
      ++local_stats.connectivity_distance_computations;
      repair_candidates.push({source, source_distance});
    };
    enqueue(navigation_point);
    size_t expanded = 0;
    while (!repair_candidates.empty() &&
           expanded < options_.candidate_pool_size) {
      const NeighborCandidate candidate = repair_candidates.top();
      repair_candidates.pop();
      consider_source(candidate.id, candidate.distance);
      ++expanded;
      for (const NeighborCandidate& neighbor :
           neighborhoods[static_cast<size_t>(candidate.id)]) {
        enqueue(neighbor.id);
      }
    }
    repair_visited.advance();
    if (best_source == kInvalidGraphId) {
      const size_t fallback_count =
          std::min(options_.candidate_pool_size, repair_sources.size());
      for (size_t candidate = 0; candidate < fallback_count; ++candidate) {
        evaluate_source(repair_sources[candidate]);
      }
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        best_source != kInvalidGraphId,
        "NSGBuilder: no capacity remains for connectivity repair");
    neighborhoods[static_cast<size_t>(best_source)].push_back(
        {target, best_distance});
    if (neighborhoods[static_cast<size_t>(best_source)].size() ==
        graph_capacity) {
      remove_source(best_source);
    }
    ++local_stats.connectivity_edges_added;
    newly_reachable.clear();
    MarkReachable(neighborhoods, target, &reachable, &reachable_count,
                  &newly_reachable);
    add_sources(newly_reachable);
  }

  for (size_t node = 0; node < node_count; ++node) {
    std::sort(neighborhoods[node].begin(), neighborhoods[node].end(),
              CandidateOrder);
    const std::vector<GraphId> ids = NeighborIds(neighborhoods[node]);
    graph.SetNeighbors(static_cast<GraphId>(node), ids);
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return graph;
}

}  // namespace hypervec
