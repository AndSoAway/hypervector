/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/neighbor_pruner.h>
#include <index/graph/nn_descent_builder.h>
#include <index/graph/visited_table.h>
#include <omp.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

struct TrackedNeighbor {
  GraphId id;
  float distance;
  bool old = false;
};

using NeighborList = std::vector<TrackedNeighbor>;
using SampleList = std::vector<NeighborCandidate>;

template <typename Neighbor>
bool CandidateOrder(const Neighbor& lhs, const Neighbor& rhs) noexcept {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

void ValidateDistance(float distance) {
  HYPERVEC_THROW_IF_NOT_MSG(
      !std::isnan(distance),
      "NNDescentBuilder: distance computation returned NaN");
}

std::vector<GraphId> NeighborIds(const NeighborList& neighbors) {
  std::vector<GraphId> ids;
  ids.reserve(neighbors.size());
  for (const TrackedNeighbor& neighbor : neighbors) {
    ids.push_back(neighbor.id);
  }
  return ids;
}

size_t CountNewNeighbors(const NeighborList& previous,
                         const NeighborList& replacement) {
  size_t updates = 0;
  for (const TrackedNeighbor& candidate : replacement) {
    const auto found =
        std::find_if(previous.begin(), previous.end(),
                     [&](const auto& old) { return old.id == candidate.id; });
    if (found == previous.end()) {
      ++updates;
    }
  }
  return updates;
}

void CompactSamples(SampleList* samples, size_t limit, NNDescentStats* stats) {
  std::sort(samples->begin(), samples->end(),
            CandidateOrder<NeighborCandidate>);
  samples->erase(std::unique(samples->begin(), samples->end(),
                             [](const auto& lhs, const auto& rhs) {
                               return lhs.id == rhs.id;
                             }),
                 samples->end());
  stats->peak_sampled_neighbors =
      std::max(stats->peak_sampled_neighbors, samples->size());
  if (samples->size() > limit) {
    stats->sampled_neighbors_trimmed += samples->size() - limit;
    samples->resize(limit);
  }
}

void InsertCandidate(GraphId source, const TrackedNeighbor& candidate,
                     size_t degree, NeighborList* neighbors) {
  if (candidate.id == source) {
    return;
  }
  if (std::any_of(neighbors->begin(), neighbors->end(),
                  [&](const auto& item) { return item.id == candidate.id; })) {
    return;
  }
  if (neighbors->size() == degree &&
      !CandidateOrder(candidate, neighbors->back())) {
    return;
  }
  const auto position =
      std::lower_bound(neighbors->begin(), neighbors->end(), candidate,
                       CandidateOrder<TrackedNeighbor>);
  neighbors->insert(position, candidate);
  if (neighbors->size() > degree) {
    neighbors->pop_back();
  }
}

size_t SampleBounded(std::mt19937_64* random, size_t bound) {
  const uint64_t unsigned_bound = static_cast<uint64_t>(bound);
  const uint64_t rejection_threshold =
      (uint64_t{0} - unsigned_bound) % unsigned_bound;
  uint64_t sample = 0;
  do {
    sample = (*random)();
  } while (sample < rejection_threshold);
  return static_cast<size_t>(sample % unsigned_bound);
}

struct JoinedPair {
  GraphId first;
  GraphId second;
  float distance;
};

std::vector<JoinedPair> JoinSamples(const SampleList& recent,
                                    const SampleList& previous,
                                    DistanceComputer& distance) {
  std::vector<JoinedPair> pairs;
  for (size_t lhs = 0; lhs < recent.size(); ++lhs) {
    const GraphId first = recent[lhs].id;
    for (size_t rhs = 0; rhs < lhs; ++rhs) {
      const GraphId second = recent[rhs].id;
      if (first != second) {
        const float value = distance.symmetric_dis(first, second);
        ValidateDistance(value);
        pairs.push_back({first, second, value});
      }
    }
    for (const NeighborCandidate& old : previous) {
      const GraphId second = old.id;
      if (first != second) {
        const float value = distance.symmetric_dis(first, second);
        ValidateDistance(value);
        pairs.push_back({first, second, value});
      }
    }
  }
  return pairs;
}

// Evaluate independent distances in parallel, but replay each pivot's pairs
// in the original order. The graph and stats remain stable across thread
// counts.
bool RefineParallel(const std::vector<SampleList>& recent,
                    const std::vector<SampleList>& previous, size_t degree,
                    size_t sample_limit, size_t requested_threads,
                    const NNDescentBuilder::DistanceFactory& factory,
                    std::vector<NeighborList>* refined, NNDescentStats* stats) {
  constexpr size_t kPairBudget = (64U * 1024U * 1024U) / sizeof(JoinedPair);
  const size_t worst_pairs =
      sample_limit > 0 && sample_limit > kPairBudget / sample_limit / 2
          ? kPairBudget
          : std::max(size_t{1}, size_t{2} * sample_limit * sample_limit);
  const size_t block_size =
      std::min({recent.size(), std::max(size_t{1}, kPairBudget / worst_pairs),
                requested_threads * size_t{4}});
  const size_t worker_count = std::min(requested_threads, block_size);
  if (worker_count < 2) {
    return false;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      static_cast<bool>(factory),
      "NNDescentBuilder: parallel build requires a distance factory");
  std::vector<std::unique_ptr<DistanceComputer>> worker_distances;
  worker_distances.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    worker_distances.push_back(factory());
    HYPERVEC_THROW_IF_NOT_MSG(
        worker_distances.back() != nullptr,
        "NNDescentBuilder: distance factory returned null");
  }

  std::vector<std::vector<JoinedPair>> pending(block_size);
  std::exception_ptr error;
  std::mutex error_mutex;
  std::atomic<bool> failed{false};
  const auto record_error = [&] {
    std::lock_guard<std::mutex> guard(error_mutex);
    if (error == nullptr) {
      error = std::current_exception();
    }
    failed.store(true);
  };
#pragma omp parallel num_threads(static_cast<int>(worker_count))
  {
    DistanceComputer& distance = *worker_distances[omp_get_thread_num()];
    for (size_t first = 0; first < recent.size(); first += block_size) {
      const auto count = static_cast<std::ptrdiff_t>(
          std::min(block_size, recent.size() - first));
#pragma omp for schedule(static)
      for (std::ptrdiff_t offset = 0; offset < count; ++offset) {
        if (!failed.load()) {
          try {
            const size_t pivot = first + static_cast<size_t>(offset);
            pending[static_cast<size_t>(offset)] =
                JoinSamples(recent[pivot], previous[pivot], distance);
          } catch (...) {
            record_error();
          }
        }
      }
#pragma omp single
      {
        if (!failed.load()) {
          try {
            for (std::ptrdiff_t offset = 0; offset < count; ++offset) {
              for (const JoinedPair& pair :
                   pending[static_cast<size_t>(offset)]) {
                InsertCandidate(pair.first, {pair.second, pair.distance, false},
                                degree,
                                &(*refined)[static_cast<size_t>(pair.first)]);
                InsertCandidate(pair.second, {pair.first, pair.distance, false},
                                degree,
                                &(*refined)[static_cast<size_t>(pair.second)]);
                ++stats->refinement_distance_computations;
              }
              pending[static_cast<size_t>(offset)].clear();
            }
          } catch (...) {
            record_error();
          }
        }
      }
      if (failed.load()) {
        break;
      }
    }
  }
  if (error != nullptr) {
    std::rethrow_exception(error);
  }
  return true;
}

}  // namespace

void NNDescentStats::Reset() noexcept { *this = {}; }

void NNDescentStats::Combine(const NNDescentStats& other) noexcept {
  iterations += other.iterations;
  initial_distance_computations += other.initial_distance_computations;
  refinement_distance_computations += other.refinement_distance_computations;
  neighbor_updates += other.neighbor_updates;
  sampled_old_neighbors += other.sampled_old_neighbors;
  sampled_new_neighbors += other.sampled_new_neighbors;
  sampled_neighbors_trimmed += other.sampled_neighbors_trimmed;
  peak_sampled_neighbors =
      std::max(peak_sampled_neighbors, other.peak_sampled_neighbors);
  converged = converged || other.converged;
}

NNDescentBuilder::NNDescentBuilder(NNDescentOptions options)
    : options_(options) {
  HYPERVEC_THROW_IF_NOT_MSG(options_.max_degree > 0,
                            "NNDescentBuilder: max_degree must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.max_iterations > 0,
      "NNDescentBuilder: max_iterations must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(options_.convergence_threshold) &&
          options_.convergence_threshold >= 0.0 &&
          options_.convergence_threshold <= 1.0,
      "NNDescentBuilder: convergence_threshold must be in [0, 1]");
  HYPERVEC_THROW_IF_NOT_MSG(std::isfinite(options_.sample_rate) &&
                                options_.sample_rate > 0.0 &&
                                options_.sample_rate <= 1.0,
                            "NNDescentBuilder: sample_rate must be in (0, 1]");
  HYPERVEC_THROW_IF_NOT_MSG(
      options_.build_threads > 0 &&
          options_.build_threads <=
              static_cast<size_t>(std::numeric_limits<int>::max()),
      "NNDescentBuilder: build_threads must be in [1, INT_MAX]");
}

MutableBoundedGraph NNDescentBuilder::Build(DistanceComputer& distance,
                                            size_t node_count,
                                            NNDescentStats* stats) const {
  return Build(distance, node_count, stats, {});
}

MutableBoundedGraph NNDescentBuilder::Build(
    DistanceComputer& distance, size_t node_count, NNDescentStats* stats,
    const DistanceFactory& distance_factory) const {
  constexpr size_t kGraphCapacity =
      static_cast<size_t>((std::numeric_limits<GraphId>::max)()) + 1;
  HYPERVEC_THROW_IF_NOT_MSG(
      node_count <= kGraphCapacity,
      "NNDescentBuilder: node count exceeds GraphId capacity");

  MutableBoundedGraph graph(node_count, options_.max_degree);
  NNDescentStats local_stats;
  if (node_count <= 1) {
    local_stats.converged = true;
    if (stats != nullptr) {
      stats->Combine(local_stats);
    }
    return graph;
  }

  const size_t degree = std::min(options_.max_degree, node_count - 1);
  std::vector<NeighborList> neighborhoods(node_count);
  std::mt19937_64 random(options_.random_seed);
  VisitedTable sampled(node_count, false);

  for (size_t node = 0; node < node_count; ++node) {
    std::vector<GraphId> initial;
    initial.reserve(degree);
    sampled.set(node);
    if (degree == node_count - 1) {
      for (size_t candidate = 0; candidate < node_count; ++candidate) {
        if (candidate != node) {
          initial.push_back(static_cast<GraphId>(candidate));
        }
      }
    } else {
      while (initial.size() < degree) {
        const size_t candidate = SampleBounded(&random, node_count);
        if (sampled.set(candidate)) {
          initial.push_back(static_cast<GraphId>(candidate));
        }
      }
    }

    NeighborList& neighbors = neighborhoods[node];
    neighbors.reserve(degree);
    for (GraphId candidate : initial) {
      const float candidate_distance =
          distance.symmetric_dis(static_cast<GraphId>(node), candidate);
      ValidateDistance(candidate_distance);
      neighbors.push_back({candidate, candidate_distance, false});
      ++local_stats.initial_distance_computations;
    }
    std::sort(neighbors.begin(), neighbors.end(),
              CandidateOrder<TrackedNeighbor>);
    sampled.advance();
  }

  const size_t edge_slots =
      mul_no_overflow(node_count, degree, "NNDescentBuilder edge slots");
  const size_t sample_limit =
      mul_no_overflow(degree, size_t{2}, "NNDescentBuilder sample limit");
  std::mt19937_64 sampling_random(options_.random_seed ^ 0xD1B54A32D192ED03ULL);
  std::bernoulli_distribution sample_edge(options_.sample_rate);
  for (size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
    std::vector<SampleList> old_samples(node_count);
    std::vector<SampleList> new_samples(node_count);
    for (size_t node = 0; node < node_count; ++node) {
      for (TrackedNeighbor& neighbor : neighborhoods[node]) {
        if (!sample_edge(sampling_random)) {
          continue;
        }
        auto& samples = neighbor.old ? old_samples : new_samples;
        samples[node].push_back({neighbor.id, neighbor.distance});
        samples[static_cast<size_t>(neighbor.id)].push_back(
            {static_cast<GraphId>(node), neighbor.distance});
        if (neighbor.old) {
          local_stats.sampled_old_neighbors += 2;
        } else {
          local_stats.sampled_new_neighbors += 2;
          neighbor.old = true;
        }
      }
    }

    for (size_t node = 0; node < node_count; ++node) {
      CompactSamples(&old_samples[node], sample_limit, &local_stats);
      CompactSamples(&new_samples[node], sample_limit, &local_stats);
    }

    std::vector<NeighborList> refined = neighborhoods;
    size_t iteration_updates = 0;
    const bool parallel =
        options_.build_threads > 1 &&
        RefineParallel(new_samples, old_samples, degree, sample_limit,
                       options_.build_threads, distance_factory, &refined,
                       &local_stats);
    if (!parallel) {
      for (size_t pivot = 0; pivot < node_count; ++pivot) {
        const SampleList& sampled_new = new_samples[pivot];
        const SampleList& sampled_old = old_samples[pivot];
        for (size_t lhs = 0; lhs < sampled_new.size(); ++lhs) {
          const GraphId first = sampled_new[lhs].id;
          for (size_t rhs = 0; rhs < lhs; ++rhs) {
            const GraphId second = sampled_new[rhs].id;
            if (first == second) {
              continue;
            }
            const float candidate_distance =
                distance.symmetric_dis(first, second);
            ValidateDistance(candidate_distance);
            InsertCandidate(first, {second, candidate_distance, false}, degree,
                            &refined[static_cast<size_t>(first)]);
            InsertCandidate(second, {first, candidate_distance, false}, degree,
                            &refined[static_cast<size_t>(second)]);
            ++local_stats.refinement_distance_computations;
          }
          for (const NeighborCandidate& old : sampled_old) {
            const GraphId second = old.id;
            if (first == second) {
              continue;
            }
            const float candidate_distance =
                distance.symmetric_dis(first, second);
            ValidateDistance(candidate_distance);
            InsertCandidate(first, {second, candidate_distance, false}, degree,
                            &refined[static_cast<size_t>(first)]);
            InsertCandidate(second, {first, candidate_distance, false}, degree,
                            &refined[static_cast<size_t>(second)]);
            ++local_stats.refinement_distance_computations;
          }
        }
      }
    }

    for (size_t node = 0; node < node_count; ++node) {
      iteration_updates +=
          CountNewNeighbors(neighborhoods[node], refined[node]);
    }

    neighborhoods = std::move(refined);
    ++local_stats.iterations;
    local_stats.neighbor_updates += iteration_updates;
    if (static_cast<double>(iteration_updates) <=
        options_.convergence_threshold * static_cast<double>(edge_slots)) {
      local_stats.converged = true;
      break;
    }
  }

  for (size_t node = 0; node < node_count; ++node) {
    const std::vector<GraphId> ids = NeighborIds(neighborhoods[node]);
    graph.SetNeighbors(static_cast<GraphId>(node), ids);
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return graph;
}

}  // namespace hypervec
