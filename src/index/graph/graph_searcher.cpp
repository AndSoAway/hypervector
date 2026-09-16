/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <index/graph/visited_table.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace hypervec {

GraphCandidateQueue::GraphCandidateQueue(int capacity)
    : n(capacity), k(0), nvalid(0) {
  HYPERVEC_THROW_IF_NOT_MSG(capacity > 0,
                            "GraphCandidateQueue: capacity must be positive");
  ids.resize(static_cast<size_t>(capacity));
  dis.resize(static_cast<size_t>(capacity));
}

void GraphCandidateQueue::push(GraphId id, float distance) {
  using HeapComparator = CMax<float, GraphId>;
  if (k == n) {
    if (distance >= dis[0]) {
      return;
    }
    if (ids[0] != kInvalidGraphId) {
      --nvalid;
    }
    heap_pop<HeapComparator>(k--, dis.data(), ids.data());
  }
  heap_push<HeapComparator>(++k, dis.data(), ids.data(), distance, id);
  ++nvalid;
}

float GraphCandidateQueue::max() const {
  HYPERVEC_THROW_IF_NOT_MSG(k > 0, "GraphCandidateQueue::max: queue is empty");
  return dis[0];
}

int GraphCandidateQueue::size() const { return nvalid; }

void GraphCandidateQueue::clear() { nvalid = k = 0; }

#ifdef __AVX512F__

int GraphCandidateQueue::PopMin(float* minimum_distance) {
  assert(k > 0);
  static_assert(std::is_same_v<GraphId, int32_t>);

  int32_t minimum_index = -1;
  float minimum = std::numeric_limits<float>::infinity();
  __m512i minimum_indices = _mm512_set1_epi32(-1);
  __m512 minimum_distances =
      _mm512_set1_ps(std::numeric_limits<float>::infinity());
  __m512i current_indices =
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i offset = _mm512_set1_epi32(16);

  const int k16 = (k / 16) * 16;
  for (size_t i = 0; i < static_cast<size_t>(k16); i += 16) {
    const __m512i indices =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ids.data() + i));
    const __m512 distances = _mm512_loadu_ps(dis.data() + i);
    const __mmask16 invalid_mask =
        _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);
    const __mmask16 distance_mask =
        _mm512_cmp_ps_mask(minimum_distances, distances, _CMP_LT_OS);
    const __mmask16 final_mask = invalid_mask | distance_mask;
    minimum_indices =
        _mm512_mask_blend_epi32(final_mask, current_indices, minimum_indices);
    minimum_distances =
        _mm512_mask_blend_ps(final_mask, distances, minimum_distances);
    current_indices = _mm512_add_epi32(current_indices, offset);
  }

  if (k16 != k) {
    const __mmask16 remaining_mask = (1 << (k - k16)) - 1;
    const __m512i indices = _mm512_mask_loadu_epi32(
        _mm512_set1_epi32(-1), remaining_mask, ids.data() + k16);
    const __m512 distances =
        _mm512_maskz_loadu_ps(remaining_mask, dis.data() + k16);
    const __mmask16 invalid_mask =
        _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);
    const __mmask16 distance_mask =
        _mm512_cmp_ps_mask(minimum_distances, distances, _CMP_LT_OS);
    const __mmask16 final_mask = invalid_mask | distance_mask;
    minimum_indices =
        _mm512_mask_blend_epi32(final_mask, current_indices, minimum_indices);
    minimum_distances =
        _mm512_mask_blend_ps(final_mask, distances, minimum_distances);
  }

  minimum = _mm512_reduce_min_ps(minimum_distances);
  const __mmask16 minimum_mask =
      _mm512_cmpeq_ps_mask(minimum_distances, _mm512_set1_ps(minimum));
  minimum_index = _mm512_mask_reduce_max_epi32(minimum_mask, minimum_indices);
  if (minimum_index == -1) {
    return -1;
  }
  if (minimum_distance != nullptr) {
    *minimum_distance = minimum;
  }
  const GraphId result = ids[minimum_index];
  ids[minimum_index] = kInvalidGraphId;
  --nvalid;
  return result;
}

#elif __AVX2__

int GraphCandidateQueue::PopMin(float* minimum_distance) {
  assert(k > 0);
  static_assert(std::is_same_v<GraphId, int32_t>);

  int32_t minimum_index = -1;
  float minimum = std::numeric_limits<float>::infinity();
  size_t i = 0;
  __m256i minimum_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
  __m256 minimum_distances =
      _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256i offset = _mm256_set1_epi32(8);

  const int k8 = (k / 8) * 8;
  for (; i < static_cast<size_t>(k8); i += 8) {
    const __m256i indices =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids.data() + i));
    const __m256 distances = _mm256_loadu_ps(dis.data() + i);
    const __m256i invalid_mask =
        _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);
    const __m256i distance_mask = _mm256_castps_si256(
        _mm256_cmp_ps(minimum_distances, distances, _CMP_LT_OS));
    const __m256 final_mask =
        _mm256_castsi256_ps(_mm256_or_si256(invalid_mask, distance_mask));
    minimum_indices = _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(current_indices),
                         _mm256_castsi256_ps(minimum_indices), final_mask));
    minimum_distances =
        _mm256_blendv_ps(distances, minimum_distances, final_mask);
    current_indices = _mm256_add_epi32(current_indices, offset);
  }

  int32_t index_lanes[8];
  float distance_lanes[8];
  _mm256_storeu_ps(distance_lanes, minimum_distances);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(index_lanes), minimum_indices);
  for (size_t lane = 0; lane < 8; ++lane) {
    if (minimum > distance_lanes[lane] || (minimum == distance_lanes[lane] &&
                                           minimum_index < index_lanes[lane])) {
      minimum_index = index_lanes[lane];
      minimum = distance_lanes[lane];
    }
  }
  for (; i < static_cast<size_t>(k); ++i) {
    if (ids[i] != kInvalidGraphId && dis[i] <= minimum) {
      minimum = dis[i];
      minimum_index = static_cast<int32_t>(i);
    }
  }
  if (minimum_index == -1) {
    return -1;
  }
  if (minimum_distance != nullptr) {
    *minimum_distance = minimum;
  }
  const GraphId result = ids[minimum_index];
  ids[minimum_index] = kInvalidGraphId;
  --nvalid;
  return result;
}

#else

int GraphCandidateQueue::PopMin(float* minimum_distance) {
  assert(k > 0);
  int index = k - 1;
  while (index >= 0 && ids[index] == kInvalidGraphId) {
    --index;
  }
  if (index == -1) {
    return -1;
  }
  int minimum_index = index;
  float minimum = dis[index];
  for (--index; index >= 0; --index) {
    if (ids[index] != kInvalidGraphId && dis[index] < minimum) {
      minimum = dis[index];
      minimum_index = index;
    }
  }
  if (minimum_distance != nullptr) {
    *minimum_distance = minimum;
  }
  const GraphId result = ids[minimum_index];
  ids[minimum_index] = kInvalidGraphId;
  --nvalid;
  return result;
}

#endif

int GraphCandidateQueue::CountBelow(float threshold) const {
  int count = 0;
  for (int i = 0; i < k; ++i) {
    if (dis[i] < threshold) {
      ++count;
    }
  }
  return count;
}

void GraphSearchStats::Reset() noexcept { *this = {}; }

void GraphSearchStats::Combine(const GraphSearchStats& other) noexcept {
  queries += other.queries;
  exhausted_queries += other.exhausted_queries;
  distance_computations += other.distance_computations;
  visited_nodes += other.visited_nodes;
  expanded_nodes += other.expanded_nodes;
  traversed_edges += other.traversed_edges;
}

int GraphSearcher::Search(DistanceComputer& distance, ResultHandler& results,
                          GraphCandidateQueue& candidates,
                          VisitedTable& visited,
                          const GraphSearchOptions& options,
                          GraphSearchStats* stats,
                          int initial_result_count) const {
  HYPERVEC_THROW_IF_NOT_MSG(options.ef_search > 0,
                            "GraphSearcher: ef_search must be positive");
  GraphSearchStats local_stats;
  int result_count = initial_result_count;
  float threshold = results.threshold;

  for (int i = 0; i < candidates.k; ++i) {
    const GraphId node = candidates.ids[i];
    HYPERVEC_THROW_IF_NOT_MSG(
        node >= 0 && static_cast<size_t>(node) < graph_.NodeCount(),
        "GraphSearcher: initial candidate is outside the graph");
    const float node_distance = candidates.dis[i];
    if (options.selector == nullptr || options.selector->IsMember(node)) {
      if (node_distance < threshold && results.AddResult(node_distance, node)) {
        threshold = results.threshold;
      }
    }
    if (visited.set(static_cast<size_t>(node))) {
      ++local_stats.visited_nodes;
    }
  }

  int steps = 0;
  while (candidates.size() > 0) {
    float current_distance = 0.0f;
    const GraphId current = candidates.PopMin(&current_distance);
    HYPERVEC_THROW_IF_NOT_MSG(current >= 0,
                              "GraphSearcher: candidate queue is corrupt");

    if (options.check_relative_distance &&
        candidates.CountBelow(current_distance) >= options.ef_search) {
      break;
    }

    const GraphNeighborView neighbors = graph_.Neighbors(current);
    local_stats.traversed_edges += neighbors.size();
    for (GraphId neighbor : neighbors) {
      visited.prefetch(static_cast<size_t>(neighbor));
    }

    int buffered_count = 0;
    GraphId buffered_ids[4];
    threshold = results.threshold;
    const auto add_candidate = [&](GraphId node, float node_distance) {
      if (options.selector == nullptr || options.selector->IsMember(node)) {
        if (node_distance < threshold &&
            results.AddResult(node_distance, node)) {
          threshold = results.threshold;
          ++result_count;
        }
      }
      candidates.push(node, node_distance);
    };

    for (GraphId neighbor : neighbors) {
      if (!visited.set(static_cast<size_t>(neighbor))) {
        continue;
      }
      ++local_stats.visited_nodes;
      buffered_ids[buffered_count++] = neighbor;
      if (buffered_count == 4) {
        float distances[4];
        distance.distances_batch_4(
            buffered_ids[0], buffered_ids[1], buffered_ids[2], buffered_ids[3],
            distances[0], distances[1], distances[2], distances[3]);
        for (size_t offset = 0; offset < 4; ++offset) {
          add_candidate(buffered_ids[offset], distances[offset]);
        }
        local_stats.distance_computations += 4;
        buffered_count = 0;
      }
    }
    for (int offset = 0; offset < buffered_count; ++offset) {
      add_candidate(buffered_ids[offset], distance(buffered_ids[offset]));
      ++local_stats.distance_computations;
    }

    ++steps;
    ++local_stats.expanded_nodes;
    if (!options.check_relative_distance && steps > options.ef_search) {
      break;
    }
  }

  ++local_stats.queries;
  if (candidates.size() == 0) {
    ++local_stats.exhausted_queries;
  }
  if (stats != nullptr) {
    stats->Combine(local_stats);
  }
  return result_count;
}

}  // namespace hypervec
