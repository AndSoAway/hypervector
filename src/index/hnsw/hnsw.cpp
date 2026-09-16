/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <index/graph/neighbor_pruner.h>
#include <index/hnsw/hnsw.h>
#include <index/hnsw/hnsw_graph_adapter.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/visited_table.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distance_computer.h>
#include <utils/selector/id_selector.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hypervec {

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

int HNSW::NbNeighbors(int layer_no) const {
  HYPERVEC_THROW_IF_NOT(layer_no + 1 < cum_nneighbor_per_level.size());
  return cum_nneighbor_per_level[layer_no + 1] -
         cum_nneighbor_per_level[layer_no];
}

void HNSW::SetNbNeighbors(int level_no, int n) {
  HYPERVEC_THROW_IF_NOT(levels.size() == 0);
  int cur_n = NbNeighbors(level_no);
  for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
    cum_nneighbor_per_level[i] += n - cur_n;
  }
}

int HNSW::CumNbNeighbors(int layer_no) const {
  HYPERVEC_CHECK_RANGE_DEBUG(layer_no, 0, (int)cum_nneighbor_per_level.size());
  return cum_nneighbor_per_level[layer_no];
}

void HNSW::NeighborRange(idx_t no, int layer_no, size_t* begin,
                         size_t* end) const {
  HYPERVEC_CHECK_RANGE_DEBUG(no, 0, (idx_t)offsets.size());
  HYPERVEC_CHECK_RANGE_DEBUG(layer_no, 0,
                             (int)cum_nneighbor_per_level.size() - 1);
  size_t o = offsets[no];
  *begin = o + CumNbNeighbors(layer_no);
  *end = o + CumNbNeighbors(layer_no + 1);
}

HNSW::HNSW(int M) : rng(12345) {
  SetDefaultProbas(M, 1.0 / log(M));
  offsets.push_back(0);
}

int HNSW::RandomLevel() {
  double f = rng.rand_float();
  // could be a bit faster with bisection
  for (int level = 0; level < assign_probas.size(); level++) {
    if (f < assign_probas[level]) {
      return level;
    }
    f -= assign_probas[level];
  }
  // happens with exponentially low probability
  return assign_probas.size() - 1;
}

void HNSW::SetDefaultProbas(int M, float levelMult) {
  int nn = 0;
  cum_nneighbor_per_level.push_back(0);
  for (int level = 0;; level++) {
    float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
    if (proba < 1e-9) {
      break;
    }
    assign_probas.push_back(proba);
    nn += level == 0 ? M * 2 : M;
    cum_nneighbor_per_level.push_back(nn);
  }
}

void HNSW::ClearNeighborTables(int level) {
  for (int i = 0; i < levels.size(); i++) {
    size_t begin, end;
    NeighborRange(i, level, &begin, &end);
    for (size_t j = begin; j < end; j++) {
      neighbors[j] = -1;
    }
  }
}

void HNSW::Reset() {
  max_level = -1;
  entry_point = -1;
  offsets.clear();
  offsets.push_back(0);
  levels.clear();
  neighbors.clear();
}

void HNSW::PrintNeighborStats(int level) const {
  HYPERVEC_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
  printf("stats on level %d, max %d neighbors per vertex:\n", level,
         NbNeighbors(level));
  size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
    reduction(+ : tot_reciprocal) reduction(+ : n_node)
  for (int i = 0; i < levels.size(); i++) {
    if (levels[i] > level) {
      n_node++;
      size_t begin, end;
      NeighborRange(i, level, &begin, &end);
      std::unordered_set<int> neighset;
      for (size_t j = begin; j < end; j++) {
        if (neighbors[j] < 0) {
          break;
        }
        neighset.insert(neighbors[j]);
      }
      int n_neigh = neighset.size();
      int n_common = 0;
      int n_reciprocal = 0;
      for (size_t j = begin; j < end; j++) {
        storage_idx_t i2 = neighbors[j];
        if (i2 < 0) {
          break;
        }
        HYPERVEC_ASSERT(i2 != i);
        size_t begin2, end2;
        NeighborRange(i2, level, &begin2, &end2);
        for (size_t j2 = begin2; j2 < end2; j2++) {
          storage_idx_t i3 = neighbors[j2];
          if (i3 < 0) {
            break;
          }
          if (i3 == i) {
            n_reciprocal++;
            continue;
          }
          if (neighset.count(i3)) {
            neighset.erase(i3);
            n_common++;
          }
        }
      }
      tot_neigh += n_neigh;
      tot_common += n_common;
      tot_reciprocal += n_reciprocal;
    }
  }
  float normalizer = n_node;
  printf("   nb of nodes at that level %zd\n", n_node);
  printf("   neighbors per node: %.2f (%zd)\n", tot_neigh / normalizer,
         tot_neigh);
  printf("   nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
  printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
         tot_common / normalizer, tot_common);
}

void HNSW::FillWithRandomLinks(size_t n) {
  int max_level_2 = PrepareLevelTab(n);
  RandomGenerator rng2(456);

  for (int level = max_level_2 - 1; level >= 0; --level) {
    std::vector<int> elts;
    for (int i = 0; i < n; i++) {
      if (levels[i] > level) {
        elts.push_back(i);
      }
    }
    printf("linking %zd elements in level %d\n", elts.size(), level);

    if (elts.size() == 1) {
      continue;
    }

    for (int ii = 0; ii < elts.size(); ii++) {
      int i = elts[ii];
      size_t begin, end;
      NeighborRange(i, 0, &begin, &end);
      for (size_t j = begin; j < end; j++) {
        int other = 0;
        do {
          other = elts[rng2.rand_int(elts.size())];
        } while (other == i);

        neighbors[j] = other;
      }
    }
  }
}

int HNSW::PrepareLevelTab(size_t n, bool preset_levels) {
  size_t n0 = offsets.size() - 1;

  if (preset_levels) {
    HYPERVEC_ASSERT(n0 + n == levels.size());
  } else {
    HYPERVEC_ASSERT(n0 == levels.size());
    for (int i = 0; i < n; i++) {
      int pt_level = RandomLevel();
      levels.push_back(pt_level + 1);
    }
  }

  int max_level_2 = 0;
  for (int i = 0; i < n; i++) {
    int pt_level = levels[i + n0] - 1;
    if (pt_level > max_level_2) {
      max_level_2 = pt_level;
    }
    offsets.push_back(offsets.back() + CumNbNeighbors(pt_level + 1));
  }
  neighbors.resize(offsets.back(), -1);

  return max_level_2;
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSW::ShrinkNeighborList(DistanceComputer& qdis,
                              std::priority_queue<NodeDistFarther>& input,
                              std::vector<NodeDistFarther>& output,
                              int max_size, bool keep_max_size_level0) {
  HYPERVEC_THROW_IF_NOT_MSG(
      max_size > 0, "HNSW::ShrinkNeighborList: max_size must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      output.empty(),
      "HNSW::ShrinkNeighborList: output must initially be empty");
  std::vector<NeighborCandidate> candidates;
  candidates.reserve(input.size());
  while (!input.empty()) {
    const NodeDistFarther candidate = input.top();
    input.pop();
    candidates.push_back({static_cast<GraphId>(candidate.id), candidate.d});
  }
  const HnswHeuristicPruner pruner(keep_max_size_level0);
  const std::vector<NeighborCandidate> selected =
      pruner.Prune(candidates, static_cast<size_t>(max_size), qdis);
  output.reserve(output.size() + selected.size());
  for (const NeighborCandidate& candidate : selected) {
    output.emplace_back(candidate.distance, candidate.id);
  }
}

namespace {

using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
void ShrinkNeighborList(DistanceComputer& qdis,
                        std::priority_queue<NodeDistCloser>& resultSet1,
                        int max_size, bool keep_max_size_level0 = false) {
  if (resultSet1.size() < max_size) {
    return;
  }
  std::priority_queue<NodeDistFarther> resultSet;
  std::vector<NodeDistFarther> returnlist;

  while (resultSet1.size() > 0) {
    resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
    resultSet1.pop();
  }

  HNSW::ShrinkNeighborList(qdis, resultSet, returnlist, max_size,
                           keep_max_size_level0);

  for (NodeDistFarther curen2 : returnlist) {
    resultSet1.emplace(curen2.d, curen2.id);
  }
}

/// Add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(HNSW& hnsw, DistanceComputer& qdis, storage_idx_t src,
              storage_idx_t dest, int level,
              bool keep_max_size_level0 = false) {
  size_t begin, end;
  hnsw.NeighborRange(src, level, &begin, &end);
  if (hnsw.neighbors[end - 1] == -1) {
    // there is enough room, find a slot to Add it
    size_t i = end;
    while (i > begin) {
      if (hnsw.neighbors[i - 1] != -1) {
        break;
      }
      i--;
    }
    hnsw.neighbors[i] = dest;
    return;
  }

  // otherwise we let them fight out which to keep

  // copy to resultSet...
  std::priority_queue<NodeDistCloser> resultSet;
  resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
  for (size_t i = begin; i < end; i++) {  // HERE WAS THE BUG
    storage_idx_t neigh = hnsw.neighbors[i];
    resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
  }

  ShrinkNeighborList(qdis, resultSet, end - begin, keep_max_size_level0);

  // ...and back
  size_t i = begin;
  while (resultSet.size()) {
    hnsw.neighbors[i++] = resultSet.top().id;
    resultSet.pop();
  }
  // they may have shrunk more than just by 1 element
  while (i < end) {
    hnsw.neighbors[i++] = -1;
  }
}

}  // namespace

/// Search neighbors on a single level, starting from an entry point
void SearchNeighborsToAdd(HNSW& hnsw, DistanceComputer& qdis,
                          std::priority_queue<NodeDistCloser>& results,
                          int entry_point, float d_entry_point, int level,
                          VisitedTable& vt, bool reference_version) {
  // top is nearest candidate
  std::priority_queue<NodeDistFarther> candidates;

  NodeDistFarther ev(d_entry_point, entry_point);
  candidates.push(ev);
  results.emplace(d_entry_point, entry_point);
  vt.set(entry_point);

  while (!candidates.empty()) {
    // get nearest
    const NodeDistFarther& currEv = candidates.top();

    if (currEv.d > results.top().d) {
      break;
    }
    int currNode = currEv.id;
    candidates.pop();

    // loop over neighbors
    size_t begin, end;
    hnsw.NeighborRange(currNode, level, &begin, &end);

    // The reference version is not used, but kept here because:
    // 1. It is easier to switch back if the optimized version has a problem
    // 2. It serves as a starting point for new optimizations
    // 3. It helps understand the code
    // 4. It ensures the reference version is still compilable if the
    // optimized version changes
    // The reference and the optimized versions' results are compared in
    // test_hnsw.cpp
    if (reference_version) {
      // a reference version
      for (size_t i = begin; i < end; i++) {
        storage_idx_t nodeId = hnsw.neighbors[i];
        if (nodeId < 0) {
          break;
        }
        if (!vt.set(nodeId)) {
          continue;
        }

        float dis = qdis(nodeId);
        NodeDistFarther evE1(dis, nodeId);

        if (results.size() < hnsw.ef_construction || results.top().d > dis) {
          results.emplace(dis, nodeId);
          candidates.emplace(dis, nodeId);
          if (results.size() > hnsw.ef_construction) {
            results.pop();
          }
        }
      }
    } else {
      // a faster version

      // the following version processes 4 neighbors at a time
      auto update_with_candidate = [&](const storage_idx_t idx,
                                       const float dis) {
        if (results.size() < hnsw.ef_construction || results.top().d > dis) {
          results.emplace(dis, idx);
          candidates.emplace(dis, idx);
          if (results.size() > hnsw.ef_construction) {
            results.pop();
          }
        }
      };

      int n_buffered = 0;
      storage_idx_t buffered_ids[4];

      for (size_t j = begin; j < end; j++) {
        storage_idx_t nodeId = hnsw.neighbors[j];
        if (nodeId < 0) {
          break;
        }
        if (!vt.set(nodeId)) {
          continue;
        }

        buffered_ids[n_buffered] = nodeId;
        n_buffered += 1;

        if (n_buffered == 4) {
          float dis[4];
          qdis.distances_batch_4(buffered_ids[0], buffered_ids[1],
                                 buffered_ids[2], buffered_ids[3], dis[0],
                                 dis[1], dis[2], dis[3]);

          for (size_t id4 = 0; id4 < 4; id4++) {
            update_with_candidate(buffered_ids[id4], dis[id4]);
          }

          n_buffered = 0;
        }
      }

      // process leftovers
      for (size_t icnt = 0; icnt < n_buffered; icnt++) {
        float dis = qdis(buffered_ids[icnt]);
        update_with_candidate(buffered_ids[icnt], dis);
      }
    }
  }

  vt.advance();
}

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::AddLinksStartingFrom(DistanceComputer& ptdis, storage_idx_t pt_id,
                                storage_idx_t nearest, float d_nearest,
                                int level, omp_lock_t* locks, VisitedTable& vt,
                                bool keep_max_size_level0) {
  std::priority_queue<NodeDistCloser> link_targets;

  SearchNeighborsToAdd(*this, ptdis, link_targets, nearest, d_nearest, level,
                       vt);

  // but we can afford only this many neighbors
  int M = NbNeighbors(level);

  ::hypervec::ShrinkNeighborList(ptdis, link_targets, M, keep_max_size_level0);

  std::vector<storage_idx_t> neighbors_to_add;
  neighbors_to_add.reserve(link_targets.size());
  while (!link_targets.empty()) {
    storage_idx_t other_id = link_targets.top().id;
    add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);
    neighbors_to_add.push_back(other_id);
    link_targets.pop();
  }

  omp_unset_lock(&locks[pt_id]);
  for (storage_idx_t other_id : neighbors_to_add) {
    omp_set_lock(&locks[other_id]);
    add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
    omp_unset_lock(&locks[other_id]);
  }
  omp_set_lock(&locks[pt_id]);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::AddWithLocks(DistanceComputer& ptdis, int pt_level, int pt_id,
                        std::vector<omp_lock_t>& locks, VisitedTable& vt,
                        bool keep_max_size_level0) {
  storage_idx_t nearest = entry_point;
  if (nearest == -1) {  // avoid locking after the first point.
#pragma omp critical
    if (entry_point == -1) {  // double-check under lock.
      max_level = pt_level;
      entry_point = pt_id;
      // leave nearest = -1 to trigger early exit after critical block.
    } else {
      // else: Another thread set the entry point.
      nearest = entry_point;
    }
  }

  if (nearest < 0) {
    return;
  }

  omp_set_lock(&locks[pt_id]);

  int level = max_level;  // level at which we start adding neighbors
  float d_nearest = ptdis(nearest);

  //  greedy Search on upper levels
  for (; level > pt_level; level--) {
    GreedyUpdateNearest(*this, ptdis, level, nearest, d_nearest);
  }

  for (; level >= 0; level--) {
    AddLinksStartingFrom(ptdis, pt_id, nearest, d_nearest, level, locks.data(),
                         vt, keep_max_size_level0);
  }

  omp_unset_lock(&locks[pt_id]);

  if (pt_level > max_level) {
    max_level = pt_level;
    entry_point = pt_id;
  }
}

/**************************************************************
 * Searching
 **************************************************************/

using MinimaxHeap = HNSW::MinimaxHeap;
using Node = HNSW::Node;
using C = HNSW::C;

/** Helper to extract Search parameters from HNSW and SearchParameters */
static inline void extract_search_params(const HNSW& hnsw,
                                         const SearchParameters* params,
                                         bool& do_dis_check, int& ef_search,
                                         const IDSelector*& sel) {
  // can be overridden by Search params
  do_dis_check = hnsw.check_relative_distance;
  ef_search = hnsw.ef_search;
  sel = nullptr;
  if (params) {
    if (const SearchParametersHNSW* hnsw_params =
            dynamic_cast<const SearchParametersHNSW*>(params)) {
      do_dis_check = hnsw_params->check_relative_distance;
      ef_search = hnsw_params->ef_search;
    }
    sel = params->sel;
  }
}

/** Do a BFS on the candidates list */
int SearchFromCandidates(const HNSW& hnsw, DistanceComputer& qdis,
                         ResultHandler& res, MinimaxHeap& candidates,
                         VisitedTable& vt, HNSWStats& stats, int level,
                         int nres_in, const SearchParameters* params) {
  bool do_dis_check;
  int ef_search;
  const IDSelector* sel;
  extract_search_params(hnsw, params, do_dis_check, ef_search, sel);
  const HnswGraphLayerView graph(hnsw, level);
  const GraphSearcher searcher(graph);
  GraphSearchStats graph_stats;
  const int result_count = searcher.Search(
      qdis, res, candidates, vt,
      GraphSearchOptions{ef_search, do_dis_check, sel}, &graph_stats, nres_in);
  if (level == 0) {
    stats.n1 += graph_stats.queries;
    stats.n2 += graph_stats.exhausted_queries;
    stats.ndis += graph_stats.distance_computations;
    stats.nhops += graph_stats.expanded_nodes;
  }
  return result_count;
}

int search_from_candidates_panorama(const HNSW& hnsw, const IndexHNSW* index,
                                    DistanceComputer& qdis, ResultHandler& res,
                                    MinimaxHeap& candidates, VisitedTable& vt,
                                    HNSWStats& stats, int level, int nres_in,
                                    const SearchParameters* params) {
  return nres_in;
}

std::priority_queue<HNSW::Node> SearchFromCandidateUnbounded(
    const HNSW& hnsw, const Node& node, DistanceComputer& qdis, int ef,
    VisitedTable* vt, HNSWStats& stats) {
  int ndis = 0;
  std::priority_queue<Node> top_candidates;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

  top_candidates.push(node);
  candidates.push(node);

  vt->set(node.second);

  while (!candidates.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidates.top();

    if (d0 > top_candidates.top().first) {
      break;
    }

    candidates.pop();

    size_t begin, end;
    hnsw.NeighborRange(v0, 0, &begin, &end);

    // a faster version: reference version in unit test test_hnsw.cpp
    // the following version processes 4 neighbors at a time
    size_t jmax = begin;
    for (size_t j = begin; j < end; j++) {
      int v1 = hnsw.neighbors[j];
      if (v1 < 0) {
        break;
      }

      vt->prefetch(v1);
      jmax += 1;
    }

    int counter = 0;
    size_t saved_j[4];

    auto add_to_heap = [&](const size_t idx, const float dis) {
      if (top_candidates.top().first > dis || top_candidates.size() < ef) {
        candidates.emplace(dis, idx);
        top_candidates.emplace(dis, idx);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
      }
    };

    for (size_t j = begin; j < jmax; j++) {
      int v1 = hnsw.neighbors[j];

      saved_j[counter] = v1;
      counter += vt->set(v1) ? 1 : 0;

      if (counter == 4) {
        float dis[4];
        qdis.distances_batch_4(saved_j[0], saved_j[1], saved_j[2], saved_j[3],
                               dis[0], dis[1], dis[2], dis[3]);

        for (size_t id4 = 0; id4 < 4; id4++) {
          add_to_heap(saved_j[id4], dis[id4]);
        }

        ndis += 4;

        counter = 0;
      }
    }

    for (size_t icnt = 0; icnt < counter; icnt++) {
      float dis = qdis(saved_j[icnt]);
      add_to_heap(saved_j[icnt], dis);

      ndis += 1;
    }

    stats.nhops += 1;
  }

  ++stats.n1;
  if (candidates.size() == 0) {
    ++stats.n2;
  }
  stats.ndis += ndis;

  return top_candidates;
}

/// greedily update a nearest vector at a given level
HNSWStats GreedyUpdateNearest(const HNSW& hnsw, DistanceComputer& qdis,
                              int level, storage_idx_t& nearest,
                              float& d_nearest) {
  HNSWStats stats;

  for (;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.NeighborRange(nearest, level, &begin, &end);

    size_t ndis = 0;

    // a faster version: reference version in unit test test_hnsw.cpp
    // the following version processes 4 neighbors at a time
    auto update_with_candidate = [&](const storage_idx_t idx, const float dis) {
      if (dis < d_nearest) {
        nearest = idx;
        d_nearest = dis;
      }
    };

    int n_buffered = 0;
    storage_idx_t buffered_ids[4];

    for (size_t j = begin; j < end; j++) {
      storage_idx_t v = hnsw.neighbors[j];
      if (v < 0) {
        break;
      }
      ndis += 1;

      buffered_ids[n_buffered] = v;
      n_buffered += 1;

      if (n_buffered == 4) {
        float dis[4];
        qdis.distances_batch_4(buffered_ids[0], buffered_ids[1],
                               buffered_ids[2], buffered_ids[3], dis[0], dis[1],
                               dis[2], dis[3]);

        for (size_t id4 = 0; id4 < 4; id4++) {
          update_with_candidate(buffered_ids[id4], dis[id4]);
        }

        n_buffered = 0;
      }
    }

    // process leftovers
    for (size_t icnt = 0; icnt < n_buffered; icnt++) {
      float dis = qdis(buffered_ids[icnt]);
      update_with_candidate(buffered_ids[icnt], dis);
    }

    // update stats
    stats.ndis += ndis;
    stats.nhops += 1;

    if (nearest == prev_nearest) {
      return stats;
    }
  }
}

namespace {
using MinimaxHeap = HNSW::MinimaxHeap;
using Node = HNSW::Node;
using C = HNSW::C;

// just used as a lower bound for the minmaxheap, but it is set for heap Search
int extract_k_from_ResultHandler(ResultHandler& res) {
  using RH = HeapBlockResultHandler<C>;
  if (auto hres = dynamic_cast<RH::SingleResultHandler*>(&res)) {
    return hres->k;
  }
  return 1;
}

}  // namespace

HNSWStats HNSW::Search(DistanceComputer& qdis, const IndexHNSW* index,
                       ResultHandler& res, VisitedTable& vt,
                       const SearchParameters* params) const {
  HNSWStats stats;
  if (entry_point == -1) {
    return stats;
  }
  int k = extract_k_from_ResultHandler(res);

  bool bounded_queue = this->search_bounded_queue;
  int ef_search = this->ef_search;
  if (params) {
    if (const SearchParametersHNSW* hnsw_params =
            dynamic_cast<const SearchParametersHNSW*>(params)) {
      bounded_queue = hnsw_params->bounded_queue;
      ef_search = hnsw_params->ef_search;
    }
  }
  HYPERVEC_THROW_IF_NOT_MSG(ef_search > 0,
                            "HNSW::Search: ef_search must be positive");

  //  greedy Search on upper levels
  storage_idx_t nearest = entry_point;
  float d_nearest = qdis(nearest);

  for (int level = max_level; level >= 1; level--) {
    HNSWStats local_stats =
        GreedyUpdateNearest(*this, qdis, level, nearest, d_nearest);
    stats.combine(local_stats);
  }

  int ef = std::max(ef_search, k);
  if (bounded_queue) {  // this is the most common branch, for now we only
    MinimaxHeap candidates(ef);

    candidates.push(nearest, d_nearest);

    if (!is_panorama) {
      SearchFromCandidates(*this, qdis, res, candidates, vt, stats, 0, 0,
                           params);
    } else {
      search_from_candidates_panorama(*this, index, qdis, res, candidates, vt,
                                      stats, 0, 0, params);
    }
  } else {
    std::priority_queue<Node> top_candidates = SearchFromCandidateUnbounded(
        *this, Node(d_nearest, nearest), qdis, ef, &vt, stats);

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }

    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      res.AddResult(d, label);
      top_candidates.pop();
    }
  }

  vt.advance();

  return stats;
}

void HNSW::SearchLevel0(DistanceComputer& qdis, ResultHandler& res,
                        idx_t nprobe, const storage_idx_t* nearest_i,
                        const float* nearest_d, int search_type,
                        HNSWStats& search_stats, VisitedTable& vt,
                        const SearchParameters* params) const {
  const HNSW& hnsw = *this;

  auto ef_search = hnsw.ef_search;
  if (params) {
    if (const SearchParametersHNSW* hnsw_params =
            dynamic_cast<const SearchParametersHNSW*>(params)) {
      ef_search = hnsw_params->ef_search;
    }
  }

  int k = extract_k_from_ResultHandler(res);

  if (search_type == 1) {
    int nres = 0;

    for (int j = 0; j < nprobe; j++) {
      storage_idx_t cj = nearest_i[j];

      if (cj < 0) {
        break;
      }

      if (vt.get(cj)) {
        continue;
      }

      int candidates_size = std::max(ef_search, k);
      MinimaxHeap candidates(candidates_size);

      candidates.push(cj, nearest_d[j]);

      nres = SearchFromCandidates(hnsw, qdis, res, candidates, vt, search_stats,
                                  0, nres, params);
      nres = std::min(nres, candidates_size);
    }
  } else if (search_type == 2) {
    int candidates_size = std::max(ef_search, static_cast<int>(k));
    candidates_size = std::max(candidates_size, static_cast<int>(nprobe));

    MinimaxHeap candidates(candidates_size);
    for (int j = 0; j < nprobe; j++) {
      storage_idx_t cj = nearest_i[j];

      if (cj < 0) {
        break;
      }
      candidates.push(cj, nearest_d[j]);
    }

    SearchFromCandidates(hnsw, qdis, res, candidates, vt, search_stats, 0, 0,
                         params);
  }
}

void HNSW::PermuteEntries(const idx_t* map) {
  // remap levels
  storage_idx_t n_total = levels.size();
  std::vector<storage_idx_t> imap(n_total);  // inverse mapping
  // map: new index -> old index
  // imap: old index -> new index
  for (int i = 0; i < n_total; i++) {
    assert(map[i] >= 0 && map[i] < n_total);
    imap[map[i]] = i;
  }
  if (entry_point != -1) {
    entry_point = imap[entry_point];
  }
  std::vector<int> new_levels(n_total);
  std::vector<size_t> new_offsets(n_total + 1);
  std::vector<storage_idx_t> new_neighbors(neighbors.size());
  size_t no = 0;
  for (int i = 0; i < n_total; i++) {
    storage_idx_t o = map[i];  // corresponding "old" index
    new_levels[i] = levels[o];
    for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
      storage_idx_t neigh = neighbors[j];
      new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
    }
    new_offsets[i + 1] = no;
  }
  assert(new_offsets[n_total] == offsets[n_total]);
  // swap everyone
  std::swap(levels, new_levels);
  std::swap(offsets, new_offsets);
  neighbors = std::move(new_neighbors);
}

}  // namespace hypervec
