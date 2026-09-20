/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/graph_searcher.h>
#include <index/hnsw/hnsw.h>
#include <index/hnsw/hnsw_graph_storage.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/visited_table.h>
#include <utils/common/result_handler.h>
#include <utils/distances/distance_computer.h>
#include <utils/selector/id_selector.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>

#include <type_traits>
#endif

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
  // This prevents number of neighbors at
  // level 0 from being shrunk to less than 2 * M.
  // This is essential for IndexHNSWCagra::copyFrom functionality
  std::vector<NodeDistFarther> outsiders;

  while (input.size() > 0) {
    NodeDistFarther v1 = input.top();
    input.pop();
    float dist_v1_q = v1.d;

    bool good = true;
    for (NodeDistFarther v2 : output) {
      float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

      if (dist_v1_v2 < dist_v1_q) {
        good = false;
        break;
      }
    }

    if (good) {
      output.push_back(v1);
      if (output.size() >= max_size) {
        return;
      }
    } else if (keep_max_size_level0) {
      outsiders.push_back(v1);
    }
  }
  size_t idx = 0;
  while (keep_max_size_level0 && (output.size() < max_size) &&
         (idx < outsiders.size())) {
    output.push_back(outsiders[idx++]);
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
              storage_idx_t dest, int level, bool keep_max_size_level0,
              const std::function<void(storage_idx_t)>& before_node_mutation) {
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
    if (before_node_mutation) {
      before_node_mutation(src);
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
  if (before_node_mutation) {
    before_node_mutation(src);
  }
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
void HNSW::AddLinksStartingFrom(
    DistanceComputer& ptdis, storage_idx_t pt_id, storage_idx_t nearest,
    float d_nearest, int level, omp_lock_t* locks, VisitedTable& vt,
    bool keep_max_size_level0,
    const std::function<void(storage_idx_t)>& before_node_mutation) {
  std::priority_queue<NodeDistCloser> link_targets;

  SearchNeighborsToAdd(*this, ptdis, link_targets, nearest, d_nearest, level,
                          vt);

  // but we can afford only this many neighbors
  int M = NbNeighbors(level);

  ::hypervec::ShrinkNeighborList(ptdis, link_targets, M,
                                   keep_max_size_level0);

  std::vector<storage_idx_t> neighbors_to_add;
  neighbors_to_add.reserve(link_targets.size());
  while (!link_targets.empty()) {
    storage_idx_t other_id = link_targets.top().id;
    add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0,
             before_node_mutation);
    neighbors_to_add.push_back(other_id);
    link_targets.pop();
  }

  omp_unset_lock(&locks[pt_id]);
  for (storage_idx_t other_id : neighbors_to_add) {
    omp_set_lock(&locks[other_id]);
    try {
      add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0,
               before_node_mutation);
    } catch (...) {
      omp_unset_lock(&locks[other_id]);
      omp_set_lock(&locks[pt_id]);
      throw;
    }
    omp_unset_lock(&locks[other_id]);
  }
  omp_set_lock(&locks[pt_id]);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::AddWithLocks(
    DistanceComputer& ptdis, int pt_level, int pt_id,
    std::vector<omp_lock_t>& locks, VisitedTable& vt, bool keep_max_size_level0,
    const std::function<void(storage_idx_t)>& before_node_mutation) {
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
  try {
    int level = max_level;  // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    //  greedy Search on upper levels
    for (; level > pt_level; level--) {
      GreedyUpdateNearest(*this, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
      AddLinksStartingFrom(ptdis, pt_id, nearest, d_nearest, level,
                           locks.data(), vt, keep_max_size_level0,
                           before_node_mutation);
    }
  } catch (...) {
    omp_unset_lock(&locks[pt_id]);
    throw;
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
  int nres = nres_in;
  int ndis = 0;

  bool do_dis_check;
  int ef_search;
  const IDSelector* sel;
  extract_search_params(hnsw, params, do_dis_check, ef_search, sel);

  C::T threshold = res.threshold;
  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    HYPERVEC_ASSERT(v1 >= 0);
    if (!sel || sel->IsMember(v1)) {
      if (d < threshold) {
        if (res.AddResult(d, v1)) {
          threshold = res.threshold;
        }
      }
    }
    vt.set(v1);
  }

  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;
    int v0 = candidates.PopMin(&d0);

    if (do_dis_check) {
      // tricky stopping condition: there are more that ef
      // distances that are processed already that are smaller
      // than d0

      int n_dis_below = candidates.CountBelow(d0);
      if (n_dis_below >= ef_search) {
        break;
      }
    }

    size_t begin, end;
    hnsw.NeighborRange(v0, level, &begin, &end);

    // a faster version: reference version in unit test test_hnsw.cpp
    // the following version processes 4 neighbors at a time
    size_t jmax = begin;
    for (size_t j = begin; j < end; j++) {
      int v1 = hnsw.neighbors[j];
      if (v1 < 0) {
        break;
      }

      vt.prefetch(v1);
      jmax += 1;
    }

    int counter = 0;
    size_t saved_j[4];

    threshold = res.threshold;

    auto add_to_heap = [&](const size_t idx, const float dis) {
      if (!sel || sel->IsMember(idx)) {
        if (dis < threshold) {
          if (res.AddResult(dis, idx)) {
            threshold = res.threshold;
            nres += 1;
          }
        }
      }
      candidates.push(idx, dis);
    };

    for (size_t j = begin; j < jmax; j++) {
      int v1 = hnsw.neighbors[j];

      saved_j[counter] = v1;
      counter += vt.set(v1) ? 1 : 0;

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

    nstep++;
    if (!do_dis_check && nstep > ef_search) {
      break;
    }
  }

  if (level == 0) {
    stats.n1++;
    if (candidates.size() == 0) {
      stats.n2++;
    }
    stats.ndis += ndis;
    stats.nhops += nstep;
  }

  return nres;
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

HNSWStats HNSW::Search(DistanceComputer& qdis, const IndexHNSW* /*index*/,
                       ResultHandler& res, VisitedTable& vt,
                       const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      !is_panorama,
      "HNSW::Search: panorama mode is not implemented in Hypervector");
  HNSWStats stats;
  if (entry_point == -1) {
    return stats;
  }
  int k = extract_k_from_ResultHandler(res);

  bool bounded_queue = this->search_bounded_queue;
  int ef_search = this->ef_search;
  bool check_relative_distance = this->check_relative_distance;
  if (params) {
    if (const SearchParametersHNSW* hnsw_params =
            dynamic_cast<const SearchParametersHNSW*>(params)) {
      bounded_queue = hnsw_params->bounded_queue;
      ef_search = hnsw_params->ef_search;
      check_relative_distance = hnsw_params->check_relative_distance;
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
  const HNSWGraphStorage graph(*this, 0, HNSWGraphValidation::kOnAccess);
  const GraphSearcher searcher(graph);
  const std::array<GraphSearchSeed, 1> seeds = {
      GraphSearchSeed{nearest, d_nearest}};
  GraphSearchOptions options;
  options.ef_search = static_cast<size_t>(ef);
  options.check_relative_distance = check_relative_distance;
  options.selector = params == nullptr ? nullptr : params->sel;
  options.frontier_policy = GraphSearchFrontierPolicy::kNavigationBound;
  options.max_expansions =
      check_relative_distance ? 0
                              : static_cast<size_t>(std::max(ef_search, 0)) + 1;
  options.max_candidates = bounded_queue ? static_cast<size_t>(ef) : 0;
  options.relative_distance_limit = static_cast<size_t>(ef_search);
  options.advance_visited = false;
  GraphSearchStats graph_stats;
  const std::vector<GraphSearchResult> results =
      searcher.Search(qdis, seeds, options, &vt, &graph_stats);

  const size_t result_count = std::min(results.size(), static_cast<size_t>(k));
  for (size_t result = 0; result < result_count; ++result) {
    res.AddResult(results[result].distance, results[result].id);
  }
  stats.n1 += graph_stats.queries;
  stats.n2 += graph_stats.exhausted_queries;
  stats.ndis += graph_stats.distance_computations;
  stats.nhops += graph_stats.expanded_nodes;

  vt.advance();

  return stats;
}

void HNSW::SearchLevel0(DistanceComputer& qdis, ResultHandler& res,
                        idx_t nprobe, const storage_idx_t* nearest_i,
                        const float* nearest_d, int search_type,
                        HNSWStats& search_stats, VisitedTable& vt,
                        const SearchParameters* params) const {
  int ef_search = this->ef_search;
  bool check_relative_distance = this->check_relative_distance;
  if (params) {
    if (const SearchParametersHNSW* hnsw_params =
            dynamic_cast<const SearchParametersHNSW*>(params)) {
      ef_search = hnsw_params->ef_search;
      check_relative_distance = hnsw_params->check_relative_distance;
    }
  }
  HYPERVEC_THROW_IF_NOT_MSG(ef_search > 0,
                            "HNSW::SearchLevel0: ef_search must be positive");

  const int k = extract_k_from_ResultHandler(res);
  const HNSWGraphStorage graph(*this, 0, HNSWGraphValidation::kOnAccess);
  const GraphSearcher searcher(graph);
  const auto search_from_seeds = [&](std::span<const GraphSearchSeed> seeds,
                                     int candidate_capacity) {
    GraphSearchOptions options;
    options.ef_search = static_cast<size_t>(candidate_capacity);
    options.check_relative_distance = check_relative_distance;
    options.selector = params == nullptr ? nullptr : params->sel;
    options.frontier_policy = GraphSearchFrontierPolicy::kNavigationBound;
    options.max_expansions =
        check_relative_distance ? 0 : static_cast<size_t>(ef_search) + 1;
    options.max_candidates = static_cast<size_t>(candidate_capacity);
    options.relative_distance_limit = static_cast<size_t>(ef_search);
    options.advance_visited = false;

    GraphSearchStats graph_stats;
    const std::vector<GraphSearchResult> results =
        searcher.Search(qdis, seeds, options, &vt, &graph_stats);
    for (const GraphSearchResult& result : results) {
      res.AddResult(result.distance, result.id);
    }
    search_stats.n1 += graph_stats.queries;
    search_stats.n2 += graph_stats.exhausted_queries;
    search_stats.ndis += graph_stats.distance_computations;
    search_stats.nhops += graph_stats.expanded_nodes;
  };

  if (search_type == 1) {
    for (int j = 0; j < nprobe; j++) {
      const storage_idx_t candidate = nearest_i[j];

      if (candidate < 0) {
        break;
      }
      if (vt.get(candidate)) {
        continue;
      }

      const int candidate_capacity = std::max(ef_search, k);
      const std::array<GraphSearchSeed, 1> seed = {
          GraphSearchSeed{candidate, nearest_d[j]}};
      search_from_seeds(seed, candidate_capacity);
    }
  } else if (search_type == 2) {
    const int candidate_capacity =
        std::max(std::max(ef_search, k), static_cast<int>(nprobe));
    std::vector<GraphSearchSeed> seeds;
    if (nprobe > 0) {
      seeds.reserve(static_cast<size_t>(nprobe));
    }
    for (int j = 0; j < nprobe; j++) {
      const storage_idx_t candidate = nearest_i[j];
      if (candidate < 0) {
        break;
      }
      seeds.push_back({candidate, nearest_d[j]});
    }
    search_from_seeds(seeds, candidate_capacity);
  }
}

void HNSW::PermuteEntries(const idx_t* map) {
  HYPERVEC_THROW_IF_NOT_MSG(
      levels.size() <=
          static_cast<size_t>((std::numeric_limits<storage_idx_t>::max)()),
      "HNSW::PermuteEntries: graph is too large for storage ids");
  const size_t n_total = levels.size();
  HYPERVEC_THROW_IF_NOT_MSG(offsets.size() == n_total + 1 && offsets[0] == 0,
                            "HNSW::PermuteEntries: invalid graph offsets");
  HYPERVEC_THROW_IF_NOT_MSG(
      offsets.back() == neighbors.size(),
      "HNSW::PermuteEntries: offsets do not cover the neighbor array");
  HYPERVEC_THROW_IF_NOT_MSG(
      entry_point == -1 ||
          (entry_point >= 0 && static_cast<size_t>(entry_point) < n_total),
      "HNSW::PermuteEntries: invalid entry point");
  if (n_total > 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        map != nullptr, "HNSW::PermuteEntries: permutation must not be null");
  }

  std::vector<storage_idx_t> imap(n_total);  // inverse mapping
  std::vector<uint8_t> seen(n_total, 0);
  // map: new index -> old index
  // imap: old index -> new index
  for (size_t i = 0; i < n_total; ++i) {
    HYPERVEC_THROW_IF_NOT_MSG(
        map[i] >= 0 && static_cast<size_t>(map[i]) < n_total,
        "HNSW::PermuteEntries: permutation entry is out of range");
    const size_t old_id = static_cast<size_t>(map[i]);
    HYPERVEC_THROW_IF_NOT_MSG(
        seen[old_id] == 0,
        "HNSW::PermuteEntries: permutation contains a duplicate id");
    seen[old_id] = 1;
    imap[old_id] = static_cast<storage_idx_t>(i);
  }

  for (size_t i = 0; i < n_total; ++i) {
    HYPERVEC_THROW_IF_NOT_MSG(
        offsets[i] <= offsets[i + 1] && offsets[i + 1] <= neighbors.size(),
        "HNSW::PermuteEntries: graph offsets are not monotonic");
    HYPERVEC_THROW_IF_NOT_MSG(levels[i] > 0,
                              "HNSW::PermuteEntries: graph level is invalid");
  }
  for (size_t i = 0; i < neighbors.size(); ++i) {
    const storage_idx_t neighbor = neighbors[i];
    HYPERVEC_THROW_IF_NOT_MSG(
        neighbor == -1 ||
            (neighbor >= 0 && static_cast<size_t>(neighbor) < n_total),
        "HNSW::PermuteEntries: graph contains an invalid neighbor");
  }

  const storage_idx_t new_entry_point =
      entry_point == -1 ? -1 : imap[static_cast<size_t>(entry_point)];
  std::vector<int> new_levels(n_total);
  std::vector<size_t> new_offsets(n_total + 1);
  MaybeOwnedVector<storage_idx_t> new_neighbors(neighbors.size());
  size_t no = 0;
  for (size_t i = 0; i < n_total; ++i) {
    const size_t o = static_cast<size_t>(map[i]);  // corresponding old index
    new_levels[i] = levels[o];
    for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
      const storage_idx_t neighbor = neighbors[j];
      new_neighbors[no++] =
          neighbor >= 0 ? imap[static_cast<size_t>(neighbor)] : neighbor;
    }
    new_offsets[i + 1] = no;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      no == neighbors.size(),
      "HNSW::PermuteEntries: permuted neighbor count is inconsistent");
  // swap everyone
  std::swap(levels, new_levels);
  std::swap(offsets, new_offsets);
  swap(neighbors, new_neighbors);
  entry_point = new_entry_point;
}

/**************************************************************
 * MinimaxHeap
 **************************************************************/

void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
  if (k == n) {
    if (v >= dis[0]) {
      return;
    }
    if (ids[0] != -1) {
      --nvalid;
    }
    hypervec::heap_pop<HC>(k--, dis.data(), ids.data());
  }
  hypervec::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
  ++nvalid;
}

float HNSW::MinimaxHeap::max() const {
  return dis[0];
}

int HNSW::MinimaxHeap::size() const {
  return nvalid;
}

void HNSW::MinimaxHeap::clear() {
  nvalid = k = 0;
}

#ifdef __AVX512F__

int HNSW::MinimaxHeap::PopMin(float* vmin_out) {
  assert(k > 0);
  static_assert(std::is_same<storage_idx_t, int32_t>::value,
                "This code expects storage_idx_t to be int32_t");

  int32_t min_idx = -1;
  float min_dis = std::numeric_limits<float>::infinity();

  __m512i min_indices = _mm512_set1_epi32(-1);
  __m512 min_distances = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  __m512i current_indices =
    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  __m512i offset = _mm512_set1_epi32(16);

  // The following loop tracks the rightmost index with the min distance.
  // -1 index values are ignored.
  const int k16 = (k / 16) * 16;
  for (size_t iii = 0; iii < k16; iii += 16) {
    __m512i indices = _mm512_loadu_si512((const __m512i*)(ids.data() + iii));
    __m512 distances = _mm512_loadu_ps(dis.data() + iii);

    // This mask filters out -1 values among indices.
    __mmask16 m1mask = _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

    __mmask16 dmask = _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
    __mmask16 finalmask = m1mask | dmask;

    const __m512i min_indices_new =
      _mm512_mask_blend_epi32(finalmask, current_indices, min_indices);
    const __m512 min_distances_new =
      _mm512_mask_blend_ps(finalmask, distances, min_distances);

    min_indices = min_indices_new;
    min_distances = min_distances_new;

    current_indices = _mm512_add_epi32(current_indices, offset);
  }

  // leftovers
  if (k16 != k) {
    const __mmask16 kmask = (1 << (k - k16)) - 1;

    __m512i indices =
      _mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), kmask, ids.data() + k16);
    __m512 distances = _mm512_maskz_loadu_ps(kmask, dis.data() + k16);

    // This mask filters out -1 values among indices.
    __mmask16 m1mask = _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

    __mmask16 dmask = _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
    __mmask16 finalmask = m1mask | dmask;

    const __m512i min_indices_new =
      _mm512_mask_blend_epi32(finalmask, current_indices, min_indices);
    const __m512 min_distances_new =
      _mm512_mask_blend_ps(finalmask, distances, min_distances);

    min_indices = min_indices_new;
    min_distances = min_distances_new;
  }

  // grab min distance
  min_dis = _mm512_reduce_min_ps(min_distances);
  // blend
  __mmask16 mindmask =
    _mm512_cmpeq_ps_mask(min_distances, _mm512_set1_ps(min_dis));
  // pick the max one
  min_idx = _mm512_mask_reduce_max_epi32(mindmask, min_indices);

  if (min_idx == -1) {
    return -1;
  }

  if (vmin_out) {
    *vmin_out = min_dis;
  }
  int ret = ids[min_idx];
  ids[min_idx] = -1;
  --nvalid;
  return ret;
}

#elif __AVX2__

int HNSW::MinimaxHeap::PopMin(float* vmin_out) {
  assert(k > 0);
  static_assert(std::is_same<storage_idx_t, int32_t>::value,
                "This code expects storage_idx_t to be int32_t");

  int32_t min_idx = -1;
  float min_dis = std::numeric_limits<float>::infinity();

  size_t iii = 0;

  __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
  __m256 min_distances = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  __m256i offset = _mm256_set1_epi32(8);

  // The baseline version is available in non-AVX2 branch.

  // The following loop tracks the rightmost index with the min distance.
  // -1 index values are ignored.
  const int k8 = (k / 8) * 8;
  for (; iii < k8; iii += 8) {
    __m256i indices = _mm256_loadu_si256((const __m256i*)(ids.data() + iii));
    __m256 distances = _mm256_loadu_ps(dis.data() + iii);

    // This mask filters out -1 values among indices.
    __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

    __m256i dmask =
      _mm256_castps_si256(_mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
    __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

    const __m256i min_indices_new = _mm256_castps_si256(
      _mm256_blendv_ps(_mm256_castsi256_ps(current_indices),
                       _mm256_castsi256_ps(min_indices), finalmask));

    const __m256 min_distances_new =
      _mm256_blendv_ps(distances, min_distances, finalmask);

    min_indices = min_indices_new;
    min_distances = min_distances_new;

    current_indices = _mm256_add_epi32(current_indices, offset);
  }

  // Vectorizing is doable, but is not practical
  int32_t vidx8[8];
  float vdis8[8];
  _mm256_storeu_ps(vdis8, min_distances);
  _mm256_storeu_si256((__m256i*)vidx8, min_indices);

  for (size_t j = 0; j < 8; j++) {
    if (min_dis > vdis8[j] || (min_dis == vdis8[j] && min_idx < vidx8[j])) {
      min_idx = vidx8[j];
      min_dis = vdis8[j];
    }
  }

  // process last values. Vectorizing is doable, but is not practical
  for (; iii < k; iii++) {
    if (ids[iii] != -1 && dis[iii] <= min_dis) {
      min_dis = dis[iii];
      min_idx = iii;
    }
  }

  if (min_idx == -1) {
    return -1;
  }

  if (vmin_out) {
    *vmin_out = min_dis;
  }
  int ret = ids[min_idx];
  ids[min_idx] = -1;
  --nvalid;
  return ret;
}

#else

// baseline non-vectorized version
int HNSW::MinimaxHeap::PopMin(float* vmin_out) {
  assert(k > 0);
  // returns min. This is an O(n) operation
  int i = k - 1;
  while (i >= 0) {
    if (ids[i] != -1) {
      break;
    }
    i--;
  }
  if (i == -1) {
    return -1;
  }
  int imin = i;
  float vmin = dis[i];
  i--;
  while (i >= 0) {
    if (ids[i] != -1 && dis[i] < vmin) {
      vmin = dis[i];
      imin = i;
    }
    i--;
  }
  if (vmin_out) {
    *vmin_out = vmin;
  }
  int ret = ids[imin];
  ids[imin] = -1;
  --nvalid;

  return ret;
}
#endif

int HNSW::MinimaxHeap::CountBelow(float thresh) {
  int n_below = 0;
  for (int i = 0; i < k; i++) {
    if (dis[i] < thresh) {
      n_below++;
    }
  }

  return n_below;
}

}  // namespace hypervec
