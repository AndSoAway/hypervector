/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <core/distance_computer.h>
#include <core/hypervec_assert.h>
#include <core/index.h>
#include <core/maybe_owned_vector.h>
#include <core/platform_macros.h>
#include <omp.h>
#include <utils/structures/heap.h>
#include <utils/structures/random.h>

#include <optional>
#include <queue>
#include <vector>

namespace hypervec {

// Forward declarations to avoid circular dependency.
struct IndexHNSW;
struct IndexHNSWFlat;
struct IndexHNSWPQ;
struct IndexHNSWSQ;
struct IndexHNSW2Level;
struct IndexHNSWCagra;

/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor Search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implementation is heavily influenced by the NMSlib
 * implementation by Yury Malkov and Leonid Boytsov
 * (https://github.com/searchivarius/nmslib)
 *
 * The HNSW object stores only the neighbor link structure, see
 * IndexHNSW.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer;  // from AuxIndexStructures
struct HNSWStats;

struct SearchParametersHNSW : SearchParameters {
  int ef_search = 16;
  bool check_relative_distance = true;
  bool bounded_queue = true;

  ~SearchParametersHNSW() {}
};

struct HNSW {
  /// internal storage of vectors (32 bits: this is expensive)
  using storage_idx_t = int32_t;

  // for now we do only these distances
  using C = CMax<float, int64_t>;

  typedef std::pair<float, storage_idx_t> Node;

  /** Heap structure that allows fast access and updates.
   */
  struct MinimaxHeap {
    int n;
    int k;
    int nvalid;

    std::vector<storage_idx_t> ids;
    std::vector<float> dis;
    typedef hypervec::CMax<float, storage_idx_t> HC;

    explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

    void push(storage_idx_t i, float v);

    float max() const;

    int size() const;

    void clear();

    int PopMin(float* vmin_out = nullptr);

    int CountBelow(float thresh);
  };

  /// to sort pairs of (id, distance) from nearest to farthest or the reverse
  struct NodeDistCloser {
    float d;
    int id;
    NodeDistCloser(float d, int id) : d(d), id(id) {}
    bool operator<(const NodeDistCloser& obj1) const {
      return d < obj1.d;
    }
  };

  struct NodeDistFarther {
    float d;
    int id;
    NodeDistFarther(float d, int id) : d(d), id(id) {}
    bool operator<(const NodeDistFarther& obj1) const {
      return d > obj1.d;
    }
  };

  /// assignment probability to each layer (sum=1)
  std::vector<double> assign_probas;

  /// number of neighbors stored per layer (cumulative), should not
  /// be changed after first Add
  std::vector<int> cum_nneighbor_per_level;

  /// level of each vector (base level = 1), size = n_total
  std::vector<int> levels;

  /// offsets[i] is the offset in the neighbors array where vector i is stored
  /// size n_total + 1
  std::vector<size_t> offsets;

  /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
  /// for all levels. this is where all storage goes.
  MaybeOwnedVector<storage_idx_t> neighbors;

  /// entry point in the Search structure (one of the points with maximum
  /// level
  storage_idx_t entry_point = -1;

  hypervec::RandomGenerator rng;

  /// maximum level
  int max_level = -1;

  /// expansion factor at construction time
  int ef_construction = 40;

  /// expansion factor at Search time
  int ef_search = 16;

  /// during Search: do we check whether the next best distance is good
  /// enough?
  bool check_relative_distance = true;

  /// use bounded queue during exploration
  bool search_bounded_queue = true;

  bool is_panorama = false;

  // See impl/VisitedTable.h.
  std::optional<bool> use_visited_hashset;

  // methods that initialize the tree sizes

  /// initialize the assign_probas and cum_nneighbor_per_level to
  /// have 2*M links on level 0 and M links on levels > 0
  void SetDefaultProbas(int M, float levelMult);

  /// set nb of neighbors for this level (before adding anything)
  void SetNbNeighbors(int level_no, int n);

  // methods that access the tree sizes

  /// nb of neighbors for this level
  int NbNeighbors(int layer_no) const;

  /// cumulative nb up to (and excluding) this level
  int CumNbNeighbors(int layer_no) const;

  /// range of entries in the neighbors table of vertex no at layer_no
  void NeighborRange(idx_t no, int layer_no, size_t* begin, size_t* end) const;

  /// only mandatory parameter: nb of neighbors
  explicit HNSW(int M = 32);

  /// pick a random level for a new point
  int RandomLevel();

  /// Add n random levels to table (for debugging...)
  void FillWithRandomLinks(size_t n);

  void AddLinksStartingFrom(DistanceComputer& ptdis, storage_idx_t pt_id,
                               storage_idx_t nearest, float d_nearest,
                               int level, omp_lock_t* locks, VisitedTable& vt,
                               bool keep_max_size_level0 = false);

  /** Add point pt_id on all levels <= pt_level and build the link
   * structure for them. */
  void AddWithLocks(DistanceComputer& ptdis, int pt_level, int pt_id,
                      std::vector<omp_lock_t>& locks, VisitedTable& vt,
                      bool keep_max_size_level0 = false);

  /// Search interface for 1 point, single thread
  ///
  /// NOTE: We pass a reference to the index itself to allow for additional
  /// The alternative would be to override both HNSW::Search and
  /// HNSWIndex::Search, which would be a nuisance of code duplication.
  HNSWStats Search(DistanceComputer& qdis, const IndexHNSW* index,
                   ResultHandler& res, VisitedTable& vt,
                   const SearchParameters* params = nullptr) const;

  /// Search only in level 0 from a given vertex
  void SearchLevel0(DistanceComputer& qdis, ResultHandler& res, idx_t nprobe,
                      const storage_idx_t* nearest_i, const float* nearest_d,
                      int search_type, HNSWStats& search_stats,
                      VisitedTable& vt,
                      const SearchParameters* params = nullptr) const;

  void Reset();

  void ClearNeighborTables(int level);
  void PrintNeighborStats(int level) const;

  int PrepareLevelTab(size_t n, bool preset_levels = false);

  static void ShrinkNeighborList(DistanceComputer& qdis,
                                   std::priority_queue<NodeDistFarther>& input,
                                   std::vector<NodeDistFarther>& output,
                                   int max_size,
                                   bool keep_max_size_level0 = false);

  void PermuteEntries(const idx_t* map);
};

struct HNSWStats {
  size_t n1 = 0;  /// number of vectors searched
  size_t n2 =
    0;  /// number of queries for which the candidate list is exhausted
  size_t ndis = 0;   /// number of distances computed
  size_t nhops = 0;  /// number of hops aka number of edges traversed

  void Reset() {
    n1 = n2 = 0;
    ndis = 0;
    nhops = 0;
  }

  void combine(const HNSWStats& other) {
    n1 += other.n1;
    n2 += other.n2;
    ndis += other.ndis;
    nhops += other.nhops;
  }
};

// global var that collects them all
HYPERVEC_API extern HNSWStats hnsw_stats;

int SearchFromCandidates(const HNSW& hnsw, DistanceComputer& qdis,
                           ResultHandler& res, HNSW::MinimaxHeap& candidates,
                           VisitedTable& vt, HNSWStats& stats, int level,
                           int nres_in = 0,
                           const SearchParameters* params = nullptr);

HYPERVEC_API HNSWStats GreedyUpdateNearest(const HNSW& hnsw,
                                        DistanceComputer& qdis, int level,
                                        HNSW::storage_idx_t& nearest,
                                        float& d_nearest);

        std::priority_queue<HNSW::Node> SearchFromCandidateUnbounded(
          const HNSW& hnsw, const HNSW::Node& node, DistanceComputer& qdis,
          int ef, VisitedTable* vt, HNSWStats& stats);

        void SearchNeighborsToAdd(
          HNSW& hnsw, DistanceComputer& qdis,
          std::priority_queue<HNSW::NodeDistCloser>& results, int entry_point,
          float d_entry_point, int level, VisitedTable& vt,
          bool reference_version = false);

        }  // namespace hypervec
