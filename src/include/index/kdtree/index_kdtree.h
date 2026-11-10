/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#ifndef INDEX_KDTREE_H
#define INDEX_KDTREE_H

#include <index/index.h>

#include <cstdint>
#include <vector>

namespace hypervec {

/** A node of the KD-tree, stored in a flat array so the whole tree
 * serializes as a single POD vector.
 *
 * Internal nodes split the point set on component `split_dim` at value
 * `split_val`: points with x[split_dim] <= split_val go to `left`, the rest
 * to `right`. Leaf nodes have `split_dim == -1` and cover the contiguous
 * range [begin, end) of the `point_ids` array.
 */
struct KDNode {
  int32_t split_dim;  ///< split axis, or -1 for a leaf
  float split_val;    ///< split threshold (internal nodes only)
  idx_t left;         ///< child node index, or -1 (internal nodes only)
  idx_t right;        ///< child node index, or -1 (internal nodes only)
  idx_t begin;        ///< first index into point_ids (leaf nodes only)
  idx_t end;          ///< one past last index into point_ids (leaf nodes only)
};

/** Exact k-NN index backed by a KD-tree.
 *
 * Stores the full vectors (like IndexFlat) plus a KD-tree over them, and
 * answers k-NN queries exactly via backtracking with hypersphere/hyperplane
 * pruning. Only the L2 metric is supported.
 *
 * KD-trees prune effectively in low dimension; as d grows the backtracking
 * visits most of the tree and the search degrades toward an exhaustive scan.
 * Results remain exact at any dimension.
 */
struct IndexKDTree : Index {
  /// max number of points stored in a leaf node
  int leaf_size;

  /// full dataset, row-major, size n_total * d
  std::vector<float> xb;

  /// tree nodes; nodes[0] is the root once the tree is built
  std::vector<KDNode> nodes;

  /// permutation of [0, n_total) grouping points by leaf
  std::vector<idx_t> point_ids;

  /// whether `nodes`/`point_ids` are consistent with `xb`
  bool tree_built;

  explicit IndexKDTree(idx_t d,  ///< dimensionality of the input vectors
                       MetricType metric = kMetricL2, int leaf_size = 16);

  /// default ctor for deserialization
  IndexKDTree();

  /// Append n vectors; invalidates the tree. Call BuildTree() before Search.
  void Add(idx_t n, const float* x) override;

  /// (Re)build the KD-tree over the currently stored vectors.
  void BuildTree();

  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void Reset() override;

  void Reconstruct(idx_t key, float* recons) const override;

  uint32_t fourcc() const override;
  void write_body(IOWriter* f) const override;
};

}  // namespace hypervec

#endif
