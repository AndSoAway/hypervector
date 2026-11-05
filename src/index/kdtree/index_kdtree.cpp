/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#include <index/kdtree/index_kdtree.h>

#include <omp.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>

namespace hypervec {

IndexKDTree::IndexKDTree(idx_t d, MetricType metric, int leaf_size)
  : Index(d, metric), leaf_size(leaf_size), tree_built(false) {
  HYPERVEC_THROW_IF_NOT_MSG(metric == kMetricL2,
                            "IndexKDTree only supports the L2 metric");
  HYPERVEC_THROW_IF_NOT_MSG(leaf_size > 0, "leaf_size must be positive");
}

IndexKDTree::IndexKDTree() : Index(), leaf_size(16), tree_built(false) {}

void IndexKDTree::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }
  xb.insert(xb.end(), x, x + static_cast<size_t>(n) * d);
  n_total += n;
  tree_built = false;
}

void IndexKDTree::Reset() {
  xb.clear();
  nodes.clear();
  point_ids.clear();
  n_total = 0;
  tree_built = false;
}

void IndexKDTree::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT(key >= 0 && key < n_total);
  memcpy(recons, xb.data() + key * static_cast<size_t>(d),
         sizeof(float) * d);
}

namespace {

/** Recursively build the subtree covering point_ids[begin, end). Returns the
 * index of the created node in `nodes`. */
idx_t BuildSubtree(std::vector<KDNode>& nodes, std::vector<idx_t>& point_ids,
                   const float* xb, int d, int leaf_size, idx_t begin,
                   idx_t end) {
  const idx_t node_id = static_cast<idx_t>(nodes.size());
  nodes.emplace_back();

  if (end - begin <= leaf_size) {
    KDNode& node = nodes[node_id];
    node.split_dim = -1;
    node.split_val = 0.0f;
    node.left = -1;
    node.right = -1;
    node.begin = begin;
    node.end = end;
    return node_id;
  }

  // Pick the split axis as the dimension of greatest spread in this range.
  int axis = 0;
  float best_spread = -1.0f;
  for (int dim = 0; dim < d; dim++) {
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (idx_t i = begin; i < end; i++) {
      const float v = xb[point_ids[i] * static_cast<size_t>(d) + dim];
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    const float spread = hi - lo;
    if (spread > best_spread) {
      best_spread = spread;
      axis = dim;
    }
  }

  // Degenerate range (all points identical) -> make it a leaf.
  if (best_spread == 0.0f) {
    KDNode& node = nodes[node_id];
    node.split_dim = -1;
    node.split_val = 0.0f;
    node.left = -1;
    node.right = -1;
    node.begin = begin;
    node.end = end;
    return node_id;
  }

  // Partition point_ids[begin, end) around the median on `axis`.
  const idx_t mid = begin + (end - begin) / 2;
  std::nth_element(
    point_ids.begin() + begin, point_ids.begin() + mid,
    point_ids.begin() + end, [&](idx_t a, idx_t b) {
      return xb[a * static_cast<size_t>(d) + axis] <
             xb[b * static_cast<size_t>(d) + axis];
    });
  const float split_val = xb[point_ids[mid] * static_cast<size_t>(d) + axis];

  const idx_t left = BuildSubtree(nodes, point_ids, xb, d, leaf_size, begin, mid);
  const idx_t right = BuildSubtree(nodes, point_ids, xb, d, leaf_size, mid, end);

  KDNode& node = nodes[node_id];
  node.split_dim = axis;
  node.split_val = split_val;
  node.left = left;
  node.right = right;
  node.begin = 0;
  node.end = 0;
  return node_id;
}

}  // namespace

void IndexKDTree::BuildTree() {
  nodes.clear();
  point_ids.resize(n_total);
  for (idx_t i = 0; i < n_total; i++) {
    point_ids[i] = i;
  }
  if (n_total > 0) {
    nodes.reserve(2 * (static_cast<size_t>(n_total) / leaf_size + 1));
    BuildSubtree(nodes, point_ids, xb.data(), d, leaf_size, 0, n_total);
  }
  tree_built = true;
}

namespace {

/** Backtracking k-NN search for a single query. `heap_val`/`heap_ids` hold a
 * max-heap of the k best (squared-distance, id) pairs seen so far. */
void SearchOne(const KDNode* nodes, const idx_t* point_ids, const float* xb,
               int d, idx_t k, const float* q, float* heap_val,
               idx_t* heap_ids) {
  // Explicit stack of node indices to visit, avoiding deep recursion.
  // Each entry is paired with the min squared distance from the query to the
  // node's region; skip the entry if that already exceeds the heap top.
  struct StackEntry {
    idx_t node;
    float min_dist;
  };
  std::vector<StackEntry> stack;
  stack.push_back({0, 0.0f});

  while (!stack.empty()) {
    const StackEntry entry = stack.back();
    stack.pop_back();

    // Prune: the closest point in this region can't beat the current k-th best.
    if (entry.min_dist > heap_val[0]) {
      continue;
    }

    const KDNode& node = nodes[entry.node];
    if (node.split_dim < 0) {
      for (idx_t i = node.begin; i < node.end; i++) {
        const idx_t id = point_ids[i];
        const float dist =
          fvec_L2sqr(q, xb + id * static_cast<size_t>(d), d);
        if (dist < heap_val[0]) {
          maxheap_replace_top(static_cast<size_t>(k), heap_val, heap_ids, dist,
                              id);
        }
      }
      continue;
    }

    // Descend into the near child first; defer the far child with the squared
    // hyperplane distance as its region lower bound so it is pruned if the
    // heap is already closer than that plane.
    const float diff = q[node.split_dim] - node.split_val;
    const idx_t near = diff <= 0.0f ? node.left : node.right;
    const idx_t far = diff <= 0.0f ? node.right : node.left;
    const float far_min = entry.min_dist + diff * diff;

    // Push far first so near is processed next (LIFO).
    stack.push_back({far, far_min});
    stack.push_back({near, entry.min_dist});
  }
}

}  // namespace

void IndexKDTree::Search(idx_t n, const float* x, idx_t k, float* distances,
                         idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT(k > 0);
  HYPERVEC_THROW_IF_NOT_MSG(params == nullptr || params->sel == nullptr,
                            "IndexKDTree does not support an IDSelector");
  HYPERVEC_THROW_IF_NOT_MSG(
    tree_built, "IndexKDTree: call BuildTree() before Search");
  HYPERVEC_THROW_IF_NOT_MSG(metric_type == kMetricL2,
                            "IndexKDTree only supports the L2 metric");

#pragma omp parallel for if (n > 1)
  for (idx_t i = 0; i < n; i++) {
    const float* q = x + i * static_cast<size_t>(d);
    float* heap_val = distances + i * k;
    idx_t* heap_ids = labels + i * k;

    maxheap_heapify(static_cast<size_t>(k), heap_val, heap_ids);
    if (n_total > 0) {
      SearchOne(nodes.data(), point_ids.data(), xb.data(), d, k, q, heap_val,
                heap_ids);
    }
    maxheap_reorder(static_cast<size_t>(k), heap_val, heap_ids);
  }
}

uint32_t IndexKDTree::fourcc() const {
  return hypervec::fourcc("IKDt");
}

void IndexKDTree::write_body(IOWriter* f) const {
  HYPERVEC_THROW_IF_NOT_MSG(
    tree_built,
    "IndexKDTree: build the tree (BuildTree/Search) before serializing");
  int leaf_size_i = leaf_size;
  WRITE1(leaf_size_i);
  WRITEVECTOR(xb);
  WRITEVECTOR(nodes);
  WRITEVECTOR(point_ids);
}

}  // namespace hypervec
