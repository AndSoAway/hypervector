/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/diskann/diskann_storage.h>
#include <index/graph/graph_searcher.h>
#include <quantization/quantizer.h>

#include <cstddef>
#include <vector>

namespace hypervec {

struct IDSelector;

struct DiskAnnSearchOptions {
  size_t search_width = 64;
  bool check_relative_distance = true;
  const IDSelector* selector = nullptr;
};

struct DiskAnnSearchStats {
  GraphSearchStats graph;
  size_t exact_distance_computations = 0;

  void Reset() noexcept;
  void Combine(const DiskAnnSearchStats& other) noexcept;
};

/** Two-stage DiskANN search over memory-resident codes and paged raw vectors.
 *
 * Approximate distances over encoded_vectors guide graph traversal. The raw
 * vectors of the retained candidates are then loaded and reranked with exact
 * squared L2 distances. All constructor arguments are non-owning and must
 * outlive this object and every concurrent Search call.
 */
class DiskAnnSearcher {
 public:
  DiskAnnSearcher(const GraphStorage& graph,
                  const PagedVectorStorage& raw_vectors,
                  const Quantizer& quantizer, EncodedVectorView encoded_vectors,
                  GraphId entry_point);

  /** Returns up to result_count results ordered by exact distance, then ID. */
  std::vector<GraphSearchResult> Search(
      const float* query, size_t result_count,
      const DiskAnnSearchOptions& options = {},
      DiskAnnSearchStats* stats = nullptr) const;

 private:
  const GraphStorage& graph_;
  const PagedVectorStorage& raw_vectors_;
  const Quantizer& quantizer_;
  EncodedVectorView encoded_vectors_;
  GraphId entry_point_;
};

}  // namespace hypervec
