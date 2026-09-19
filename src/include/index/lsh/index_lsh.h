/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <quantization/quantizer.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hypervec {

struct SearchParametersLSH : SearchParameters {
  static constexpr size_t kUseConfiguredCandidateLimit =
      (std::numeric_limits<size_t>::max)();

  /** Zero preserves the value configured on the index. */
  size_t probe_count = 0;
  /** Maximum exact evaluations; zero is unlimited, max preserves config. */
  size_t candidate_limit = kUseConfiguredCandidateLimit;

  ~SearchParametersLSH() override = default;
};

struct LSHIndexOptions {
  size_t table_count = 8;
  size_t bits_per_table = 16;
  /** Exact bucket plus query-adaptive one-bit probes. */
  size_t probe_count = 4;
  /** Maximum exact distance evaluations per query; zero means unlimited. */
  size_t candidate_limit = 0;
  uint64_t random_seed = 0x9E3779B97F4A7C15ULL;
};

/** Random-hyperplane LSH with exact inner-product reranking.
 *
 * Hash tables generate an angular-similarity candidate set. Original float32
 * vectors are retained through the FlatQuantizer so returned scores preserve
 * HyperVec's inner-product metric contract. Query-adaptive probes flip the
 * bits with the smallest projection margins first.
 */
class IndexLSH final : public Index {
 public:
  IndexLSH(idx_t dimension, MetricType metric = kMetricInnerProduct,
           LSHIndexOptions options = {});
  ~IndexLSH() override = default;

  IndexCapabilities GetCapabilities() const override;
  void Train(idx_t n, const float* x) override;
  void Add(idx_t n, const float* x) override;
  void Search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;
  void Reset() override;
  void Reconstruct(idx_t key, float* recons) const override;
  DistanceComputer* GetDistanceComputer() const override;

  const LSHIndexOptions& Options() const noexcept { return options_; }
  const std::vector<float>& Hyperplanes() const noexcept {
    return hyperplanes_;
  }
  const InMemoryCodeStore& CodeStore() const noexcept { return code_store_; }

 private:
  using Bucket = std::vector<idx_t>;
  using HashTable = std::unordered_map<uint64_t, Bucket>;

  void ValidateVector(const float* vector, const char* operation) const;
  uint64_t Signature(const float* vector, size_t table,
                     std::vector<std::pair<float, size_t>>* margins) const;
  std::vector<uint64_t> ProbeSignatures(const float* vector, size_t table,
                                        size_t probe_count) const;

  LSHIndexOptions options_;
  FlatQuantizer quantizer_;
  InMemoryCodeStore code_store_;
  std::vector<float> hyperplanes_;
  std::vector<HashTable> tables_;
};

}  // namespace hypervec
