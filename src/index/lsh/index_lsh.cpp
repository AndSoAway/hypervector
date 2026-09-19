/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/graph/visited_table.h>
#include <index/lsh/index_lsh.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>
#include <utils/selector/id_selector.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

uint64_t SplitMix64(uint64_t* state) noexcept {
  *state += 0x9E3779B97F4A7C15ULL;
  uint64_t value = *state;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

float SymmetricRandomWeight(uint64_t* state) noexcept {
  constexpr double kScale = 1.0 / static_cast<double>(uint64_t{1} << 53U);
  const double unit = static_cast<double>(SplitMix64(state) >> 11U) * kScale;
  return static_cast<float>(2.0 * unit - 1.0);
}

size_t ProjectionCount(idx_t dimension, const LSHIndexOptions& options) {
  const size_t table_bits = mul_no_overflow(
      options.table_count, options.bits_per_table, "IndexLSH hyperplane count");
  return mul_no_overflow(table_bits, static_cast<size_t>(dimension),
                         "IndexLSH hyperplane elements");
}

void ValidateOptions(const LSHIndexOptions& options) {
  HYPERVEC_THROW_IF_NOT_MSG(options.table_count > 0,
                            "IndexLSH: table_count must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      options.bits_per_table > 0 && options.bits_per_table <= 63,
      "IndexLSH: bits_per_table must be in [1, 63]");
  HYPERVEC_THROW_IF_NOT_MSG(
      options.probe_count > 0 &&
          options.probe_count <= options.bits_per_table + 1,
      "IndexLSH: probe_count must be in [1, bits_per_table + 1]");
}

struct SearchResult {
  float score;
  idx_t id;
};

bool BetterResult(const SearchResult& lhs, const SearchResult& rhs) noexcept {
  if (lhs.score != rhs.score) {
    return lhs.score > rhs.score;
  }
  return lhs.id < rhs.id;
}

}  // namespace

IndexLSH::IndexLSH(idx_t dimension, MetricType metric, LSHIndexOptions options)
    : Index(dimension, metric),
      options_(options),
      quantizer_(dimension, metric),
      code_store_(quantizer_.CodeSize()) {
  HYPERVEC_THROW_IF_NOT_FMT(
      dimension > 0 && dimension <= std::numeric_limits<int>::max(),
      "IndexLSH: dimension must be in [1, %d]",
      std::numeric_limits<int>::max());
  HYPERVEC_THROW_IF_NOT_MSG(
      metric == kMetricInnerProduct,
      "IndexLSH: random-hyperplane LSH supports kMetricInnerProduct only");
  ValidateOptions(options_);
  hyperplanes_.resize(ProjectionCount(dimension, options_));
  uint64_t random_state = options_.random_seed;
  for (float& weight : hyperplanes_) {
    weight = SymmetricRandomWeight(&random_state);
  }
  tables_.resize(options_.table_count);
  is_trained = true;
}

IndexCapabilities IndexLSH::GetCapabilities() const {
  IndexCapabilities capabilities;
  capabilities.supports_reconstruct = true;
  return capabilities;
}

void IndexLSH::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexLSH::Train: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexLSH::Train: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n_total == 0, "IndexLSH::Train: a populated index cannot be retrained");
  is_trained = true;
}

void IndexLSH::ValidateVector(const float* vector,
                              const char* operation) const {
  HYPERVEC_THROW_IF_NOT_FMT(vector != nullptr, "%s: vector must not be null",
                            operation);
  for (int component = 0; component < d; ++component) {
    HYPERVEC_THROW_IF_NOT_FMT(std::isfinite(vector[component]),
                              "%s: vector components must be finite",
                              operation);
  }
}

uint64_t IndexLSH::Signature(
    const float* vector, size_t table,
    std::vector<std::pair<float, size_t>>* margins) const {
  uint64_t signature = 0;
  if (margins != nullptr) {
    margins->clear();
    margins->reserve(options_.bits_per_table);
  }
  const size_t table_offset = mul_no_overflow(
      mul_no_overflow(table, options_.bits_per_table,
                      "IndexLSH projection table offset"),
      static_cast<size_t>(d), "IndexLSH projection table offset");
  for (size_t bit = 0; bit < options_.bits_per_table; ++bit) {
    const float* hyperplane =
        hyperplanes_.data() + table_offset + bit * static_cast<size_t>(d);
    double projection = 0.0;
    for (int component = 0; component < d; ++component) {
      projection += static_cast<double>(vector[component]) *
                    static_cast<double>(hyperplane[component]);
    }
    HYPERVEC_THROW_IF_NOT_MSG(std::isfinite(projection),
                              "IndexLSH: projection is not finite");
    if (projection >= 0.0) {
      signature |= uint64_t{1} << bit;
    }
    if (margins != nullptr) {
      margins->emplace_back(static_cast<float>(std::abs(projection)), bit);
    }
  }
  return signature;
}

std::vector<uint64_t> IndexLSH::ProbeSignatures(const float* vector,
                                                size_t table,
                                                size_t probe_count) const {
  std::vector<std::pair<float, size_t>> margins;
  const uint64_t exact = Signature(vector, table, &margins);
  std::sort(margins.begin(), margins.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.first < rhs.first ||
                     (lhs.first == rhs.first && lhs.second < rhs.second);
            });
  std::vector<uint64_t> probes;
  probes.reserve(probe_count);
  probes.push_back(exact);
  for (size_t probe = 1; probe < probe_count; ++probe) {
    probes.push_back(exact ^ (uint64_t{1} << margins[probe - 1].second));
  }
  return probes;
}

void IndexLSH::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexLSH::Add: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexLSH::Add: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(is_trained, "IndexLSH::Add: index is not trained");
  HYPERVEC_THROW_IF_NOT_MSG(
      n <= (std::numeric_limits<idx_t>::max)() - n_total,
      "IndexLSH::Add: vector count exceeds idx_t capacity");
  if (n == 0) {
    return;
  }

  const size_t signature_count =
      mul_no_overflow(static_cast<size_t>(n), options_.table_count,
                      "IndexLSH staged signatures");
  std::vector<uint64_t> signatures(signature_count);
  for (idx_t vector_index = 0; vector_index < n; ++vector_index) {
    const float* vector = x + static_cast<size_t>(vector_index) * d;
    ValidateVector(vector, "IndexLSH::Add");
    for (size_t table = 0; table < options_.table_count; ++table) {
      signatures[static_cast<size_t>(vector_index) * options_.table_count +
                 table] = Signature(vector, table, nullptr);
    }
  }

  const size_t encoded_size =
      mul_no_overflow(static_cast<size_t>(n), quantizer_.CodeSize(),
                      "IndexLSH staged encoded vectors");
  std::vector<uint8_t> encoded(encoded_size);
  quantizer_.Encode(n, x, encoded.data());
  InMemoryCodeStore staged_store = code_store_;
  staged_store.Append(n, encoded.data());
  std::vector<HashTable> staged_tables = tables_;
  for (idx_t vector_index = 0; vector_index < n; ++vector_index) {
    const idx_t id = n_total + vector_index;
    for (size_t table = 0; table < options_.table_count; ++table) {
      const uint64_t signature =
          signatures[static_cast<size_t>(vector_index) * options_.table_count +
                     table];
      staged_tables[table][signature].push_back(id);
    }
  }

  code_store_ = std::move(staged_store);
  tables_ = std::move(staged_tables);
  n_total += n;
}

void IndexLSH::Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels, const SearchParameters* params) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      n >= 0, "IndexLSH::Search: n must be non-negative, got %" PRId64,
      static_cast<int64_t>(n));
  HYPERVEC_THROW_IF_NOT_FMT(
      k > 0, "IndexLSH::Search: k must be positive, got %" PRId64,
      static_cast<int64_t>(k));
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || x != nullptr,
      "IndexLSH::Search: x must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || distances != nullptr,
      "IndexLSH::Search: distances must not be null when n is positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      n == 0 || labels != nullptr,
      "IndexLSH::Search: labels must not be null when n is positive");
  if (n == 0) {
    return;
  }

  size_t probe_count = options_.probe_count;
  size_t candidate_limit = options_.candidate_limit;
  if (const auto* lsh_params =
          dynamic_cast<const SearchParametersLSH*>(params)) {
    if (lsh_params->probe_count != 0) {
      probe_count = lsh_params->probe_count;
    }
    if (lsh_params->candidate_limit !=
        SearchParametersLSH::kUseConfiguredCandidateLimit) {
      candidate_limit = lsh_params->candidate_limit;
    }
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      probe_count > 0 && probe_count <= options_.bits_per_table + 1,
      "IndexLSH::Search: probe_count must be in [1, bits_per_table + 1]");

  const size_t output_size =
      mul_no_overflow(static_cast<size_t>(n), static_cast<size_t>(k),
                      "IndexLSH::Search output size");
  std::fill_n(distances, output_size,
              -(std::numeric_limits<float>::infinity)());
  std::fill_n(labels, output_size, static_cast<idx_t>(-1));
  if (n_total == 0) {
    return;
  }

  std::unique_ptr<DistanceComputer> distance =
      quantizer_.CreateDistanceComputer(code_store_.View());
  VisitedTable visited(static_cast<size_t>(n_total), false);
  const IDSelector* selector = params == nullptr ? nullptr : params->sel;
  for (idx_t query_index = 0; query_index < n; ++query_index) {
    const float* query = x + static_cast<size_t>(query_index) * d;
    ValidateVector(query, "IndexLSH::Search");
    distance->SetQuery(query);
    std::vector<idx_t> candidates;
    if (candidate_limit != 0) {
      candidates.reserve(
          std::min(candidate_limit, static_cast<size_t>(n_total)));
    }
    bool limit_reached = false;
    for (size_t table = 0; table < options_.table_count && !limit_reached;
         ++table) {
      for (uint64_t signature : ProbeSignatures(query, table, probe_count)) {
        const auto bucket = tables_[table].find(signature);
        if (bucket == tables_[table].end()) {
          continue;
        }
        for (idx_t candidate : bucket->second) {
          if (!visited.set(static_cast<size_t>(candidate))) {
            continue;
          }
          if (selector != nullptr && !selector->IsMember(candidate)) {
            continue;
          }
          candidates.push_back(candidate);
          if (candidate_limit != 0 && candidates.size() >= candidate_limit) {
            limit_reached = true;
            break;
          }
        }
        if (limit_reached) {
          break;
        }
      }
    }

    std::vector<SearchResult> results;
    results.reserve(candidates.size());
    for (idx_t candidate : candidates) {
      results.push_back({(*distance)(candidate), candidate});
    }
    std::sort(results.begin(), results.end(), BetterResult);
    const size_t result_count =
        std::min(results.size(), static_cast<size_t>(k));
    const size_t output_offset =
        static_cast<size_t>(query_index) * static_cast<size_t>(k);
    for (size_t result = 0; result < result_count; ++result) {
      distances[output_offset + result] = results[result].score;
      labels[output_offset + result] = results[result].id;
    }
    visited.advance();
  }
}

void IndexLSH::Reset() {
  code_store_.Reset();
  tables_.clear();
  tables_.resize(options_.table_count);
  n_total = 0;
}

void IndexLSH::Reconstruct(idx_t key, float* recons) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      key >= 0 && key < n_total,
      "IndexLSH::Reconstruct: key is outside [0, n_total)");
  HYPERVEC_THROW_IF_NOT_MSG(recons != nullptr,
                            "IndexLSH::Reconstruct: output must not be null");
  quantizer_.Decode(1, code_store_.Code(key), recons);
}

DistanceComputer* IndexLSH::GetDistanceComputer() const {
  return quantizer_.CreateDistanceComputer(code_store_.View()).release();
}

}  // namespace hypervec
