/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/graph/graph_validation.h>
#include <index/nsw/index_nsw.h>
#include <quantization/quantizer.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/assert.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace {

constexpr hypervec::NSWIndexOptions kExhaustiveOptions = {8, 8, 8, true, true};

void ExpectSameSearch(const hypervec::Index& expected,
                      const hypervec::Index& actual,
                      const std::vector<float>& queries, hypervec::idx_t k) {
  const hypervec::idx_t count =
      static_cast<hypervec::idx_t>(queries.size()) / expected.d;
  std::vector<float> expected_distances(static_cast<size_t>(count * k));
  std::vector<float> actual_distances(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> expected_labels(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> actual_labels(static_cast<size_t>(count * k));

  expected.Search(count, queries.data(), k, expected_distances.data(),
                  expected_labels.data());
  actual.Search(count, queries.data(), k, actual_distances.data(),
                actual_labels.data());

  EXPECT_EQ(actual_labels, expected_labels);
  ASSERT_EQ(actual_distances.size(), expected_distances.size());
  for (size_t i = 0; i < actual_distances.size(); ++i) {
    EXPECT_FLOAT_EQ(actual_distances[i], expected_distances[i]);
  }
}

class ThrowingDistanceComputer final : public hypervec::DistanceComputer {
 public:
  explicit ThrowingDistanceComputer(hypervec::EncodedVectorView store)
      : store_(store) {}

  void SetQuery(const float* query) override { query_ = *query; }

  float operator()(hypervec::idx_t index) override {
    const float difference = Value(index) - query_;
    return difference * difference;
  }

  float symmetric_dis(hypervec::idx_t lhs, hypervec::idx_t rhs) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        lhs != 2 && rhs != 2,
        "ThrowingQuantizer: requested failure during reciprocal update");
    const float difference = Value(lhs) - Value(rhs);
    return difference * difference;
  }

 private:
  float Value(hypervec::idx_t index) const {
    float value = 0.0F;
    std::memcpy(&value, store_.Code(index), sizeof(value));
    return value;
  }

  hypervec::EncodedVectorView store_;
  float query_ = 0.0F;
};

class ThrowingQuantizer final : public hypervec::Quantizer {
 public:
  std::string_view TypeName() const noexcept override { return "throwing"; }
  hypervec::idx_t Dimension() const noexcept override { return 1; }
  hypervec::MetricType Metric() const noexcept override {
    return hypervec::kMetricL2;
  }
  bool NeedsTraining() const noexcept override { return false; }
  bool IsTrained() const noexcept override { return true; }
  size_t CodeSize() const noexcept override { return sizeof(float); }

 protected:
  void TrainImpl(hypervec::idx_t, const float*) override {}

  void EncodeImpl(hypervec::idx_t count, const float* vectors,
                  uint8_t* codes) const override {
    std::memcpy(codes, vectors, static_cast<size_t>(count) * sizeof(float));
  }

  void DecodeImpl(hypervec::idx_t count, const uint8_t* codes,
                  float* vectors) const override {
    std::memcpy(vectors, codes, static_cast<size_t>(count) * sizeof(float));
  }

  std::unique_ptr<hypervec::DistanceComputer> CreateDistanceComputerImpl(
      hypervec::EncodedVectorView store) const override {
    return std::make_unique<ThrowingDistanceComputer>(store);
  }
};

}  // namespace

TEST(IndexNSW, FlatL2MatchesExhaustiveSearchAcrossRepeatedAdds) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  const std::vector<float> queries = {4.0F, 16.0F};
  hypervec::IndexFlatL2 expected(1);
  expected.Add(static_cast<hypervec::idx_t>(database.size()), database.data());

  hypervec::IndexNSWFlat index(1, hypervec::kMetricL2, kExhaustiveOptions);
  index.Add(3, database.data());
  index.Add(3, database.data() + 3);

  EXPECT_EQ(index.n_total, 6);
  EXPECT_EQ(index.CodeStore().Size(), 6);
  EXPECT_EQ(index.Graph().NodeCount(), 6U);
  EXPECT_EQ(index.EntryPoint(), 0);
  EXPECT_EQ(index.BuildStats().inserted_nodes, 6U);
  const auto report =
      hypervec::ValidateGraph(index.Graph(), index.EntryPoint());
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, 6U);
  ExpectSameSearch(expected, index, queries, 3);
}

TEST(IndexNSW, SimilaritySearchRestoresExternalDistanceDirection) {
  const std::vector<float> database = {
      1.0F, 0.0F, 0.0F, 2.0F, 2.0F, 1.0F, 3.0F, 4.0F, 1.0F, 3.0F, 6.0F, 2.0F,
  };
  const std::vector<float> queries = {1.0F, 1.0F, 2.0F, 0.5F};
  hypervec::IndexFlatIP expected(2);
  expected.Add(6, database.data());

  hypervec::IndexNSWFlat index(2, hypervec::kMetricInnerProduct,
                               kExhaustiveOptions);
  index.Build(6, database.data());

  ExpectSameSearch(expected, index, queries, 3);
  std::unique_ptr<hypervec::DistanceComputer> distance(
      index.GetDistanceComputer());
  distance->SetQuery(queries.data());
  EXPECT_FLOAT_EQ((*distance)(5), 8.0F);
}

TEST(IndexNSW, SelectorFiltersResultsWithoutBlockingTraversal) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  hypervec::IndexNSWFlat index(1, hypervec::kMetricL2, kExhaustiveOptions);
  index.Add(6, database.data());

  hypervec::IDSelectorRange selector(2, 5);
  hypervec::SearchParametersNSW params;
  params.ef_search = 8;
  params.sel = &selector;
  std::array<float, 4> distances;
  std::array<hypervec::idx_t, 4> labels;
  const float query = 1.0F;
  index.Search(1, &query, 4, distances.data(), labels.data(), &params);

  EXPECT_EQ(labels[0], 2);
  EXPECT_EQ(labels[1], 3);
  EXPECT_EQ(labels[2], 4);
  EXPECT_EQ(labels[3], -1);
  EXPECT_TRUE(std::isinf(distances[3]));

  params.ef_search = 0;
  EXPECT_THROW(
      index.Search(1, &query, 1, distances.data(), labels.data(), &params),
      hypervec::HypervecException);
}

TEST(IndexNSW, FailedBatchDoesNotPublishCodesOrGraph) {
  hypervec::NSWIndexOptions options;
  options.max_degree = 1;
  options.ef_construction = 3;
  options.ef_search = 3;
  hypervec::IndexNSW index(std::make_unique<ThrowingQuantizer>(), options);
  const std::array<float, 3> database = {0.0F, 1.0F, 2.0F};

  EXPECT_THROW(index.Add(3, database.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.CodeStore().Size(), 0);
  EXPECT_EQ(index.Graph().NodeCount(), 0U);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(index.BuildStats().inserted_nodes, 0U);
}

TEST(IndexNSW, ReconstructCapabilitiesResetAndInputValidation) {
  hypervec::IndexNSWFlat index(2, hypervec::kMetricL2, kExhaustiveOptions);
  const std::array<float, 4> database = {1.0F, 2.0F, 3.0F, 4.0F};
  index.Add(2, database.data());

  const auto capabilities = index.GetCapabilities();
  EXPECT_FALSE(capabilities.requires_training);
  EXPECT_TRUE(capabilities.supports_reconstruct);
  EXPECT_FALSE(capabilities.supports_range_search);
  std::array<float, 2> reconstructed;
  index.Reconstruct(1, reconstructed.data());
  EXPECT_EQ(reconstructed, (std::array<float, 2>{3.0F, 4.0F}));

  EXPECT_THROW(index.Add(-1, nullptr), hypervec::HypervecException);
  EXPECT_THROW(index.Search(-1, nullptr, 1, nullptr, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.Search(1, nullptr, 1, reconstructed.data(), nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.Reconstruct(2, reconstructed.data()),
               hypervec::HypervecException);

  index.Reset();
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.CodeStore().Size(), 0);
  EXPECT_EQ(index.Graph().NodeCount(), 0U);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);

  float empty_distance = 0.0F;
  hypervec::idx_t empty_label = 0;
  index.Search(1, database.data(), 1, &empty_distance, &empty_label);
  EXPECT_TRUE(std::isinf(empty_distance));
  EXPECT_EQ(empty_label, -1);
}

TEST(IndexNSW, RejectsInvalidConstructionOptions) {
  hypervec::NSWIndexOptions zero_search;
  zero_search.ef_search = 0;
  EXPECT_THROW((hypervec::IndexNSWFlat(2, hypervec::kMetricL2, zero_search)),
               hypervec::HypervecException);

  hypervec::NSWIndexOptions narrow_construction;
  narrow_construction.max_degree = 8;
  narrow_construction.ef_construction = 4;
  EXPECT_THROW(
      (hypervec::IndexNSWFlat(2, hypervec::kMetricL2, narrow_construction)),
      hypervec::HypervecException);

  EXPECT_THROW((hypervec::IndexNSW(nullptr, kExhaustiveOptions)),
               hypervec::HypervecException);
}
