/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/index.h>
#include <index/ivf/index_ivf_flat.h>
#include <utils/log/exception.h>

#include <vector>

namespace {

class RecordingBuildIndex final : public hypervec::Index {
 public:
  RecordingBuildIndex() : Index(2, hypervec::kMetricL2) { is_trained = false; }

  bool query_training_called = false;
  int add_calls = 0;

  void Train(hypervec::idx_t, const float*) final { is_trained = true; }

  void Train(hypervec::idx_t, const float*, hypervec::idx_t,
             const float*) final {
    query_training_called = true;
    is_trained = true;
  }

  void Add(hypervec::idx_t n, const float*) final {
    ++add_calls;
    n_total += n;
  }

  void Search(hypervec::idx_t, const float*, hypervec::idx_t, float*,
              hypervec::idx_t*, const hypervec::SearchParameters*) const final {
  }

  void Reset() final { n_total = 0; }
};

class NeverTrainedIndex final : public hypervec::Index {
 public:
  NeverTrainedIndex() : Index(2, hypervec::kMetricL2) { is_trained = false; }

  int add_calls = 0;

  void Add(hypervec::idx_t n, const float*) final {
    ++add_calls;
    n_total += n;
  }

  void Search(hypervec::idx_t, const float*, hypervec::idx_t, float*,
              hypervec::idx_t*, const hypervec::SearchParameters*) const final {
  }

  void Reset() final { n_total = 0; }
};

std::vector<float> IvfTrainingVectors() {
  return {
      0.0f,  0.0f,  0.1f,  0.0f,  0.0f,  0.1f,  0.1f,  0.1f,
      10.0f, 10.0f, 10.1f, 10.0f, 10.0f, 10.1f, 10.1f, 10.1f,
  };
}

}  // namespace

TEST(IndexBuild, BuildsFlatIndexAndMakesItSearchable) {
  hypervec::IndexFlatL2 index(2);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f};

  index.Build(3, vectors.data());
  EXPECT_TRUE(index.is_trained);
  EXPECT_EQ(index.n_total, 3);

  float distance = -1.0f;
  hypervec::idx_t label = -1;
  index.Search(1, vectors.data() + 2, 1, &distance, &label);
  EXPECT_FLOAT_EQ(distance, 0.0f);
  EXPECT_EQ(label, 1);
}

TEST(IndexBuild, TrainsAndPopulatesIvfIndex) {
  hypervec::IndexIVFFlat index(2, 2);
  const auto vectors = IvfTrainingVectors();

  index.Build(8, vectors.data());
  EXPECT_TRUE(index.is_trained);
  EXPECT_EQ(index.n_total, 8);
  EXPECT_EQ(index.invlists->compute_ntotal(), 8);
}

TEST(IndexBuild, QueryAwareBuildFallsBackToRegularTraining) {
  hypervec::IndexIVFFlat index(2, 2);
  const auto vectors = IvfTrainingVectors();
  const std::vector<float> queries = {0.0f, 0.0f, 10.0f, 10.0f};

  index.Build(8, vectors.data(), 2, queries.data());
  EXPECT_TRUE(index.is_trained);
  EXPECT_EQ(index.n_total, 8);
}

TEST(IndexBuild, DispatchesQueryAwareTrainingBeforeAdd) {
  RecordingBuildIndex index;
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const std::vector<float> queries = {0.5f, 0.5f};

  index.Build(2, vectors.data(), 1, queries.data());
  EXPECT_TRUE(index.query_training_called);
  EXPECT_EQ(index.add_calls, 1);
  EXPECT_EQ(index.n_total, 2);
}

TEST(IndexBuild, RejectsNonEmptyIndexBeforeMutation) {
  hypervec::IndexFlatL2 index(2);
  const std::vector<float> vector = {0.0f, 0.0f};
  index.Add(1, vector.data());

  EXPECT_THROW(index.Build(1, vector.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
}

TEST(IndexBuild, ValidatesInputsBeforeTrainingOrAdding) {
  RecordingBuildIndex index;
  const std::vector<float> vector = {0.0f, 0.0f};

  EXPECT_THROW(index.Build(0, vector.data()), hypervec::HypervecException);
  EXPECT_THROW(index.Build(1, nullptr), hypervec::HypervecException);
  EXPECT_THROW(index.Build(1, vector.data(), -1, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.Build(1, vector.data(), 1, nullptr),
               hypervec::HypervecException);
  EXPECT_EQ(index.add_calls, 0);
  EXPECT_FALSE(index.is_trained);
}

TEST(IndexBuild, DoesNotAddWhenTrainingDoesNotComplete) {
  NeverTrainedIndex index;
  const std::vector<float> vector = {0.0f, 0.0f};

  EXPECT_THROW(index.Build(1, vector.data()), hypervec::HypervecException);
  EXPECT_EQ(index.add_calls, 0);
  EXPECT_EQ(index.n_total, 0);
}

TEST(IndexBuild, BuildExRejectsUnsupportedNumericType) {
  RecordingBuildIndex index;
  const std::vector<float> vector = {0.0f, 0.0f};

  EXPECT_THROW(index.BuildEx(1, vector.data(), hypervec::kFloat16),
               hypervec::HypervecException);
  EXPECT_EQ(index.add_calls, 0);
}
