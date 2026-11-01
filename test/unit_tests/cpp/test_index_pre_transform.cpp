/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/pretransform/index_pre_transform.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <quantization/pq/index_pq.h>
#include <transform/opq_matrix.h>
#include <transform/vector_transform.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>

#include <memory>
#include <utility>
#include <vector>

namespace {

std::vector<float> TrainingData(hypervec::idx_t n) {
  constexpr hypervec::idx_t kDimension = 8;
  std::vector<float> data(static_cast<size_t>(n * kDimension));
  for (hypervec::idx_t i = 0; i < n; ++i) {
    for (hypervec::idx_t j = 0; j < kDimension; ++j) {
      data[i * kDimension + j] =
          static_cast<float>(((i * (j + 3) + j * j) % 37) - 18) / 9.0F;
    }
  }
  return data;
}

TEST(IndexPreTransform, IdentityPreservesFlatIndexBehavior) {
  auto linear = std::make_unique<hypervec::LinearTransform>(2, 2);
  linear->SetIdentity();
  auto flat = std::make_unique<hypervec::IndexFlatL2>(2);
  hypervec::IndexPreTransform index(std::move(linear), std::move(flat));
  EXPECT_TRUE(index.is_trained);

  const std::vector<float> database = {0.0F, 0.0F, 1.0F, 0.0F,
                                       0.0F, 2.0F, 3.0F, 3.0F};
  index.Add(4, database.data());
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  const float query[2] = {0.9F, 0.1F};
  index.Search(1, query, 1, &distance, &label);
  EXPECT_EQ(label, 1);
  EXPECT_NEAR(distance, 0.02F, 1e-6F);

  float reconstructed[2] = {};
  index.Reconstruct(2, reconstructed);
  EXPECT_FLOAT_EQ(reconstructed[0], 0.0F);
  EXPECT_FLOAT_EQ(reconstructed[1], 2.0F);

  hypervec::RangeSearchResult range(1);
  index.RangeSearch(1, query, 0.5F, &range);
  EXPECT_EQ(range.lims[1], 1U);
  EXPECT_EQ(range.labels[0], 1);
}

TEST(IndexPreTransform, OPQAndPQShareTrainAddSearchAndReconstructPipeline) {
  constexpr hypervec::idx_t n = 128;
  const std::vector<float> data = TrainingData(n);
  auto opq = std::make_unique<hypervec::OPQMatrix>(8, 4, 2);
  opq->parameters.iterations = 2;
  opq->parameters.pq_parameters.niter = 6;
  hypervec::OPQMatrix* opq_view = opq.get();
  auto pq = std::make_unique<hypervec::IndexPQ>(8, 4, 2);
  hypervec::IndexPQ* pq_view = pq.get();
  hypervec::IndexPreTransform index(std::move(opq), std::move(pq));

  EXPECT_FALSE(index.is_trained);
  EXPECT_TRUE(index.GetCapabilities().requires_training);
  EXPECT_FALSE(index.GetCapabilities().supports_reconstruct);
  index.Build(n, data.data(), 3, data.data());
  EXPECT_TRUE(index.is_trained);
  EXPECT_EQ(index.n_total, n);
  EXPECT_TRUE(index.GetCapabilities().supports_reconstruct);

  const std::vector<float> transformed_query = opq_view->Apply(1, data.data());
  float expected_distance = 0.0F;
  hypervec::idx_t expected_label = -1;
  pq_view->Search(1, transformed_query.data(), 1, &expected_distance,
                  &expected_label);
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  index.Search(1, data.data(), 1, &distance, &label);
  EXPECT_EQ(label, expected_label);
  EXPECT_FLOAT_EQ(distance, expected_distance);

  float inner_reconstruction[8] = {};
  float expected_reconstruction[8] = {};
  float reconstruction[8] = {};
  pq_view->Reconstruct(label, inner_reconstruction);
  opq_view->ReverseTransform(1, inner_reconstruction, expected_reconstruction);
  index.Reconstruct(label, reconstruction);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(reconstruction[i], expected_reconstruction[i], 1e-6F);
  }
}

TEST(IndexPreTransform, OPQPersistencePreservesPipelineAndTrainingOptions) {
  constexpr hypervec::idx_t n = 128;
  const std::vector<float> data = TrainingData(n);
  auto opq = std::make_unique<hypervec::OPQMatrix>(8, 4, 2);
  opq->parameters.iterations = 2;
  opq->parameters.pq_parameters.niter = 6;
  opq->parameters.pq_parameters.seed = 2026;
  opq->parameters.pq_parameters.nredo = 2;
  auto pq = std::make_unique<hypervec::IndexPQ>(8, 4, 2);
  hypervec::IndexPreTransform source(std::move(opq), std::move(pq));
  source.Build(n, data.data());

  float expected_distances[6] = {};
  hypervec::idx_t expected_labels[6] = {};
  source.Search(2, data.data(), 3, expected_distances, expected_labels);

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> loaded = hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexPreTransform*>(loaded.get());
  ASSERT_NE(restored, nullptr);
  auto* restored_opq =
      dynamic_cast<hypervec::OPQMatrix*>(restored->transform.get());
  auto* restored_pq = dynamic_cast<hypervec::IndexPQ*>(restored->index.get());
  ASSERT_NE(restored_opq, nullptr);
  ASSERT_NE(restored_pq, nullptr);
  EXPECT_EQ(restored_opq->subquantizer_count, 4);
  EXPECT_EQ(restored_opq->nbits, 2);
  EXPECT_EQ(restored_opq->parameters.iterations, 2);
  EXPECT_EQ(restored_opq->parameters.pq_parameters.niter, 6);
  EXPECT_EQ(restored_opq->parameters.pq_parameters.seed, 2026);
  EXPECT_EQ(restored_opq->parameters.pq_parameters.nredo, 2);
  EXPECT_EQ(restored_opq->matrix,
            dynamic_cast<hypervec::OPQMatrix*>(source.transform.get())->matrix);

  float actual_distances[6] = {};
  hypervec::idx_t actual_labels[6] = {};
  restored->Search(2, data.data(), 3, actual_distances, actual_labels);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(actual_distances[i], expected_distances[i]);
    EXPECT_EQ(actual_labels[i], expected_labels[i]);
  }

  const std::vector<float> appended(data.begin(), data.begin() + 8);
  restored->Add(1, appended.data());
  EXPECT_EQ(restored->n_total, n + 1);
}

TEST(IndexPreTransform, LinearPersistencePreservesAffineTransform) {
  auto linear = std::make_unique<hypervec::LinearTransform>(2, 2);
  linear->SetTransform({0.0F, -1.0F, 1.0F, 0.0F}, {2.0F, -3.0F}, true);
  auto flat = std::make_unique<hypervec::IndexFlatL2>(2);
  hypervec::IndexPreTransform source(std::move(linear), std::move(flat));
  const std::vector<float> database = {0.0F, 0.0F, 1.0F, 2.0F, -3.0F, 4.0F};
  source.Add(3, database.data());

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> loaded = hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexPreTransform*>(loaded.get());
  ASSERT_NE(restored, nullptr);
  auto* restored_linear =
      dynamic_cast<hypervec::LinearTransform*>(restored->transform.get());
  ASSERT_NE(restored_linear, nullptr);
  EXPECT_EQ(restored_linear->bias, (std::vector<float>{2.0F, -3.0F}));
  EXPECT_TRUE(restored_linear->is_orthonormal);

  float reconstruction[2] = {};
  restored->Reconstruct(1, reconstruction);
  EXPECT_NEAR(reconstruction[0], 1.0F, 1e-6F);
  EXPECT_NEAR(reconstruction[1], 2.0F, 1e-6F);
}

TEST(IndexPreTransform, StandaloneCodecUsesBothTransformDirections) {
  auto linear = std::make_unique<hypervec::LinearTransform>(2, 2);
  linear->SetTransform({0.0F, -1.0F, 1.0F, 0.0F}, {}, true);
  auto pq = std::make_unique<hypervec::IndexPQ>(2, 1, 1);
  hypervec::IndexPreTransform index(std::move(linear), std::move(pq));
  const std::vector<float> training = {-2.0F, 0.0F, -1.0F, 0.0F,
                                       1.0F,  0.0F, 2.0F,  0.0F};
  index.Train(4, training.data());

  const size_t code_size = index.SaCodeSize();
  std::vector<uint8_t> code(code_size);
  float decoded[2] = {};
  index.SaEncode(1, training.data(), code.data());
  index.SaDecode(1, code.data(), decoded);
  EXPECT_NEAR(decoded[1], 0.0F, 1e-6F);
  EXPECT_LT(decoded[0], 0.0F);
}

TEST(IndexPreTransform, ValidatesCompositionAndLifecycle) {
  auto identity = std::make_unique<hypervec::LinearTransform>(2, 2);
  identity->SetIdentity();
  EXPECT_THROW(hypervec::IndexPreTransform(
                   nullptr, std::make_unique<hypervec::IndexFlatL2>(2)),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::IndexPreTransform(std::move(identity), nullptr),
               hypervec::HypervecException);

  auto wrong_dimension = std::make_unique<hypervec::LinearTransform>(3, 3);
  wrong_dimension->SetIdentity();
  EXPECT_THROW(
      hypervec::IndexPreTransform(std::move(wrong_dimension),
                                  std::make_unique<hypervec::IndexFlatL2>(2)),
      hypervec::HypervecException);

  auto unconfigured = std::make_unique<hypervec::LinearTransform>(2, 2);
  EXPECT_THROW(
      hypervec::IndexPreTransform(std::move(unconfigured),
                                  std::make_unique<hypervec::IndexFlatL2>(2)),
      hypervec::HypervecException);

  auto projection = std::make_unique<hypervec::LinearTransform>(2, 1);
  projection->SetTransform({1.0F, 0.0F});
  hypervec::IndexPreTransform non_reversible(
      std::move(projection), std::make_unique<hypervec::IndexFlatL2>(1));
  const float projected_data[2] = {1.0F, 2.0F};
  non_reversible.Add(1, projected_data);
  EXPECT_FALSE(non_reversible.GetCapabilities().supports_reconstruct);
  float reconstruction[2] = {};
  EXPECT_THROW(non_reversible.Reconstruct(0, reconstruction),
               hypervec::HypervecException);

  auto opq = std::make_unique<hypervec::OPQMatrix>(8, 4, 2);
  opq->parameters.iterations = 1;
  opq->parameters.pq_parameters.niter = 3;
  hypervec::IndexPreTransform index(
      std::move(opq), std::make_unique<hypervec::IndexPQ>(8, 4, 2));
  const std::vector<float> data = TrainingData(32);
  EXPECT_THROW(index.Add(1, data.data()), hypervec::HypervecException);
  hypervec::VectorIOWriter writer;
  EXPECT_THROW(hypervec::WriteIndex(&index, &writer),
               hypervec::HypervecException);
  index.Build(32, data.data());
  EXPECT_THROW(index.Train(32, data.data()), hypervec::HypervecException);
  index.Reset();
  EXPECT_TRUE(index.is_trained);
  EXPECT_EQ(index.n_total, 0);
}

}  // namespace
