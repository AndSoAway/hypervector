/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <quantization/lvq/lvq_quantizer_adapter.h>
#include <quantization/pq/pq_quantizer_adapter.h>
#include <quantization/quantizer.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {

std::vector<float> TrainingVectors(hypervec::idx_t count,
                                   hypervec::idx_t dimension) {
  std::vector<float> vectors(static_cast<size_t>(count * dimension));
  for (hypervec::idx_t i = 0; i < count; ++i) {
    for (hypervec::idx_t j = 0; j < dimension; ++j) {
      vectors[static_cast<size_t>(i * dimension + j)] =
          0.25f * static_cast<float>(i) +
          0.1f * static_cast<float>((i + 3 * j) % 7);
    }
  }
  return vectors;
}

float L2Squared(const float* lhs, const float* rhs, hypervec::idx_t dimension) {
  float distance = 0.0f;
  for (hypervec::idx_t i = 0; i < dimension; ++i) {
    const float difference = lhs[i] - rhs[i];
    distance += difference * difference;
  }
  return distance;
}

}  // namespace

TEST(InMemoryCodeStore, AppendViewAndReset) {
  hypervec::InMemoryCodeStore store(2);
  const uint8_t first[] = {1, 2, 3, 4};
  const uint8_t second[] = {5, 6};

  store.Append(2, first);
  store.Append(1, second);

  ASSERT_EQ(store.Size(), 3);
  ASSERT_EQ(store.CodeSize(), 2U);
  EXPECT_EQ(store.Code(0)[0], 1);
  EXPECT_EQ(store.Code(1)[1], 4);
  EXPECT_EQ(store.Code(2)[0], 5);

  store.Append(1, store.Data());
  ASSERT_EQ(store.Size(), 4);
  EXPECT_EQ(store.Code(3)[0], 1);
  EXPECT_EQ(store.Code(3)[1], 2);

  const hypervec::EncodedVectorView view = store.View();
  EXPECT_EQ(view.Data(), store.Data());
  EXPECT_EQ(view.Size(), store.Size());
  EXPECT_EQ(view.CodeSize(), store.CodeSize());

  store.Append(0, nullptr);
  EXPECT_EQ(store.Size(), 4);
  store.Reset();
  EXPECT_EQ(store.Size(), 0);
}

TEST(InMemoryCodeStore, RejectsInvalidInputs) {
  EXPECT_THROW(hypervec::InMemoryCodeStore(0), hypervec::HypervecException);

  hypervec::InMemoryCodeStore store(2);
  EXPECT_THROW(store.Append(-1, nullptr), hypervec::HypervecException);
  EXPECT_THROW(store.Append(1, nullptr), hypervec::HypervecException);
  EXPECT_THROW(store.Code(0), hypervec::HypervecException);
  EXPECT_THROW(hypervec::EncodedVectorView(nullptr, 1, 2),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::EncodedVectorView(nullptr, 0, 0),
               hypervec::HypervecException);

  hypervec::InMemoryCodeStore overflow_store(3);
  const uint8_t byte = 0;
  EXPECT_THROW(overflow_store.Append(
                   (std::numeric_limits<hypervec::idx_t>::max)(), &byte),
               hypervec::HypervecException);
}

TEST(FlatQuantizer, CodecAndDistanceComputer) {
  constexpr hypervec::idx_t kDimension = 3;
  const std::vector<float> vectors = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> query = {2.0f, 2.0f, 2.0f};

  hypervec::FlatQuantizer quantizer(kDimension, hypervec::kMetricL2);
  EXPECT_EQ(quantizer.TypeName(), "flat");
  EXPECT_FALSE(quantizer.NeedsTraining());
  EXPECT_TRUE(quantizer.IsTrained());

  quantizer.Train({2, kDimension, vectors.data(), nullptr});
  hypervec::InMemoryCodeStore store(quantizer.CodeSize());
  std::vector<uint8_t> codes(2 * quantizer.CodeSize());
  quantizer.Encode(2, vectors.data(), codes.data());
  store.Append(2, codes.data());

  std::vector<float> decoded(vectors.size());
  quantizer.Decode(2, store.Data(), decoded.data());
  EXPECT_EQ(decoded, vectors);

  std::unique_ptr<hypervec::DistanceComputer> distance =
      quantizer.CreateDistanceComputer(store.View());
  distance->SetQuery(query.data());
  EXPECT_FLOAT_EQ((*distance)(0), 2.0f);
  EXPECT_FLOAT_EQ((*distance)(1), 29.0f);
  EXPECT_FLOAT_EQ(distance->symmetric_dis(0, 1), 27.0f);
  EXPECT_THROW((*distance)(2), hypervec::HypervecException);
}

TEST(QuantizerProtocol, ValidatesDimensionsPointersAndCodeWidth) {
  hypervec::FlatQuantizer quantizer(2, hypervec::kMetricL2);
  const float vector[] = {1.0f, 2.0f};
  float decoded[2];
  uint8_t code[sizeof(vector)];

  EXPECT_THROW(quantizer.Train({1, 3, vector, nullptr}),
               hypervec::HypervecException);
  EXPECT_THROW(quantizer.Encode(1, nullptr, code), hypervec::HypervecException);
  EXPECT_THROW(quantizer.Decode(1, nullptr, decoded),
               hypervec::HypervecException);
  EXPECT_THROW(
      quantizer.CreateDistanceComputer(hypervec::EncodedVectorView(code, 1, 1)),
      hypervec::HypervecException);
}

TEST(ProductQuantizerAdapter, MatchesExistingCodecAndAdc) {
  constexpr hypervec::idx_t kDimension = 4;
  constexpr hypervec::idx_t kCount = 32;
  std::vector<float> vectors = TrainingVectors(kCount, kDimension);

  hypervec::ProductQuantizer model(kDimension, 2, 2);
  hypervec::PQParameters parameters;
  parameters.niter = 5;
  hypervec::ProductQuantizerAdapter quantizer(model, parameters);
  quantizer.Train({kCount, kDimension, vectors.data(), nullptr});

  EXPECT_TRUE(model.is_trained);
  EXPECT_TRUE(quantizer.IsTrained());
  EXPECT_EQ(quantizer.CodeSize(), model.code_size);

  std::vector<uint8_t> adapter_codes(static_cast<size_t>(kCount) *
                                     model.code_size);
  std::vector<uint8_t> direct_codes(adapter_codes.size());
  quantizer.Encode(kCount, vectors.data(), adapter_codes.data());
  model.ComputeCodes(kCount, vectors.data(), direct_codes.data());
  EXPECT_EQ(adapter_codes, direct_codes);
  std::vector<uint8_t> single_code(model.code_size);
  quantizer.Encode(1, vectors.data() + 3 * kDimension, single_code.data());
  EXPECT_TRUE(std::equal(single_code.begin(), single_code.end(),
                         direct_codes.begin() + 3 * model.code_size));

  std::vector<float> adapter_decoded(static_cast<size_t>(kCount) * kDimension);
  std::vector<float> direct_decoded(adapter_decoded.size());
  quantizer.Decode(kCount, adapter_codes.data(), adapter_decoded.data());
  model.DecodeBatch(kCount, direct_codes.data(), direct_decoded.data());
  EXPECT_EQ(adapter_decoded, direct_decoded);

  const hypervec::EncodedVectorView view(adapter_codes.data(), kCount,
                                         model.code_size);
  std::unique_ptr<hypervec::DistanceComputer> distance =
      quantizer.CreateDistanceComputer(view);
  distance->SetQuery(vectors.data() + 3 * kDimension);

  std::vector<float> table(static_cast<size_t>(model.M * model.ksub));
  model.ComputeDistanceTable(vectors.data() + 3 * kDimension, table.data());
  for (hypervec::idx_t i = 0; i < kCount; ++i) {
    const float expected = model.ApplyDistanceTable(
        table.data(), direct_codes.data() + i * model.code_size);
    EXPECT_FLOAT_EQ((*distance)(i), expected);
  }
}

TEST(LocalVectorQuantizerAdapter, MatchesExistingCodecAndAdc) {
  constexpr hypervec::idx_t kDimension = 4;
  constexpr hypervec::idx_t kCount = 32;
  std::vector<float> vectors = TrainingVectors(kCount, kDimension);

  hypervec::LocalVectorQuantizer model(kDimension, 2, 2);
  hypervec::LVQParameters parameters;
  parameters.niter = 5;
  hypervec::LocalVectorQuantizerAdapter quantizer(model, parameters);
  quantizer.Train({kCount, kDimension, vectors.data(), nullptr});

  EXPECT_TRUE(model.is_trained);
  EXPECT_TRUE(quantizer.IsTrained());
  EXPECT_EQ(quantizer.CodeSize(), model.code_size);

  std::vector<uint8_t> adapter_codes(static_cast<size_t>(kCount) *
                                     model.code_size);
  std::vector<uint8_t> direct_codes(adapter_codes.size());
  quantizer.Encode(kCount, vectors.data(), adapter_codes.data());
  model.ComputeCodes(kCount, vectors.data(), direct_codes.data());
  EXPECT_EQ(adapter_codes, direct_codes);
  std::vector<uint8_t> single_code(model.code_size);
  quantizer.Encode(1, vectors.data() + 3 * kDimension, single_code.data());
  EXPECT_TRUE(std::equal(single_code.begin(), single_code.end(),
                         direct_codes.begin() + 3 * model.code_size));

  std::vector<float> adapter_decoded(static_cast<size_t>(kCount) * kDimension);
  std::vector<float> direct_decoded(adapter_decoded.size());
  quantizer.Decode(kCount, adapter_codes.data(), adapter_decoded.data());
  model.DecodeBatch(kCount, direct_codes.data(), direct_decoded.data());
  EXPECT_EQ(adapter_decoded, direct_decoded);

  const hypervec::EncodedVectorView view(adapter_codes.data(), kCount,
                                         model.code_size);
  std::unique_ptr<hypervec::DistanceComputer> distance =
      quantizer.CreateDistanceComputer(view);
  distance->SetQuery(vectors.data() + 3 * kDimension);

  std::vector<float> table(static_cast<size_t>(model.nlocal * model.ksub));
  model.ComputeDistanceTable(vectors.data() + 3 * kDimension, table.data());
  for (hypervec::idx_t i = 0; i < kCount; ++i) {
    const float expected = model.ApplyDistanceTable(
        table.data(), direct_codes.data() + i * model.code_size);
    EXPECT_FLOAT_EQ((*distance)(i), expected);
  }

  std::vector<float> lhs(kDimension);
  std::vector<float> rhs(kDimension);
  model.Decode(direct_codes.data(), lhs.data());
  model.Decode(direct_codes.data() + model.code_size, rhs.data());
  EXPECT_FLOAT_EQ(distance->symmetric_dis(0, 1),
                  L2Squared(lhs.data(), rhs.data(), kDimension));
}

TEST(QuantizerAdapter, ReadOnlyModelCannotBeTrained) {
  const hypervec::ProductQuantizer model(4, 2, 2);
  hypervec::ProductQuantizerAdapter quantizer(model);
  const std::vector<float> vectors = TrainingVectors(4, 4);
  EXPECT_THROW(quantizer.Train({4, 4, vectors.data(), nullptr}),
               hypervec::HypervecException);
}

TEST(QuantizerAdapter, UntrainedModelsRejectCodecOperations) {
  const hypervec::ProductQuantizer model(4, 2, 2);
  const hypervec::ProductQuantizerAdapter quantizer(model);
  const float vector[4] = {};
  uint8_t code[1] = {};
  float decoded[4] = {};

  EXPECT_THROW(quantizer.Encode(1, vector, code), hypervec::HypervecException);
  EXPECT_THROW(quantizer.Decode(1, code, decoded), hypervec::HypervecException);
  EXPECT_THROW(quantizer.CreateDistanceComputer(
                   hypervec::EncodedVectorView(code, 1, sizeof(code))),
               hypervec::HypervecException);
}
