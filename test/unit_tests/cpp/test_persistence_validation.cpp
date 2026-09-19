/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <utils/log/exception.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

template <typename T>
void WriteOne(hypervec::VectorIOWriter* writer, const T& value) {
  ASSERT_EQ((*writer)(&value, sizeof(T), 1), 1);
}

template <typename T>
void WriteVector(hypervec::VectorIOWriter* writer,
                 const std::vector<T>& values) {
  const size_t size = values.size();
  WriteOne(writer, size);
  ASSERT_EQ((*writer)(values.data(), sizeof(T), size), size);
}

void WriteIndexHeader(hypervec::VectorIOWriter* writer, hypervec::idx_t d,
                      hypervec::idx_t n_total, uint8_t is_trained,
                      hypervec::MetricType metric = hypervec::kMetricL2) {
  WriteOne(writer, d);
  WriteOne(writer, n_total);
  const hypervec::idx_t dummy = 1 << 20;
  WriteOne(writer, dummy);
  WriteOne(writer, dummy);
  WriteOne(writer, is_trained);
  const int serialized_metric = static_cast<int>(metric);
  WriteOne(writer, serialized_metric);
  if (metric > hypervec::kMetricL2) {
    const float metric_arg = 0.0f;
    WriteOne(writer, metric_arg);
  }
}

void WritePq(hypervec::VectorIOWriter* writer, hypervec::idx_t d,
             hypervec::idx_t m, int nbits,
             const std::vector<float>& centroids) {
  WriteOne(writer, d);
  WriteOne(writer, m);
  WriteOne(writer, nbits);
  WriteVector(writer, centroids);
}

TEST(PersistenceValidation, RejectsFlatCodeCountMismatch) {
  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("IFlm");
  WriteOne(&writer, tag);
  WriteIndexHeader(&writer, 2, 2, true);
  WriteVector(&writer, std::vector<uint8_t>(3));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsInvalidTrainingFlag) {
  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("IFlm");
  WriteOne(&writer, tag);
  WriteIndexHeader(&writer, 2, 0, 2);

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsPqCentroidCountMismatch) {
  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("PqPq");
  WriteOne(&writer, tag);
  WritePq(&writer, 4, 2, 2, std::vector<float>(15));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::read_ProductQuantizer_up(&reader),
               hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsIndexPqCodeCountMismatch) {
  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("IPQ8");
  WriteOne(&writer, tag);
  WriteIndexHeader(&writer, 4, 2, true);
  WritePq(&writer, 4, 2, 1, std::vector<float>(8));
  WriteVector(&writer, std::vector<uint8_t>(1));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsIvfCentroidCountMismatch) {
  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("IVFf");
  WriteOne(&writer, tag);
  WriteIndexHeader(&writer, 2, 0, true);
  const hypervec::idx_t nlist = 2;
  const hypervec::idx_t nprobe = 1;
  WriteOne(&writer, nlist);
  WriteOne(&writer, nprobe);
  WriteVector(&writer, std::vector<float>(3));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsIvfPqPrecomputedTableCountMismatch) {
  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("IVPQ");
  WriteOne(&writer, tag);
  WriteIndexHeader(&writer, 4, 0, true);
  const hypervec::idx_t nlist = 2;
  const hypervec::idx_t nprobe = 1;
  WriteOne(&writer, nlist);
  WriteOne(&writer, nprobe);
  WriteVector(&writer, std::vector<float>(8));
  const int8_t by_residual = 1;
  const int use_precomputed_table = 1;
  WriteOne(&writer, by_residual);
  WriteOne(&writer, use_precomputed_table);
  WritePq(&writer, 4, 2, 1, std::vector<float>(8));
  WriteVector(&writer, std::vector<float>(7));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsInvalidIvfPqModeBeforeCodecPayload) {
  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("IVPQ");
  WriteOne(&writer, tag);
  WriteIndexHeader(&writer, 4, 0, true);
  const hypervec::idx_t nlist = 1;
  const hypervec::idx_t nprobe = 1;
  WriteOne(&writer, nlist);
  WriteOne(&writer, nprobe);
  WriteVector(&writer, std::vector<float>(4));
  const int8_t invalid_by_residual = 2;
  const int use_precomputed_table = 0;
  WriteOne(&writer, invalid_by_residual);
  WriteOne(&writer, use_precomputed_table);

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, PreservesUntrainedPqState) {
  hypervec::IndexPQ source(4, 2, 1);
  ASSERT_FALSE(source.is_trained);
  ASSERT_FALSE(source.pq.is_trained);

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> restored_base =
      hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexPQ*>(restored_base.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_FALSE(restored->is_trained);
  EXPECT_FALSE(restored->pq.is_trained);
  EXPECT_EQ(restored->n_total, 0);
}

}  // namespace
