/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <utils/log/exception.h>

#include <algorithm>
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

void WriteIvfFlatPrefix(hypervec::VectorIOWriter* writer, hypervec::idx_t d,
                        hypervec::idx_t n_total, hypervec::idx_t nlist) {
  const uint32_t tag = hypervec::fourcc("IVFf");
  WriteOne(writer, tag);
  WriteIndexHeader(writer, d, n_total, true);
  const hypervec::idx_t nprobe = 1;
  WriteOne(writer, nlist);
  WriteOne(writer, nprobe);
  WriteVector(writer, std::vector<float>(static_cast<size_t>(d * nlist), 0.0f));
}

void WriteIvfFlatList(hypervec::VectorIOWriter* writer,
                      const std::vector<hypervec::idx_t>& ids,
                      hypervec::idx_t d) {
  const size_t list_size = ids.size();
  WriteOne(writer, list_size);
  ASSERT_EQ((*writer)(ids.data(), sizeof(hypervec::idx_t), ids.size()),
            ids.size());
  const std::vector<uint8_t> codes(list_size * static_cast<size_t>(d) *
                                   sizeof(float));
  ASSERT_EQ((*writer)(codes.data(), sizeof(uint8_t), codes.size()),
            codes.size());
}

class DeserializationLimitsGuard {
 public:
  DeserializationLimitsGuard()
      : loop_limit_(hypervec::get_deserialization_loop_limit()),
        vector_byte_limit_(hypervec::get_deserialization_vector_byte_limit()) {}

  ~DeserializationLimitsGuard() {
    hypervec::set_deserialization_loop_limit(loop_limit_);
    hypervec::set_deserialization_vector_byte_limit(vector_byte_limit_);
  }

 private:
  size_t loop_limit_;
  size_t vector_byte_limit_;
};

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

TEST(PersistenceValidation, RejectsIvfListTotalBelowHeaderCount) {
  hypervec::VectorIOWriter writer;
  WriteIvfFlatPrefix(&writer, 2, 2, 1);
  WriteIvfFlatList(&writer, {7}, 2);

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsIvfListTotalAboveHeaderCountEarly) {
  hypervec::VectorIOWriter writer;
  WriteIvfFlatPrefix(&writer, 2, 1, 1);
  const size_t invalid_list_size = 2;
  WriteOne(&writer, invalid_list_size);

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, EnforcesIvfLoopLimitBeforeReadingCentroids) {
  DeserializationLimitsGuard guard;
  hypervec::set_deserialization_loop_limit(1);

  hypervec::VectorIOWriter writer;
  const uint32_t tag = hypervec::fourcc("IVFf");
  WriteOne(&writer, tag);
  WriteIndexHeader(&writer, 2, 0, true);
  const hypervec::idx_t nlist = 2;
  const hypervec::idx_t nprobe = 1;
  WriteOne(&writer, nlist);
  WriteOne(&writer, nprobe);

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, EnforcesByteLimitForIvfListIds) {
  DeserializationLimitsGuard guard;
  hypervec::set_deserialization_vector_byte_limit(12);

  hypervec::VectorIOWriter writer;
  WriteIvfFlatPrefix(&writer, 1, 2, 1);
  const size_t list_size = 2;
  WriteOne(&writer, list_size);

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

TEST(PersistenceValidation, RebuildsOneHnswProbabilityTableAndAllowsAdd) {
  hypervec::IndexHNSWFlat source(2, 8);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f,
                                      2.0f, 2.0f, 3.0f, 3.0f};
  source.Add(4, vectors.data());
  const auto expected_probabilities = source.hnsw.assign_probas;

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> restored_base =
      hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexHNSWFlat*>(restored_base.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->hnsw.assign_probas, expected_probabilities);
  EXPECT_EQ(restored->hnsw.cum_nneighbor_per_level,
            source.hnsw.cum_nneighbor_per_level);

  const std::vector<float> appended = {4.0f, 4.0f};
  restored->Add(1, appended.data());
  EXPECT_EQ(restored->n_total, 5);
  EXPECT_EQ(restored->storage->n_total, 5);
  EXPECT_EQ(restored->hnsw.levels.size(), 5);
  EXPECT_EQ(restored->hnsw.offsets.size(), 6);
}

TEST(PersistenceValidation, RejectsHnswOffsetMismatch) {
  hypervec::IndexHNSWFlat source(2, 8);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f,
                                      2.0f, 2.0f, 3.0f, 3.0f};
  source.Add(4, vectors.data());
  source.hnsw.offsets.back() += 1;

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsHnswOutOfRangeNeighbor) {
  hypervec::IndexHNSWFlat source(2, 8);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f,
                                      2.0f, 2.0f, 3.0f, 3.0f};
  source.Add(4, vectors.data());
  auto neighbor = std::find_if(
      source.hnsw.neighbors.begin(), source.hnsw.neighbors.end(),
      [](hypervec::HNSW::storage_idx_t value) { return value >= 0; });
  ASSERT_NE(neighbor, source.hnsw.neighbors.end());
  *neighbor = 100;

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RoundtripsUntrainedHnswPqState) {
  hypervec::IndexHNSWPQ source(4, 2, 1, 8);
  ASSERT_FALSE(source.is_trained);
  ASSERT_FALSE(source.storage->is_trained);

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> restored_base =
      hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexHNSWPQ*>(restored_base.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_FALSE(restored->is_trained);
  EXPECT_FALSE(restored->storage->is_trained);
  EXPECT_EQ(restored->n_total, 0);
  EXPECT_EQ(restored->hnsw.offsets, (std::vector<size_t>{0}));
}

TEST(PersistenceValidation, RoundtripsIdMapWithExternalIds) {
  hypervec::IndexFlatL2 storage(2);
  hypervec::IndexIDMap source(&storage);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f};
  const std::vector<hypervec::idx_t> external_ids = {101, 205, 999};
  source.AddWithIds(3, vectors.data(), external_ids.data());

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> restored_base =
      hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexIDMap*>(restored_base.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_TRUE(restored->own_fields);
  EXPECT_EQ(restored->rev_map, external_ids);
  EXPECT_EQ(restored->to_internal(205), 1);
  EXPECT_EQ(restored->from_internal(2), 999);

  float distance = -1.0f;
  hypervec::idx_t label = -1;
  restored->Search(1, vectors.data() + 2, 1, &distance, &label);
  EXPECT_FLOAT_EQ(distance, 0.0f);
  EXPECT_EQ(label, 205);

  const std::vector<float> appended = {3.0f, 3.0f};
  const hypervec::idx_t appended_id = 4001;
  restored->AddWithIds(1, appended.data(), &appended_id);
  EXPECT_EQ(restored->n_total, 4);
  EXPECT_EQ(restored->to_internal(appended_id), 3);
}

TEST(PersistenceValidation, RejectsDuplicateIdMapExternalIds) {
  hypervec::VectorIOWriter writer;
  const uint32_t id_map_tag = hypervec::fourcc("IxMp");
  WriteOne(&writer, id_map_tag);
  WriteIndexHeader(&writer, 2, 2, true);
  WriteVector(&writer, std::vector<hypervec::idx_t>{7, 7});

  const uint32_t flat_tag = hypervec::fourcc("IFlm");
  WriteOne(&writer, flat_tag);
  WriteIndexHeader(&writer, 2, 2, true);
  WriteVector(&writer, std::vector<uint8_t>(4 * sizeof(float)));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsNegativeIdMapExternalId) {
  hypervec::VectorIOWriter writer;
  const uint32_t id_map_tag = hypervec::fourcc("IxMp");
  WriteOne(&writer, id_map_tag);
  WriteIndexHeader(&writer, 2, 1, true);
  WriteVector(&writer, std::vector<hypervec::idx_t>{-1});

  const uint32_t flat_tag = hypervec::fourcc("IFlm");
  WriteOne(&writer, flat_tag);
  WriteIndexHeader(&writer, 2, 1, true);
  WriteVector(&writer, std::vector<uint8_t>(2 * sizeof(float)));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsIdMapStorageMetadataMismatch) {
  hypervec::VectorIOWriter writer;
  const uint32_t id_map_tag = hypervec::fourcc("IxMp");
  WriteOne(&writer, id_map_tag);
  WriteIndexHeader(&writer, 2, 1, true);
  WriteVector(&writer, std::vector<hypervec::idx_t>{7});

  const uint32_t flat_tag = hypervec::fourcc("IFlm");
  WriteOne(&writer, flat_tag);
  WriteIndexHeader(&writer, 3, 1, true);
  WriteVector(&writer, std::vector<uint8_t>(3 * sizeof(float)));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

TEST(PersistenceValidation, RejectsIdMapExternalIdCountMismatch) {
  hypervec::VectorIOWriter writer;
  const uint32_t id_map_tag = hypervec::fourcc("IxMp");
  WriteOne(&writer, id_map_tag);
  WriteIndexHeader(&writer, 2, 2, true);
  WriteVector(&writer, std::vector<hypervec::idx_t>{7});

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  EXPECT_THROW(hypervec::ReadIndexUp(&reader), hypervec::HypervecException);
}

}  // namespace
