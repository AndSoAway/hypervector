/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/engine.h>
#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/idmap/index_id_map.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <utils/log/exception.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

template <typename T>
void AppendValue(VectorIOWriter* writer, const T& value) {
  ASSERT_EQ((*writer)(&value, sizeof(T), 1), 1);
}

void AppendBytes(VectorIOWriter* writer, const void* data, size_t size) {
  ASSERT_EQ((*writer)(data, 1, size), size);
}

std::vector<uint8_t> MakeLegacyFlatL2File() {
  VectorIOWriter writer;
  const uint32_t magic = fourcc("IFlm");
  const int dimension = 2;
  const idx_t count = 2;
  const idx_t dummy = 1 << 20;
  const bool trained = true;
  const int metric = static_cast<int>(kMetricL2);
  const std::vector<float> vectors = {0.0F, 0.0F, 1.0F, 1.0F};
  const size_t code_bytes = vectors.size() * sizeof(float);

  AppendValue(&writer, magic);
  AppendValue(&writer, dimension);
  AppendValue(&writer, count);
  AppendValue(&writer, dummy);
  AppendValue(&writer, dummy);
  AppendValue(&writer, trained);
  AppendValue(&writer, metric);
  AppendValue(&writer, code_bytes);
  AppendBytes(&writer, vectors.data(), code_bytes);
  return std::move(writer.data);
}

size_t EnvelopeSize(const std::vector<uint8_t>& bytes) {
  constexpr size_t config_size_offset =
      sizeof(uint32_t) + 2 * sizeof(uint16_t) + sizeof(uint32_t);
  EXPECT_GE(bytes.size(), config_size_offset + sizeof(uint64_t));
  uint64_t config_size = 0;
  std::memcpy(&config_size, bytes.data() + config_size_offset,
              sizeof(config_size));
  const size_t envelope_size = config_size_offset + sizeof(uint64_t) +
                               static_cast<size_t>(config_size) +
                               sizeof(uint32_t);
  EXPECT_LE(envelope_size, bytes.size());
  return envelope_size;
}

std::vector<uint8_t> WriteToBytes(const Index& index) {
  VectorIOWriter writer;
  WriteIndex(&index, &writer);
  return std::move(writer.data);
}

std::unique_ptr<Index> ReadFromBytes(std::vector<uint8_t> bytes,
                                     IndexFileMetadata* metadata = nullptr) {
  VectorIOReader reader;
  reader.data = std::move(bytes);
  return ReadIndexUp(&reader, metadata);
}

TEST(IndexPersistenceMetadata, RoundtripsEveryFactoryBuiltinConfig) {
  Engine engine;
  const std::vector<std::string> configs = {
      R"({"type":"flat","dimension":8})",
      R"({"type":"pq","dimension":8,"storage":{"pq_m":2,"nbits":2}})",
      R"({"type":"lvq","dimension":8,"storage":{"nlocal":2,"nbits":2}})",
      R"({"type":"ivf_flat","dimension":8,"build":{"nlist":4},"defaults":{"nprobe":2}})",
      R"({"type":"ivf_pq","dimension":8,"storage":{"pq_m":2,"nbits":2},"build":{"nlist":4},"defaults":{"nprobe":2}})",
      R"({"type":"ivf_lvq","dimension":8,"storage":{"nlocal":2,"nbits":2},"build":{"nlist":4},"defaults":{"nprobe":2}})",
      R"({"type":"hnsw_flat","dimension":8,"build":{"graph_degree":4}})",
      R"({"type":"hnsw_pq","dimension":8,"storage":{"pq_m":2,"nbits":2},"build":{"graph_degree":4}})",
      R"({"type":"hnsw_lvq","dimension":8,"storage":{"nlocal":2,"nbits":2},"build":{"graph_degree":4}})",
      R"({"type":"nsw_flat","dimension":8,"build":{"graph_degree":4}})",
  };
  std::vector<float> vectors(32 * 8);
  for (size_t row = 0; row < 32; ++row) {
    for (size_t column = 0; column < 8; ++column) {
      vectors[row * 8 + column] =
          static_cast<float>((row * (column + 3) + column * 7) % 29) / 29.0F;
    }
  }

  for (const std::string& json : configs) {
    SCOPED_TRACE(json);
    const IndexConfig expected = engine.NormalizeConfig(ParseIndexConfig(json));
    std::unique_ptr<Index> index = engine.CreateIndex(expected);
    index->Build({32, 8, vectors.data(), nullptr});

    const IndexFileMetadata source_metadata = GetIndexFileMetadata(*index);
    EXPECT_EQ(SerializeIndexConfig(source_metadata.config),
              SerializeIndexConfig(expected));
    EXPECT_EQ(source_metadata.format_major, kIndexFileFormatMajor);
    EXPECT_EQ(source_metadata.algorithm_version, kIndexAlgorithmVersion);
    EXPECT_FALSE(source_metadata.is_legacy);

    IndexFileMetadata restored_metadata;
    std::unique_ptr<Index> restored =
        ReadFromBytes(WriteToBytes(*index), &restored_metadata);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->n_total, index->n_total);
    EXPECT_EQ(SerializeIndexConfig(restored_metadata.config),
              SerializeIndexConfig(expected));
  }
}

TEST(IndexPersistenceMetadata, RoundtripRestoresConfigAndHnswSearchDefault) {
  const std::string config_json = R"({
    "type":"hnsw_flat", "dimension":2, "id_mode":"external",
    "build":{"graph_degree":2,"ef_construction":77},
    "defaults":{"ef_search":31}
  })";
  Engine engine;
  const IndexConfig expected =
      engine.NormalizeConfig(ParseIndexConfig(config_json));
  std::unique_ptr<Index> source = engine.CreateIndex(expected);
  const std::vector<float> vectors = {0.0F, 0.0F, 1.0F, 0.0F,
                                      0.0F, 1.0F, 1.0F, 1.0F};
  const std::vector<idx_t> ids = {10, 20, 30, 40};
  source->Build({4, 2, vectors.data(), ids.data()});

  std::vector<uint8_t> bytes = WriteToBytes(*source);
  uint32_t magic = 0;
  std::memcpy(&magic, bytes.data(), sizeof(magic));
  EXPECT_EQ(magic, fourcc("HVX2"));

  IndexFileMetadata metadata;
  std::unique_ptr<Index> restored = ReadFromBytes(std::move(bytes), &metadata);
  EXPECT_EQ(SerializeIndexConfig(metadata.config),
            SerializeIndexConfig(expected));
  EXPECT_EQ(metadata.format_major, kIndexFileFormatMajor);
  EXPECT_EQ(metadata.format_minor, kIndexFileFormatMinor);
  EXPECT_EQ(metadata.algorithm_version, kIndexAlgorithmVersion);
  EXPECT_FALSE(metadata.is_legacy);

  auto* id_map = dynamic_cast<IndexIDMap*>(restored.get());
  ASSERT_NE(id_map, nullptr);
  auto* hnsw = id_map->get_index<IndexHNSWFlat>();
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->hnsw.ef_search, 31);
  EXPECT_EQ(hnsw->hnsw.ef_construction, 77);

  float distance = -1.0F;
  idx_t label = -1;
  const float query[2] = {0.0F, 0.0F};
  restored->Search(1, query, 1, &distance, &label);
  EXPECT_EQ(label, 10);
  EXPECT_FLOAT_EQ(distance, 0.0F);
}

TEST(IndexPersistenceMetadata, ReadsLegacyPayloadAndMarksInferredMetadata) {
  IndexFileMetadata metadata;
  std::unique_ptr<Index> restored =
      ReadFromBytes(MakeLegacyFlatL2File(), &metadata);

  ASSERT_NE(dynamic_cast<IndexFlatL2*>(restored.get()), nullptr);
  EXPECT_EQ(metadata.format_major, 1);
  EXPECT_EQ(metadata.format_minor, 0);
  EXPECT_EQ(metadata.algorithm_version, 0);
  EXPECT_TRUE(metadata.is_legacy);
  EXPECT_EQ(metadata.config.type, "flat");
  EXPECT_EQ(metadata.config.dimension, 2);
  EXPECT_EQ(metadata.config.metric, kMetricL2);

  float distance = -1.0F;
  idx_t label = -1;
  const float query[2] = {1.0F, 1.0F};
  restored->Search(1, query, 1, &distance, &label);
  EXPECT_EQ(label, 1);
  EXPECT_FLOAT_EQ(distance, 0.0F);
}

TEST(IndexPersistenceMetadata, RejectsUnsupportedOrCorruptEnvelope) {
  IndexFlatL2 flat(2);
  const std::vector<uint8_t> valid = WriteToBytes(flat);

  std::vector<uint8_t> bad_major = valid;
  const uint16_t unsupported_major = kIndexFileFormatMajor + 1;
  std::memcpy(bad_major.data() + sizeof(uint32_t), &unsupported_major,
              sizeof(unsupported_major));
  EXPECT_THROW(ReadFromBytes(std::move(bad_major)), HypervecException);

  std::vector<uint8_t> bad_algorithm = valid;
  const uint32_t unsupported_algorithm = kIndexAlgorithmVersion + 1;
  constexpr size_t algorithm_offset = sizeof(uint32_t) + 2 * sizeof(uint16_t);
  std::memcpy(bad_algorithm.data() + algorithm_offset, &unsupported_algorithm,
              sizeof(unsupported_algorithm));
  EXPECT_THROW(ReadFromBytes(std::move(bad_algorithm)), HypervecException);

  std::vector<uint8_t> bad_checksum = valid;
  constexpr size_t config_offset = sizeof(uint32_t) + 2 * sizeof(uint16_t) +
                                   sizeof(uint32_t) + sizeof(uint64_t);
  bad_checksum[config_offset] ^= 1U;
  EXPECT_THROW(ReadFromBytes(std::move(bad_checksum)), HypervecException);
}

TEST(IndexPersistenceMetadata, RejectsMetadataPayloadMismatch) {
  IndexFlatL2 flat(2);
  IndexHNSWFlat hnsw(2, 2);
  const std::vector<uint8_t> flat_bytes = WriteToBytes(flat);
  const std::vector<uint8_t> hnsw_bytes = WriteToBytes(hnsw);
  const size_t flat_envelope = EnvelopeSize(flat_bytes);
  const size_t hnsw_envelope = EnvelopeSize(hnsw_bytes);

  std::vector<uint8_t> mixed(flat_bytes.begin(),
                             flat_bytes.begin() + flat_envelope);
  mixed.insert(mixed.end(), hnsw_bytes.begin() + hnsw_envelope,
               hnsw_bytes.end());
  EXPECT_THROW(ReadFromBytes(std::move(mixed)), HypervecException);
}

TEST(IndexPersistenceMetadata, RejectsAmbiguousGenericHnswBeforeWriting) {
  IndexFlatL2 storage(2);
  IndexHNSW generic(&storage, 2);
  VectorIOWriter writer;

  EXPECT_THROW(WriteIndex(&generic, &writer), HypervecException);
  EXPECT_TRUE(writer.data.empty());
}

}  // namespace
}  // namespace hypervec
