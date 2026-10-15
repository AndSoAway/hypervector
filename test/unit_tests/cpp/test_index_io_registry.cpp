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
#include <index/index_factory.h>
#include <persistence/index_io.h>
#include <persistence/index_io_registry.h>
#include <persistence/io.h>
#include <utils/log/assert.h>
#include <utils/log/exception.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>

namespace {

class TestIndex : public hypervec::Index {
 public:
  explicit TestIndex(int32_t value = 0) : value(value) {}

  void Add(hypervec::idx_t, const float*) override {}
  void Search(hypervec::idx_t, const float*, hypervec::idx_t, float*,
              hypervec::idx_t*,
              const hypervec::SearchParameters*) const override {}
  void Reset() override {}

  int32_t value;
};

class OtherIndex final : public TestIndex {};

constexpr char kCurrentTag[] = "Ts01";
constexpr char kLegacyTag[] = "Ts00";

hypervec::IndexPayloadWriter TestWriter() {
  return [](const hypervec::Index& base, hypervec::IOWriter* writer,
            int io_flags) {
    EXPECT_EQ(io_flags, 17);
    const auto& index = static_cast<const TestIndex&>(base);
    HYPERVEC_THROW_IF_NOT_MSG(
        (*writer)(&index.value, sizeof(index.value), 1) == 1,
        "test payload write failed");
  };
}

hypervec::IndexPayloadReader TestReader() {
  return [](hypervec::IOReader* reader, int io_flags) {
    EXPECT_EQ(io_flags, 19);
    int32_t value = 0;
    HYPERVEC_THROW_IF_NOT_MSG((*reader)(&value, sizeof(value), 1) == 1,
                              "test payload read failed");
    return std::make_unique<TestIndex>(value);
  };
}

void RegisterTestCodec(hypervec::IndexIORegistry* registry) {
  registry->Register(
      {"test", hypervec::fourcc(kCurrentTag), {hypervec::fourcc(kLegacyTag)}},
      std::type_index(typeid(TestIndex)), TestWriter(), TestReader());
}

}  // namespace

TEST(IndexIORegistry, RoundtripsCanonicalTagAndReadsLegacyAlias) {
  hypervec::IndexIORegistry registry;
  RegisterTestCodec(&registry);
  EXPECT_TRUE(registry.Contains(std::type_index(typeid(TestIndex))));
  EXPECT_TRUE(registry.Contains(hypervec::fourcc(kCurrentTag)));
  EXPECT_TRUE(registry.Contains(hypervec::fourcc(kLegacyTag)));

  hypervec::VectorIOWriter writer;
  registry.Write(TestIndex(42), &writer, 17);
  ASSERT_EQ(writer.data.size(), sizeof(uint32_t) + sizeof(int32_t));
  uint32_t written_tag = 0;
  std::memcpy(&written_tag, writer.data.data(), sizeof(written_tag));
  EXPECT_EQ(written_tag, hypervec::fourcc(kCurrentTag));

  hypervec::VectorIOReader current_reader;
  current_reader.data = writer.data;
  std::unique_ptr<hypervec::Index> current = registry.Read(&current_reader, 19);
  auto* current_test = dynamic_cast<TestIndex*>(current.get());
  ASSERT_NE(current_test, nullptr);
  EXPECT_EQ(current_test->value, 42);

  hypervec::VectorIOReader payload_reader;
  payload_reader.data = writer.data;
  payload_reader.rp = sizeof(uint32_t);
  std::unique_ptr<hypervec::Index> payload =
      registry.ReadPayload(hypervec::fourcc(kCurrentTag), &payload_reader, 19);
  auto* payload_test = dynamic_cast<TestIndex*>(payload.get());
  ASSERT_NE(payload_test, nullptr);
  EXPECT_EQ(payload_test->value, 42);

  const uint32_t legacy_tag = hypervec::fourcc(kLegacyTag);
  std::memcpy(writer.data.data(), &legacy_tag, sizeof(legacy_tag));
  hypervec::VectorIOReader legacy_reader;
  legacy_reader.data = writer.data;
  std::unique_ptr<hypervec::Index> legacy = registry.Read(&legacy_reader, 19);
  auto* legacy_test = dynamic_cast<TestIndex*>(legacy.get());
  ASSERT_NE(legacy_test, nullptr);
  EXPECT_EQ(legacy_test->value, 42);

  const auto descriptors = registry.List();
  ASSERT_EQ(descriptors.size(), 1);
  EXPECT_EQ(descriptors.front().name, "test");
  EXPECT_EQ(descriptors.front().write_tag, hypervec::fourcc(kCurrentTag));
  EXPECT_EQ(descriptors.front().read_tags.size(), 2);
}

TEST(IndexIORegistry, GlobalEntrypointsUseFixedCodeRegistry) {
  hypervec::IndexFlatL2 source(2);
  const float vectors[] = {1.0F, 2.0F, 3.0F, 4.0F};
  source.Add(2, vectors);

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  ASSERT_GE(writer.data.size(), sizeof(uint32_t));
  uint32_t tag = 0;
  std::memcpy(&tag, writer.data.data(), sizeof(tag));
  EXPECT_EQ(tag, hypervec::fourcc("IFlm"));

  const uint32_t legacy_tag = hypervec::fourcc("IFll");
  std::memcpy(writer.data.data(), &legacy_tag, sizeof(legacy_tag));
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> loaded = hypervec::ReadIndexUp(&reader);
  auto* flat = dynamic_cast<hypervec::IndexFlatL2*>(loaded.get());
  ASSERT_NE(flat, nullptr);
  ASSERT_EQ(flat->n_total, 2);
  float reconstructed[2] = {};
  flat->Reconstruct(1, reconstructed);
  EXPECT_FLOAT_EQ(reconstructed[0], 3.0F);
  EXPECT_FLOAT_EQ(reconstructed[1], 4.0F);
}

TEST(IndexIORegistry, RoundtripsGenericFlatMetricsWithDistinctTag) {
  hypervec::IndexConfig config("flat", 2, hypervec::kMetricLp);
  config.metric_arg = 3.0F;
  std::unique_ptr<hypervec::Index> source = hypervec::CreateIndex(config);
  const hypervec::Index* source_index = source.get();
  ASSERT_EQ(typeid(*source_index), typeid(hypervec::IndexFlat));
  const float vectors[] = {0.0F, 0.0F, 2.0F, 2.0F};
  source->Add(2, vectors);

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(source.get(), &writer);
  ASSERT_GE(writer.data.size(), sizeof(uint32_t));
  uint32_t tag = 0;
  std::memcpy(&tag, writer.data.data(), sizeof(tag));
  EXPECT_EQ(tag, hypervec::fourcc("IFlx"));

  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> loaded = hypervec::ReadIndexUp(&reader);
  const hypervec::Index* loaded_index = loaded.get();
  ASSERT_EQ(typeid(*loaded_index), typeid(hypervec::IndexFlat));
  EXPECT_EQ(loaded->metric_type, hypervec::kMetricLp);
  EXPECT_FLOAT_EQ(loaded->metric_arg, 3.0F);

  const float query[] = {0.0F, 1.0F};
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  loaded->Search(1, query, 1, &distance, &label);
  EXPECT_EQ(label, 0);
  EXPECT_FLOAT_EQ(distance, 1.0F);
}

TEST(IndexIORegistry, RoundtripsFactoryCreatedHnswWithGenericFlatStorage) {
  hypervec::IndexConfig config("hnsw_flat", 2, hypervec::kMetricLp);
  config.metric_arg = 3.0F;
  config.SetInteger("m_hnsw", 4).SetInteger("ef_construction", 8);
  std::unique_ptr<hypervec::Index> source = hypervec::CreateIndex(config);
  const float vectors[] = {0.0F, 0.0F, 2.0F, 2.0F, 4.0F, 4.0F};
  source->Add(3, vectors);

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(source.get(), &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> loaded = hypervec::ReadIndexUp(&reader);
  auto* hnsw = dynamic_cast<hypervec::IndexHNSWFlat*>(loaded.get());
  ASSERT_NE(hnsw, nullptr);
  ASSERT_NE(hnsw->storage, nullptr);
  EXPECT_EQ(typeid(*hnsw->storage), typeid(hypervec::IndexFlat));
  EXPECT_EQ(hnsw->metric_type, hypervec::kMetricLp);
  EXPECT_FLOAT_EQ(hnsw->metric_arg, 3.0F);
  EXPECT_FLOAT_EQ(hnsw->storage->metric_arg, 3.0F);

  const float query[] = {0.0F, 1.0F};
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  loaded->Search(1, query, 1, &distance, &label);
  EXPECT_EQ(label, 0);
  EXPECT_FLOAT_EQ(distance, 1.0F);
}

TEST(IndexIORegistry, GlobalEntrypointsSupportCustomRegistrations) {
  hypervec::GetIndexIORegistry().Register(
      {"test_global", hypervec::fourcc("Tg01"), {}},
      std::type_index(typeid(TestIndex)),
      [](const hypervec::Index& base, hypervec::IOWriter* writer, int) {
        const auto& index = static_cast<const TestIndex&>(base);
        HYPERVEC_THROW_IF_NOT_MSG(
            (*writer)(&index.value, sizeof(index.value), 1) == 1,
            "global test payload write failed");
      },
      [](hypervec::IOReader* reader, int) {
        int32_t value = 0;
        HYPERVEC_THROW_IF_NOT_MSG((*reader)(&value, sizeof(value), 1) == 1,
                                  "global test payload read failed");
        return std::make_unique<TestIndex>(value);
      });

  hypervec::VectorIOWriter writer;
  TestIndex source(73);
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> restored = hypervec::ReadIndexUp(&reader);
  auto* test = dynamic_cast<TestIndex*>(restored.get());
  ASSERT_NE(test, nullptr);
  EXPECT_EQ(test->value, 73);
}

TEST(IndexIORegistry, GlobalWriterRejectsUnregisteredDerivedTypes) {
  hypervec::IndexFlat1D derived;
  hypervec::VectorIOWriter writer;
  EXPECT_THROW(hypervec::WriteIndex(&derived, &writer),
               hypervec::HypervecException);
}

TEST(IndexIORegistry, GlobalWriterRejectsBareHnswWithoutWritingAFlatTag) {
  hypervec::IndexHNSW bare(2, 8);
  hypervec::VectorIOWriter writer;
  EXPECT_THROW(hypervec::WriteIndex(&bare, &writer),
               hypervec::HypervecException);
  EXPECT_TRUE(writer.data.empty());
}

TEST(IndexIORegistry, RejectsAmbiguousOrIncompleteRegistrations) {
  hypervec::IndexIORegistry registry;
  RegisterTestCodec(&registry);

  EXPECT_THROW(RegisterTestCodec(&registry), hypervec::HypervecException);
  EXPECT_THROW(registry.Register({"other", hypervec::fourcc("Ot01"), {}},
                                 std::type_index(typeid(TestIndex)),
                                 TestWriter(), TestReader()),
               hypervec::HypervecException);
  EXPECT_THROW(registry.Register({"other", hypervec::fourcc(kLegacyTag), {}},
                                 std::type_index(typeid(OtherIndex)),
                                 TestWriter(), TestReader()),
               hypervec::HypervecException);
  EXPECT_THROW(registry.Register({"", hypervec::fourcc("Ot01"), {}},
                                 std::type_index(typeid(OtherIndex)),
                                 TestWriter(), TestReader()),
               hypervec::HypervecException);
  EXPECT_THROW(
      registry.Register({"other", hypervec::fourcc("Ot01"), {}},
                        std::type_index(typeid(OtherIndex)), {}, TestReader()),
      hypervec::HypervecException);
}

TEST(IndexIORegistry, RejectsUnknownTypesTagsAndMismatchedReaders) {
  hypervec::IndexIORegistry registry;
  RegisterTestCodec(&registry);
  hypervec::VectorIOWriter writer;

  EXPECT_THROW(registry.Write(OtherIndex(), &writer),
               hypervec::HypervecException);
  EXPECT_THROW(registry.Write(TestIndex(), nullptr),
               hypervec::HypervecException);

  const uint32_t unknown_tag = hypervec::fourcc("Unkn");
  writer.data.resize(sizeof(unknown_tag));
  std::memcpy(writer.data.data(), &unknown_tag, sizeof(unknown_tag));
  hypervec::VectorIOReader unknown_reader;
  unknown_reader.data = writer.data;
  EXPECT_THROW(registry.Read(&unknown_reader), hypervec::HypervecException);
  EXPECT_THROW(registry.Read(nullptr), hypervec::HypervecException);

  hypervec::IndexIORegistry mismatched;
  mismatched.Register(
      {"mismatch", hypervec::fourcc("Mm01"), {}},
      std::type_index(typeid(TestIndex)), TestWriter(),
      [](hypervec::IOReader*, int) { return std::make_unique<OtherIndex>(); });
  hypervec::VectorIOWriter mismatched_writer;
  const uint32_t mismatch_tag = hypervec::fourcc("Mm01");
  mismatched_writer.data.resize(sizeof(mismatch_tag));
  std::memcpy(mismatched_writer.data.data(), &mismatch_tag,
              sizeof(mismatch_tag));
  hypervec::VectorIOReader mismatched_reader;
  mismatched_reader.data = mismatched_writer.data;
  EXPECT_THROW(mismatched.Read(&mismatched_reader),
               hypervec::HypervecException);
}

TEST(IndexIORegistry, ValidatesBeforeWritingTheFormatTag) {
  hypervec::IndexIORegistry registry;
  bool validated = false;
  bool payload_written = false;
  registry.Register(
      {"validated", hypervec::fourcc("Va01"), {}},
      std::type_index(typeid(TestIndex)),
      [&](const hypervec::Index&, hypervec::IOWriter*, int) {
        payload_written = true;
      },
      TestReader(),
      [&](const hypervec::Index&, int io_flags) {
        EXPECT_EQ(io_flags, 23);
        validated = true;
        HYPERVEC_THROW_MSG("validation failed");
      });

  hypervec::VectorIOWriter writer;
  EXPECT_THROW(registry.Write(TestIndex(), &writer, 23),
               hypervec::HypervecException);
  EXPECT_TRUE(validated);
  EXPECT_FALSE(payload_written);
  EXPECT_TRUE(writer.data.empty());
}
