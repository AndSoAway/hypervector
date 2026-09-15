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
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <persistence/index_clone.h>
#include <persistence/index_io.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TempFile {
  TempFile() {
    char buffer[L_tmpnam];
    std::tmpnam(buffer);
    path = buffer;
  }
  ~TempFile() { std::remove(path.c_str()); }

  std::string path;
};

class ThrowingAddIndex : public hypervec::IndexFlatL2 {
 public:
  explicit ThrowingAddIndex(hypervec::idx_t d) : IndexFlatL2(d) {}

  void Add(hypervec::idx_t, const float*) override {
    throw std::runtime_error("injected add failure");
  }
};

class ThrowingResetIndex : public hypervec::IndexFlatL2 {
 public:
  explicit ThrowingResetIndex(hypervec::idx_t d) : IndexFlatL2(d) {}

  void Reset() override { throw std::runtime_error("injected reset failure"); }
};

class CountingIndex : public hypervec::IndexFlatL2 {
 public:
  CountingIndex(hypervec::idx_t d, int* destructions)
      : IndexFlatL2(d), destructions_(destructions) {}

  ~CountingIndex() override { ++*destructions_; }

 private:
  int* destructions_;
};

struct UnknownSearchParameters : hypervec::SearchParameters {};

}  // namespace

TEST(IndexIDMapCorrectness, TrainAndExplicitIdsAreForwardedAndTranslated) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexIVFFlat storage(d, 2);
  hypervec::IndexIDMap index(&storage);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f,  1.0f,
                                      9.0f, 9.0f, 10.0f, 10.0f};
  const std::vector<hypervec::idx_t> ids = {101, 205, 999, 1001};

  index.Train(4, vectors.data());
  EXPECT_TRUE(index.is_trained);
  index.AddWithIds(4, vectors.data(), ids.data());
  index.check_consistency();

  hypervec::IVFSearchParameters params;
  params.nprobe = 2;
  float distance;
  hypervec::idx_t label;
  index.Search(1, vectors.data() + 2 * d, 1, &distance, &label, &params);
  EXPECT_EQ(label, 999);
}

TEST(IndexIDMapCorrectness, SelectorUsesExternalIdsAndPreservesSubtype) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexHNSWFlat storage(d, 4);
  hypervec::IndexIDMap index(&storage);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f,
                                      2.0f, 2.0f, 3.0f, 3.0f};
  const std::vector<hypervec::idx_t> ids = {100, 200, 300, 400};
  index.AddWithIds(4, vectors.data(), ids.data());

  hypervec::IDSelectorRange selector(200, 201);
  hypervec::SearchParametersHNSW params;
  params.ef_search = 32;
  params.sel = &selector;
  std::vector<float> distances(3);
  std::vector<hypervec::idx_t> labels(3);
  index.Search(1, vectors.data(), 3, distances.data(), labels.data(), &params);

  EXPECT_EQ(labels[0], 200);
  EXPECT_EQ(labels[1], -1);
  EXPECT_EQ(labels[2], -1);
}

TEST(IndexIDMapCorrectness, RangeSearchTranslatesLabels) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexFlatL2 storage(d);
  hypervec::IndexIDMap index(&storage);
  const std::vector<float> vectors = {0.0f, 0.0f, 10.0f, 10.0f};
  const std::vector<hypervec::idx_t> ids = {501, 777};
  index.AddWithIds(2, vectors.data(), ids.data());

  hypervec::RangeSearchResult result(1);
  index.RangeSearch(1, vectors.data(), 0.1f, &result);
  ASSERT_EQ(result.lims[1], 1);
  EXPECT_EQ(result.labels[0], 501);
}

TEST(IndexIDMapCorrectness, FailedAddAndResetDoNotPublishMapChanges) {
  constexpr hypervec::idx_t d = 2;
  const std::vector<float> vector = {0.0f, 0.0f};
  const hypervec::idx_t id = 42;

  ThrowingAddIndex add_storage(d);
  hypervec::IndexIDMap add_index(&add_storage);
  EXPECT_THROW(add_index.AddWithIds(1, vector.data(), &id), std::runtime_error);
  EXPECT_EQ(add_index.n_total, 0);
  EXPECT_TRUE(add_index.id_map.empty());
  EXPECT_TRUE(add_index.rev_map.empty());

  ThrowingResetIndex reset_storage(d);
  hypervec::IndexIDMap reset_index(&reset_storage);
  reset_index.AddWithIds(1, vector.data(), &id);
  EXPECT_THROW(reset_index.Reset(), std::runtime_error);
  EXPECT_EQ(reset_index.n_total, 1);
  EXPECT_EQ(reset_index.to_internal(id), 0);
}

TEST(IndexIDMapCorrectness, DuplicateIdsAreRejectedBeforeUnderlyingAdd) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexFlatL2 storage(d);
  hypervec::IndexIDMap index(&storage);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const hypervec::idx_t id = 7;
  index.AddWithIds(1, vectors.data(), &id);

  EXPECT_THROW(index.AddWithIds(1, vectors.data() + d, &id),
               hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(storage.n_total, 1);
  index.check_consistency();
}

TEST(IndexIDMapCorrectness, OwnedUnderlyingIndexIsDestroyed) {
  int destructions = 0;
  {
    hypervec::IndexIDMap index(
        std::make_unique<CountingIndex>(2, &destructions));
    EXPECT_TRUE(index.OwnsIndex());
  }
  EXPECT_EQ(destructions, 1);
}

TEST(IndexIDMapCorrectness, BorrowedUnderlyingIndexIsNotDestroyed) {
  int destructions = 0;
  auto storage = std::make_unique<CountingIndex>(2, &destructions);
  {
    hypervec::IndexIDMap index(storage.get());
    EXPECT_FALSE(index.OwnsIndex());
  }
  EXPECT_EQ(destructions, 0);
  storage.reset();
  EXPECT_EQ(destructions, 1);
}

TEST(IndexIDMapCorrectness, UnknownFilteredParameterSubtypeIsRejected) {
  hypervec::IndexFlatL2 storage(2);
  hypervec::IndexIDMap index(&storage);
  const std::vector<float> vector = {0.0f, 0.0f};
  const hypervec::idx_t id = 10;
  index.AddWithIds(1, vector.data(), &id);

  hypervec::IDSelectorRange selector(10, 11);
  UnknownSearchParameters params;
  params.sel = &selector;
  float distance;
  hypervec::idx_t label;
  EXPECT_THROW(index.Search(1, vector.data(), 1, &distance, &label, &params),
               hypervec::HypervecException);
}

TEST(IndexIDMapCorrectness, PersistenceRoundtripPreservesMappingAndSearch) {
  constexpr hypervec::idx_t d = 2;
  hypervec::IndexFlatL2 storage(d);
  hypervec::IndexIDMap source(&storage);
  const std::vector<float> vectors = {0.0f, 0.0f, 2.0f, 2.0f, 4.0f, 4.0f};
  const std::vector<hypervec::idx_t> ids = {11, 55, 89};
  source.AddWithIds(3, vectors.data(), ids.data());

  TempFile file;
  hypervec::WriteIndex(&source, file.path.c_str());
  std::unique_ptr<hypervec::Index> loaded(
      hypervec::ReadIndex(file.path.c_str()));
  auto* restored = dynamic_cast<hypervec::IndexIDMap*>(loaded.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_TRUE(restored->OwnsIndex());
  restored->check_consistency();

  float distance;
  hypervec::idx_t label;
  restored->Search(1, vectors.data() + d, 1, &distance, &label);
  EXPECT_EQ(label, 55);
}

TEST(IndexIDMapCorrectness, CloneOwnsItsUnderlyingIndex) {
  hypervec::IndexFlatL2 storage(2);
  hypervec::IndexIDMap source(&storage);
  std::unique_ptr<hypervec::Index> cloned(hypervec::clone_index(&source));
  auto* id_map = dynamic_cast<hypervec::IndexIDMap*>(cloned.get());
  ASSERT_NE(id_map, nullptr);
  EXPECT_TRUE(id_map->OwnsIndex());
  EXPECT_NE(id_map->index, &storage);
}
