/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <utils/distances/distance_computer.h>
#include <utils/selector/id_selector.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

class FailingFlatCodes final : public hypervec::IndexFlatCodes {
 public:
  FailingFlatCodes() : IndexFlatCodes(sizeof(float), 1) {}

  bool fail = false;

  void SaEncode(hypervec::idx_t n, const float* x,
                uint8_t* bytes) const override {
    if (fail) {
      throw std::runtime_error("encoding failed");
    }
    std::memcpy(bytes, x, static_cast<size_t>(n) * sizeof(float));
  }

  void SaDecode(hypervec::idx_t n, const uint8_t* bytes,
                float* x) const override {
    std::memcpy(x, bytes, static_cast<size_t>(n) * sizeof(float));
  }
};

void ExpectCached(hypervec::IndexFlatL2* index) {
  index->SyncL2Norms();
  ASSERT_EQ(index->cached_l2norms.size(), static_cast<size_t>(index->n_total));
}

}  // namespace

TEST(IndexFlatCorrectness, MutationsInvalidateCachedL2Norms) {
  constexpr hypervec::idx_t dimension = 2;
  const std::vector<float> database = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

  hypervec::IndexFlatL2 added(dimension);
  added.Add(2, database.data());
  ExpectCached(&added);
  added.Add(1, database.data() + 4);
  EXPECT_TRUE(added.cached_l2norms.empty());
  std::unique_ptr<hypervec::DistanceComputer> distance(
      added.GetDistanceComputer());
  distance->SetQuery(database.data() + 4);
  EXPECT_FLOAT_EQ((*distance)(2), 0.0F);

  ExpectCached(&added);
  const std::array<hypervec::idx_t, 3> permutation = {2, 0, 1};
  hypervec::IndexFlatCodes* flat_codes = &added;
  flat_codes->PermuteEntries(permutation.data());
  EXPECT_TRUE(added.cached_l2norms.empty());

  ExpectCached(&added);
  hypervec::IDSelectorRange remove_first(0, 1);
  EXPECT_EQ(added.RemoveIds(remove_first), 1U);
  EXPECT_TRUE(added.cached_l2norms.empty());

  ExpectCached(&added);
  std::array<uint8_t, sizeof(float) * dimension> encoded{};
  added.SaEncode(1, database.data(), encoded.data());
  added.AddSaCodes(1, encoded.data(), nullptr);
  EXPECT_TRUE(added.cached_l2norms.empty());

  hypervec::IndexFlatL2 source(dimension);
  source.Add(1, database.data());
  ExpectCached(&source);
  ExpectCached(&added);
  added.MergeFrom(source);
  EXPECT_TRUE(added.cached_l2norms.empty());
  EXPECT_TRUE(source.cached_l2norms.empty());
  EXPECT_EQ(source.n_total, 0);

  ExpectCached(&added);
  added.Reset();
  EXPECT_TRUE(added.cached_l2norms.empty());
  EXPECT_EQ(added.n_total, 0);
}

TEST(IndexFlatCorrectness, AddValidatesInputsWithoutChangingState) {
  hypervec::IndexFlatL2 index(2);
  const std::array<float, 2> vector = {1.0F, 2.0F};
  index.Add(1, vector.data());
  const std::vector<uint8_t> original_codes = index.codes.owned_data;

  EXPECT_THROW(index.Add(-1, vector.data()), hypervec::HypervecException);
  EXPECT_THROW(index.Add(1, nullptr), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(index.codes.owned_data, original_codes);

  index.Add(0, nullptr);
  index.n_total = (std::numeric_limits<hypervec::idx_t>::max)();
  EXPECT_THROW(index.Add(1, vector.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, (std::numeric_limits<hypervec::idx_t>::max)());
  EXPECT_EQ(index.codes.owned_data, original_codes);
  index.n_total = 1;
}

TEST(IndexFlatCorrectness, AddKeepsStateWhenEncodingFails) {
  FailingFlatCodes index;
  const float first = 1.0F;
  const float second = 2.0F;
  index.Add(1, &first);
  const std::vector<uint8_t> original_codes = index.codes.owned_data;
  index.fail = true;

  EXPECT_THROW(index.Add(1, &second), std::runtime_error);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(index.codes.owned_data, original_codes);
}

TEST(IndexFlatCorrectness, AddSaCodesValidatesAndSupportsAliasedInput) {
  hypervec::IndexFlatL2 index(2);
  const std::array<float, 2> vector = {1.0F, 2.0F};
  index.Add(1, vector.data());
  const std::vector<uint8_t> original_codes = index.codes.owned_data;

  EXPECT_THROW(index.AddSaCodes(-1, original_codes.data(), nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.AddSaCodes(1, nullptr, nullptr),
               hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(index.codes.owned_data, original_codes);

  index.AddSaCodes(1, index.codes.data(), nullptr);
  EXPECT_EQ(index.n_total, 2);
  ASSERT_EQ(index.codes.size(), original_codes.size() * 2);
  EXPECT_EQ(std::memcmp(index.codes.data(),
                        index.codes.data() + original_codes.size(),
                        original_codes.size()),
            0);
}

TEST(IndexFlatCorrectness, MappedStorageRejectsAppendWithoutAborting) {
  std::array<uint8_t, sizeof(float) * 2> mapped_bytes{};
  hypervec::IndexFlatL2 index(2);
  index.codes = hypervec::MaybeOwnedVector<uint8_t>::create_view(
      mapped_bytes.data(), mapped_bytes.size(), nullptr);
  index.n_total = 1;
  const std::array<float, 2> vector = {1.0F, 2.0F};

  EXPECT_THROW(index.Add(1, vector.data()), hypervec::HypervecException);
  EXPECT_THROW(index.AddSaCodes(1, mapped_bytes.data(), nullptr),
               hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_FALSE(index.codes.is_owned);
  EXPECT_EQ(index.codes.data(), mapped_bytes.data());
}
