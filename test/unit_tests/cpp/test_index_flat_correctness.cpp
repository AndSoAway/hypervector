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
#include <memory>
#include <vector>

namespace {

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
