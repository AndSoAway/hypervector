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
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/nsw/index_nsw.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

std::vector<float> MakeVectors(hypervec::idx_t n, hypervec::idx_t d) {
  std::vector<float> vectors(static_cast<size_t>(n * d));
  for (hypervec::idx_t i = 0; i < n; ++i) {
    for (hypervec::idx_t j = 0; j < d; ++j) {
      vectors[static_cast<size_t>(i * d + j)] =
          static_cast<float>((i * 17 + j * 7) % 31) / 5.0f;
    }
  }
  return vectors;
}

class ThrowingBuildIndex : public hypervec::IndexFlatL2 {
 public:
  explicit ThrowingBuildIndex(hypervec::idx_t d) : IndexFlatL2(d) {}

  void Build(const hypervec::DenseDatasetView & /*data*/) override {
    throw std::runtime_error("injected Build failure");
  }
};

class WrongCountBuildIndex : public hypervec::IndexFlatL2 {
 public:
  explicit WrongCountBuildIndex(hypervec::idx_t d) : IndexFlatL2(d) {}

  void Build(const hypervec::DenseDatasetView &data) override {
    n_total = data.count - 1;
    is_trained = false;
  }
};

class PartiallyAddingIndex : public hypervec::IndexFlatL2 {
 public:
  explicit PartiallyAddingIndex(hypervec::idx_t d) : IndexFlatL2(d) {}

  void Add(hypervec::idx_t n, const float *x) override {
    if (fail_next_add_) {
      fail_next_add_ = false;
      IndexFlatL2::Add(1, x);
      throw std::runtime_error("injected partial Add failure");
    }
    IndexFlatL2::Add(n, x);
  }

 private:
  bool fail_next_add_ = true;
};

void ExpectCapability(const hypervec::Index &index,
                      hypervec::IndexCapability capability, bool expected) {
  EXPECT_EQ(hypervec::HasCapability(index.Capabilities(), capability), expected)
      << "type=" << index.TypeName();
}

}  // namespace

TEST(IndexBuild, FlatBuildsAndDoesNotResetExistingData) {
  constexpr hypervec::idx_t d = 2;
  const std::vector<float> vectors = {0.0f, 0.0f, 2.0f, 2.0f, 4.0f, 4.0f};
  hypervec::IndexFlatL2 index(d);

  index.Build({3, d, vectors.data(), nullptr});
  EXPECT_EQ(index.n_total, 3);

  float distance = 0.0f;
  hypervec::idx_t label = -1;
  index.Search(1, vectors.data() + d, 1, &distance, &label);
  EXPECT_EQ(label, 1);

  EXPECT_THROW(index.Build({3, d, vectors.data(), nullptr}),
               hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 3);
}

TEST(IndexBuild, ValidatesDatasetViewBeforeMutation) {
  constexpr hypervec::idx_t d = 2;
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const hypervec::idx_t ids[] = {10, 20};

  hypervec::IndexFlatL2 negative_count(d);
  EXPECT_THROW(negative_count.Build({-1, d, vectors.data(), nullptr}),
               hypervec::HypervecException);

  hypervec::IndexFlatL2 wrong_dimension(d);
  EXPECT_THROW(wrong_dimension.Build({2, d + 1, vectors.data(), nullptr}),
               hypervec::HypervecException);

  hypervec::IndexFlatL2 negative_dimension(d);
  EXPECT_THROW(negative_dimension.Build({2, -1, vectors.data(), nullptr}),
               hypervec::HypervecException);

  hypervec::IndexFlatL2 null_vectors(d);
  EXPECT_THROW(null_vectors.Build({2, d, nullptr, nullptr}),
               hypervec::HypervecException);

  hypervec::IndexFlatL2 external_ids(d);
  EXPECT_THROW(external_ids.Build({2, d, vectors.data(), ids}),
               hypervec::HypervecException);

  hypervec::IndexFlatL2 empty(d);
  EXPECT_NO_THROW(empty.Build({0, d, nullptr, nullptr}));
  EXPECT_EQ(empty.n_total, 0);
}

TEST(IndexBuild, IDMapBuildPublishesExternalIdsAfterUnderlyingSuccess) {
  constexpr hypervec::idx_t d = 2;
  const std::vector<float> vectors = {0.0f, 0.0f, 2.0f, 2.0f, 4.0f, 4.0f};
  const hypervec::idx_t ids[] = {101, 205, 999};
  hypervec::IndexFlatL2 storage(d);
  hypervec::IndexIDMap index(&storage);

  index.Build({3, d, vectors.data(), ids});
  EXPECT_EQ(index.n_total, 3);
  EXPECT_EQ(storage.n_total, 3);
  EXPECT_EQ(index.to_internal(205), 1);
  EXPECT_EQ(index.from_internal(2), 999);
  EXPECT_NO_THROW(index.check_consistency());

  float distance = 0.0f;
  hypervec::idx_t label = -1;
  index.Search(1, vectors.data() + d, 1, &distance, &label);
  EXPECT_EQ(label, 205);
}

TEST(IndexBuild, IDMapRejectsInvalidIdsBeforeCallingUnderlyingBuild) {
  constexpr hypervec::idx_t d = 2;
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const hypervec::idx_t duplicate_ids[] = {7, 7};
  ThrowingBuildIndex storage(d);
  hypervec::IndexIDMap index(&storage);

  EXPECT_THROW(index.Build({2, d, vectors.data(), duplicate_ids}),
               hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_TRUE(index.id_map.empty());
  EXPECT_TRUE(index.rev_map.empty());

  const hypervec::idx_t unique_ids[] = {7, 8};
  EXPECT_THROW(index.Build({2, d, vectors.data(), unique_ids}),
               std::runtime_error);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(storage.n_total, 0);
  EXPECT_TRUE(index.id_map.empty());
  EXPECT_TRUE(index.rev_map.empty());
}

TEST(IndexBuild, DefaultBuildRollsBackAPartialAdd) {
  constexpr hypervec::idx_t d = 2;
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  PartiallyAddingIndex index(d);

  EXPECT_THROW(index.Build({2, d, vectors.data(), nullptr}),
               std::runtime_error);
  EXPECT_EQ(index.n_total, 0);

  EXPECT_NO_THROW(index.Build({2, d, vectors.data(), nullptr}));
  EXPECT_EQ(index.n_total, 2);
}

TEST(IndexBuild, IDMapDoesNotPublishIdsForUnexpectedUnderlyingCount) {
  constexpr hypervec::idx_t d = 2;
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const hypervec::idx_t ids[] = {7, 8};
  WrongCountBuildIndex storage(d);
  hypervec::IndexIDMap index(&storage);

  EXPECT_THROW(index.Build({2, d, vectors.data(), ids}),
               hypervec::HypervecException);
  EXPECT_FALSE(index.is_trained);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(storage.n_total, 0);
  EXPECT_TRUE(index.id_map.empty());
  EXPECT_TRUE(index.rev_map.empty());
}

TEST(IndexBuild, DefaultBuildWorksForEveryCurrentIndexFamily) {
  constexpr hypervec::idx_t d = 4;
  constexpr hypervec::idx_t n = 32;
  const std::vector<float> vectors = MakeVectors(n, d);
  const hypervec::DenseDatasetView data{n, d, vectors.data(), nullptr};

  std::vector<std::unique_ptr<hypervec::Index>> indexes;
  indexes.emplace_back(std::make_unique<hypervec::IndexFlatL2>(d));
  indexes.emplace_back(std::make_unique<hypervec::IndexPQ>(d, 2, 2));
  indexes.emplace_back(std::make_unique<hypervec::IndexLVQ>(d, 2, 2));
  indexes.emplace_back(std::make_unique<hypervec::IndexIVFFlat>(d, 2));
  indexes.emplace_back(std::make_unique<hypervec::IndexIVFPQ>(d, 2, 2, 2));
  indexes.emplace_back(std::make_unique<hypervec::IndexIVFLVQ>(d, 2, 2, 2));
  indexes.emplace_back(std::make_unique<hypervec::IndexHNSWFlat>(d, 4));
  indexes.emplace_back(std::make_unique<hypervec::IndexHNSWPQ>(d, 2, 2, 4));
  indexes.emplace_back(std::make_unique<hypervec::IndexHNSWLVQ>(d, 2, 2, 4));
  indexes.emplace_back(std::make_unique<hypervec::IndexNSWFlat>(d, 4));

  for (const auto &index : indexes) {
    ASSERT_TRUE(hypervec::HasCapability(index->Capabilities(),
                                        hypervec::IndexCapability::BatchBuild))
        << "type=" << index->TypeName();
    EXPECT_NO_THROW(index->Build(data)) << "type=" << index->TypeName();
    EXPECT_EQ(index->n_total, n) << "type=" << index->TypeName();

    if (hypervec::HasCapability(index->Capabilities(),
                                hypervec::IndexCapability::Filter)) {
      hypervec::IDSelectorRange only_five(5, 6);
      hypervec::SearchParameters params;
      params.sel = &only_five;
      float distance = 0.0f;
      hypervec::idx_t label = -1;
      index->Search(1, vectors.data() + 5 * d, 1, &distance, &label, &params);
      EXPECT_EQ(label, 5) << "type=" << index->TypeName();
    }
  }
}

TEST(IndexCapabilities, MatchImplementedBehavior) {
  hypervec::IndexFlatCodes flat_codes(sizeof(float) * 4, 4);
  EXPECT_EQ(flat_codes.Capabilities(), 0);

  hypervec::IndexFlatL2 flat(4);
  EXPECT_EQ(flat.TypeName(), std::string_view("flat"));
  ExpectCapability(flat, hypervec::IndexCapability::NeedsTrain, false);
  ExpectCapability(flat, hypervec::IndexCapability::RangeSearch, true);
  ExpectCapability(flat, hypervec::IndexCapability::Filter, true);
  ExpectCapability(flat, hypervec::IndexCapability::Serialize, true);
  ExpectCapability(flat, hypervec::IndexCapability::Clone, true);

  hypervec::IndexFlat flat_l1(4, hypervec::kMetricL1);
  ExpectCapability(flat_l1, hypervec::IndexCapability::KnnSearch, true);
  ExpectCapability(flat_l1, hypervec::IndexCapability::RangeSearch, false);

  hypervec::IndexFlat flat_lp(4, hypervec::kMetricLp);
  flat_lp.metric_arg = 3.0f;
  ExpectCapability(flat_lp, hypervec::IndexCapability::Clone, false);

  hypervec::IndexHNSWFlat hnsw_l1(4, 4, hypervec::kMetricL1);
  ExpectCapability(hnsw_l1, hypervec::IndexCapability::KnnSearch, true);
  ExpectCapability(hnsw_l1, hypervec::IndexCapability::RangeSearch, false);
  ExpectCapability(hnsw_l1, hypervec::IndexCapability::Serialize, false);

  hypervec::IndexHNSW generic_hnsw(&flat, 4);
  ExpectCapability(generic_hnsw, hypervec::IndexCapability::Serialize, true);
  ExpectCapability(generic_hnsw, hypervec::IndexCapability::Clone, false);

  hypervec::IndexNSWFlat nsw_flat(4, 4);
  ExpectCapability(nsw_flat, hypervec::IndexCapability::IncrementalAdd, true);
  ExpectCapability(nsw_flat, hypervec::IndexCapability::RangeSearch, false);
  ExpectCapability(nsw_flat, hypervec::IndexCapability::Filter, true);
  ExpectCapability(nsw_flat, hypervec::IndexCapability::Serialize, true);
  ExpectCapability(nsw_flat, hypervec::IndexCapability::Clone, false);

  hypervec::IndexPQ pq(4, 2, 2);
  EXPECT_EQ(pq.TypeName(), std::string_view("pq"));
  ExpectCapability(pq, hypervec::IndexCapability::NeedsTrain, true);
  ExpectCapability(pq, hypervec::IndexCapability::RangeSearch, false);
  ExpectCapability(pq, hypervec::IndexCapability::Filter, false);

  hypervec::IndexIVFPQ ivf_pq(4, 2, 2, 2);
  EXPECT_EQ(ivf_pq.TypeName(), std::string_view("ivf_pq"));
  ExpectCapability(ivf_pq, hypervec::IndexCapability::RangeSearch, true);
  ExpectCapability(ivf_pq, hypervec::IndexCapability::Filter, true);

  hypervec::IndexIVFLVQ ivf_lvq(4, 2, 2, 2);
  EXPECT_EQ(ivf_lvq.TypeName(), std::string_view("ivf_lvq"));
  ExpectCapability(ivf_lvq, hypervec::IndexCapability::RangeSearch, true);
  ExpectCapability(ivf_lvq, hypervec::IndexCapability::Filter, true);

  hypervec::IndexHNSWPQ hnsw_pq(4, 2, 2, 4);
  EXPECT_EQ(hnsw_pq.TypeName(), std::string_view("hnsw_pq"));
  ExpectCapability(hnsw_pq, hypervec::IndexCapability::IncrementalAdd, true);
  ExpectCapability(hnsw_pq, hypervec::IndexCapability::RangeSearch, false);
  ExpectCapability(hnsw_pq, hypervec::IndexCapability::Filter, true);
  ExpectCapability(hnsw_pq, hypervec::IndexCapability::Clone, false);
  hnsw_pq.Freeze();
  ExpectCapability(hnsw_pq, hypervec::IndexCapability::BatchBuild, false);
  ExpectCapability(hnsw_pq, hypervec::IndexCapability::IncrementalAdd, false);

  hypervec::IndexIDMap id_map(&flat);
  EXPECT_EQ(id_map.TypeName(), std::string_view("idmap"));
  ExpectCapability(id_map, hypervec::IndexCapability::ExternalIds, true);
  ExpectCapability(id_map, hypervec::IndexCapability::RangeSearch, true);
  ExpectCapability(id_map, hypervec::IndexCapability::Serialize, true);
  ExpectCapability(id_map, hypervec::IndexCapability::Clone, true);
}

TEST(IndexTypeName, IsStableForEveryCurrentConcreteFamily) {
  hypervec::IndexFlatCodes flat_codes(sizeof(float) * 4, 4);
  hypervec::IndexFlat flat(4);
  hypervec::IndexFlatIP flat_ip(4);
  hypervec::IndexFlatL2 flat_l2(4);
  hypervec::IndexFlat1D flat_1d;
  hypervec::IndexPQ pq(4, 2, 2);
  hypervec::IndexLVQ lvq(4, 2, 2);
  hypervec::IndexIVFFlat ivf_flat(4, 2);
  hypervec::IndexIVFPQ ivf_pq(4, 2, 2, 2);
  hypervec::IndexIVFLVQ ivf_lvq(4, 2, 2, 2);
  hypervec::IndexHNSW generic_hnsw(&flat_l2, 4);
  hypervec::IndexHNSWFlat hnsw_flat(4, 4);
  hypervec::IndexHNSWPQ hnsw_pq(4, 2, 2, 4);
  hypervec::IndexHNSWLVQ hnsw_lvq(4, 2, 2, 4);
  hypervec::IndexNSWFlat nsw_flat(4, 4);
  hypervec::IndexIDMap id_map(&flat_l2);

  EXPECT_EQ(flat_codes.TypeName(), std::string_view("flat_codes"));
  EXPECT_EQ(flat.TypeName(), std::string_view("flat"));
  EXPECT_EQ(flat_ip.TypeName(), std::string_view("flat"));
  EXPECT_EQ(flat_l2.TypeName(), std::string_view("flat"));
  EXPECT_EQ(flat_1d.TypeName(), std::string_view("flat_1d"));
  EXPECT_EQ(pq.TypeName(), std::string_view("pq"));
  EXPECT_EQ(lvq.TypeName(), std::string_view("lvq"));
  EXPECT_EQ(ivf_flat.TypeName(), std::string_view("ivf_flat"));
  EXPECT_EQ(ivf_pq.TypeName(), std::string_view("ivf_pq"));
  EXPECT_EQ(ivf_lvq.TypeName(), std::string_view("ivf_lvq"));
  EXPECT_EQ(generic_hnsw.TypeName(), std::string_view("hnsw"));
  EXPECT_EQ(hnsw_flat.TypeName(), std::string_view("hnsw_flat"));
  EXPECT_EQ(hnsw_pq.TypeName(), std::string_view("hnsw_pq"));
  EXPECT_EQ(hnsw_lvq.TypeName(), std::string_view("hnsw_lvq"));
  EXPECT_EQ(nsw_flat.TypeName(), std::string_view("nsw_flat"));
  EXPECT_EQ(id_map.TypeName(), std::string_view("idmap"));
}
