/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/hnsw.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <persistence/index_clone.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

class ThrowingAddIndex final : public hypervec::Index {
 public:
  ThrowingAddIndex() : Index(2, hypervec::kMetricL2) {}

  void Add(hypervec::idx_t, const float*) final {
    throw std::runtime_error("injected add failure");
  }

  void Search(hypervec::idx_t, const float*, hypervec::idx_t, float*,
              hypervec::idx_t*, const hypervec::SearchParameters*) const final {
  }

  void Reset() final { n_total = 0; }
};

class ThrowingRemoveIndex final : public hypervec::Index {
 public:
  ThrowingRemoveIndex() : Index(2, hypervec::kMetricL2) {}

  void Add(hypervec::idx_t n, const float*) final { n_total += n; }

  void Search(hypervec::idx_t, const float*, hypervec::idx_t, float*,
              hypervec::idx_t*, const hypervec::SearchParameters*) const final {
  }

  size_t RemoveIds(const hypervec::IDSelector&) final {
    throw std::runtime_error("injected remove failure");
  }

  void Reset() final { n_total = 0; }
};

class RecordingIndex final : public hypervec::Index {
 public:
  RecordingIndex() : Index(2, hypervec::kMetricL2) {}

  mutable int received_ef_search = -1;
  bool received_query_training = false;

  void Train(hypervec::idx_t, const float*, hypervec::idx_t,
             const float*) final {
    received_query_training = true;
    is_trained = true;
  }

  void Add(hypervec::idx_t n, const float*) final { n_total += n; }

  void Search(hypervec::idx_t n, const float*, hypervec::idx_t k,
              float* distances, hypervec::idx_t* labels,
              const hypervec::SearchParameters* params) const final {
    const auto* hnsw_params =
        dynamic_cast<const hypervec::SearchParametersHNSW*>(params);
    received_ef_search = hnsw_params == nullptr ? -1 : hnsw_params->ef_search;
    for (hypervec::idx_t i = 0; i < n * k; ++i) {
      distances[i] = std::numeric_limits<float>::infinity();
      labels[i] = -1;
    }
    if (n > 0 && k > 0 && n_total > 0 &&
        (params == nullptr || params->sel == nullptr ||
         params->sel->IsMember(0))) {
      distances[0] = 0.0f;
      labels[0] = 0;
    }
  }

  void Reset() final { n_total = 0; }
};

std::vector<float> TrainingVectors() {
  return {
      0.0f,  0.0f,  0.0f,  0.0f,  0.1f, 0.0f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,
      0.0f,  0.0f,  0.0f,  0.1f,  0.0f, 9.9f,  10.0f, 10.0f, 10.0f, 10.0f, 9.9f,
      10.0f, 10.0f, 10.0f, 10.0f, 9.9f, 10.0f, 10.0f, 10.0f, 10.0f, 9.9f,
  };
}

}  // namespace

TEST(IndexIDMapCorrectness, ForwardsBothTrainingOverloads) {
  hypervec::IndexIVFFlat ivf(4, 2);
  hypervec::IndexIDMap mapped_ivf(&ivf);
  const auto training = TrainingVectors();

  mapped_ivf.Train(8, training.data());
  EXPECT_TRUE(ivf.is_trained);
  EXPECT_TRUE(mapped_ivf.is_trained);

  RecordingIndex recording;
  recording.is_trained = false;
  hypervec::IndexIDMap mapped_recording(&recording);
  const std::vector<float> query_training(4, 0.0f);
  mapped_recording.Train(8, training.data(), 2, query_training.data());
  EXPECT_TRUE(recording.received_query_training);
  EXPECT_TRUE(mapped_recording.is_trained);
}

TEST(IndexIDMapCorrectness, TranslatesExplicitIdsForSearchRangeAndReconstruct) {
  hypervec::IndexFlatL2 flat(2);
  hypervec::IndexIDMap index(&flat);
  const std::vector<float> vectors = {0.0f, 0.0f, 10.0f, 10.0f, 5.0f, 5.0f};
  const std::vector<hypervec::idx_t> ids = {10086, 90001, 42};
  index.AddWithIds(3, vectors.data(), ids.data());
  index.check_consistency();

  const std::vector<float> query = {10.0f, 10.0f};
  float distance = -1.0f;
  hypervec::idx_t label = -1;
  index.Search(1, query.data(), 1, &distance, &label);
  EXPECT_EQ(label, 90001);

  std::vector<float> reconstructed(2);
  index.Reconstruct(42, reconstructed.data());
  EXPECT_EQ(reconstructed, (std::vector<float>{5.0f, 5.0f}));

  hypervec::RangeSearchResult range_result(1);
  index.RangeSearch(1, query.data(), 0.5f, &range_result);
  ASSERT_EQ(range_result.lims[1], 1);
  EXPECT_EQ(range_result.labels[0], 90001);
}

TEST(IndexIDMapCorrectness, TranslatesExternalSelectors) {
  hypervec::IndexFlatL2 flat(2);
  hypervec::IndexIDMap index(&flat);
  const std::vector<float> vectors = {0.0f, 0.0f, 10.0f, 10.0f};
  const std::vector<hypervec::idx_t> ids = {10086, 90001};
  index.AddWithIds(2, vectors.data(), ids.data());

  const hypervec::idx_t allowed_id = 90001;
  hypervec::IDSelectorBatch selector(1, &allowed_id);
  hypervec::SearchParameters params;
  params.sel = &selector;
  const std::vector<float> query = {0.0f, 0.0f};
  float distance = -1.0f;
  hypervec::idx_t label = -1;
  index.Search(1, query.data(), 1, &distance, &label, &params);
  EXPECT_EQ(label, allowed_id);

  hypervec::RangeSearchResult range_result(1);
  index.RangeSearch(1, query.data(), 1000.0f, &range_result, &params);
  ASSERT_EQ(range_result.lims[1], 1);
  EXPECT_EQ(range_result.labels[0], allowed_id);
}

TEST(IndexIDMapCorrectness, PreservesHnswParametersWhileTranslatingSelector) {
  RecordingIndex recording;
  hypervec::IndexIDMap index(&recording);
  const std::vector<float> vector = {0.0f, 0.0f};
  const hypervec::idx_t external_id = 12345;
  index.AddWithIds(1, vector.data(), &external_id);

  hypervec::IDSelectorRange selector(external_id, external_id + 1);
  hypervec::SearchParametersHNSW params;
  params.ef_search = 77;
  params.sel = &selector;
  float distance = -1.0f;
  hypervec::idx_t label = -1;
  index.Search(1, vector.data(), 1, &distance, &label, &params);

  EXPECT_EQ(recording.received_ef_search, 77);
  EXPECT_EQ(label, external_id);
}

TEST(IndexIDMapCorrectness, FailedAddDoesNotPublishMappings) {
  ThrowingAddIndex throwing;
  hypervec::IndexIDMap index(&throwing);
  const std::vector<float> vector = {0.0f, 0.0f};
  const hypervec::idx_t external_id = 77;

  EXPECT_THROW(index.AddWithIds(1, vector.data(), &external_id),
               std::runtime_error);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(throwing.n_total, 0);
  EXPECT_TRUE(index.id_map.empty());
  EXPECT_TRUE(index.rev_map.empty());
}

TEST(IndexIDMapCorrectness, RejectsInvalidIdsBeforeAdding) {
  hypervec::IndexFlatL2 flat(2);
  hypervec::IndexIDMap index(&flat);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const std::vector<hypervec::idx_t> duplicate_ids = {7, 7};
  EXPECT_THROW(index.AddWithIds(2, vectors.data(), duplicate_ids.data()),
               hypervec::HypervecException);

  const hypervec::idx_t negative_id = -2;
  EXPECT_THROW(index.AddWithIds(1, vectors.data(), &negative_id),
               hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(flat.n_total, 0);
}

TEST(IndexIDMapCorrectness, WrapsExistingEntriesWithIdentityIds) {
  hypervec::IndexFlatL2 flat(2);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  flat.Add(2, vectors.data());

  hypervec::IndexIDMap index(&flat);
  index.check_consistency();
  EXPECT_EQ(index.to_internal(0), 0);
  EXPECT_EQ(index.to_internal(1), 1);
  EXPECT_EQ(index.from_internal(0), 0);
  EXPECT_EQ(index.from_internal(1), 1);
}

TEST(IndexIDMapCorrectness, RemovesExternalIdsAndCompactsInternalMappings) {
  hypervec::IndexFlatL2 flat(2);
  hypervec::IndexIDMap index(&flat);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f,
                                      2.0f, 2.0f, 3.0f, 3.0f};
  const std::vector<hypervec::idx_t> ids = {10, 20, 30, 40};
  index.AddWithIds(4, vectors.data(), ids.data());

  const std::vector<hypervec::idx_t> removed_ids = {20, 40};
  hypervec::IDSelectorBatch selector(removed_ids.size(), removed_ids.data());
  EXPECT_EQ(index.RemoveIds(selector), 2);
  index.check_consistency();

  EXPECT_EQ(index.n_total, 2);
  EXPECT_EQ(flat.n_total, 2);
  EXPECT_EQ(index.to_internal(10), 0);
  EXPECT_EQ(index.to_internal(30), 1);
  EXPECT_EQ(index.to_internal(20), -1);
  EXPECT_EQ(index.from_internal(1), 30);

  const std::vector<float> query = {2.0f, 2.0f};
  float distance = -1.0f;
  hypervec::idx_t label = -1;
  index.Search(1, query.data(), 1, &distance, &label);
  EXPECT_EQ(label, 30);
  std::vector<float> reconstructed(2);
  index.Reconstruct(30, reconstructed.data());
  EXPECT_EQ(reconstructed, query);
}

TEST(IndexIDMapCorrectness, FailedRemoveDoesNotPublishCompactedMappings) {
  ThrowingRemoveIndex throwing;
  hypervec::IndexIDMap index(&throwing);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const std::vector<hypervec::idx_t> ids = {10, 20};
  index.AddWithIds(2, vectors.data(), ids.data());
  hypervec::IDSelectorRange selector(10, 11);

  EXPECT_THROW(index.RemoveIds(selector), std::runtime_error);
  EXPECT_EQ(index.n_total, 2);
  EXPECT_EQ(index.to_internal(10), 0);
  EXPECT_EQ(index.to_internal(20), 1);
  index.check_consistency();
}

TEST(IndexIDMapCorrectness, MergesMappingsAndShiftsOnlyExternalIds) {
  hypervec::IndexFlatL2 destination_flat(2);
  hypervec::IndexFlatL2 source_flat(2);
  hypervec::IndexIDMap destination(&destination_flat);
  hypervec::IndexIDMap source(&source_flat);
  const std::vector<float> destination_vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const std::vector<float> source_vectors = {10.0f, 10.0f, 11.0f, 11.0f};
  const std::vector<hypervec::idx_t> destination_ids = {10, 20};
  const std::vector<hypervec::idx_t> source_ids = {30, 40};
  destination.AddWithIds(2, destination_vectors.data(), destination_ids.data());
  source.AddWithIds(2, source_vectors.data(), source_ids.data());

  destination.MergeFrom(source, 100);
  destination.check_consistency();
  source.check_consistency();

  EXPECT_EQ(destination.n_total, 4);
  EXPECT_EQ(destination.to_internal(130), 2);
  EXPECT_EQ(destination.to_internal(140), 3);
  EXPECT_EQ(source.n_total, 0);
  EXPECT_TRUE(source.id_map.empty());
  EXPECT_TRUE(source.rev_map.empty());

  float distance = -1.0f;
  hypervec::idx_t label = -1;
  destination.Search(1, source_vectors.data(), 1, &distance, &label);
  EXPECT_EQ(label, 130);
  std::vector<float> reconstructed(2);
  destination.Reconstruct(140, reconstructed.data());
  EXPECT_EQ(reconstructed, (std::vector<float>{11.0f, 11.0f}));
}

TEST(IndexIDMapCorrectness, RejectsMergeConflictsBeforeMutatingEitherIndex) {
  hypervec::IndexFlatL2 destination_flat(2);
  hypervec::IndexFlatL2 source_flat(2);
  hypervec::IndexIDMap destination(&destination_flat);
  hypervec::IndexIDMap source(&source_flat);
  const std::vector<float> vector = {1.0f, 1.0f};
  const hypervec::idx_t destination_id = 130;
  const hypervec::idx_t source_id = 30;
  destination.AddWithIds(1, vector.data(), &destination_id);
  source.AddWithIds(1, vector.data(), &source_id);

  EXPECT_THROW(destination.MergeFrom(source, 100), hypervec::HypervecException);
  EXPECT_EQ(destination.n_total, 1);
  EXPECT_EQ(source.n_total, 1);
  EXPECT_EQ(destination.to_internal(destination_id), 0);
  EXPECT_EQ(source.to_internal(source_id), 0);
  destination.check_consistency();
  source.check_consistency();
}

TEST(IndexIDMapCorrectness, RebuildsAndValidatesReverseMappings) {
  hypervec::IndexFlatL2 flat(2);
  hypervec::IndexIDMap index(&flat);
  const std::vector<float> vectors = {0.0f, 0.0f, 1.0f, 1.0f};
  const std::vector<hypervec::idx_t> ids = {10, 20};
  index.AddWithIds(2, vectors.data(), ids.data());

  index.rev_map.clear();
  index.maintain_rev_map = false;
  index.construct_rev_map();
  EXPECT_TRUE(index.maintain_rev_map);
  EXPECT_EQ(index.rev_map, ids);
  index.check_consistency();

  index.id_map[10] = 2;
  const auto previous_rev_map = index.rev_map;
  EXPECT_THROW(index.construct_rev_map(), hypervec::HypervecException);
  EXPECT_EQ(index.rev_map, previous_rev_map);
}

TEST(IndexIDMapCorrectness, CloneOwnsDistinctStorage) {
  hypervec::IndexFlatL2 storage(2);
  hypervec::IndexIDMap source(&storage);

  std::unique_ptr<hypervec::Index> clone_base(hypervec::clone_index(&source));
  auto* clone = dynamic_cast<hypervec::IndexIDMap*>(clone_base.get());
  ASSERT_NE(clone, nullptr);
  EXPECT_TRUE(clone->own_fields);
  EXPECT_NE(clone->index, source.index);
  clone->check_consistency();
}
