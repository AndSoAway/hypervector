/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/ivf/index_ivf_flat.h>
#include <persistence/index_io.h>
#include <utils/common/range_search_result.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/random.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<float> RandomVectors(hypervec::idx_t n, hypervec::idx_t d,
                                 int64_t seed) {
  hypervec::RandomGenerator rng(seed);
  std::vector<float> vectors(static_cast<size_t>(n) * d);
  for (float& value : vectors) {
    value = rng.rand_float();
  }
  return vectors;
}

struct TempFile {
  std::string path;
  TempFile() {
    char buffer[L_tmpnam];
    std::tmpnam(buffer);
    path = buffer;
  }
  ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST(IndexIVFFlat, ScannerPreservesDistanceKnnAndReconstruction) {
  const std::vector<float> vectors = {
      0.0F, 0.0F, 1.0F, 0.0F, 3.0F, 0.0F,
  };
  const hypervec::idx_t ids[] = {100, 101, 102};
  const float query[] = {0.0F, 0.0F};

  hypervec::IndexIVFFlat index(2, 1, hypervec::kMetricL2);
  index.Train(3, vectors.data());
  index.AddWithIds(3, vectors.data(), ids);

  float distances[3];
  hypervec::idx_t labels[3];
  index.Search(1, query, 3, distances, labels);
  EXPECT_EQ((std::vector<hypervec::idx_t>(labels, labels + 3)),
            (std::vector<hypervec::idx_t>{100, 101, 102}));
  EXPECT_EQ((std::vector<float>(distances, distances + 3)),
            (std::vector<float>{0.0F, 1.0F, 9.0F}));

  float reconstructed[2];
  index.Reconstruct(101, reconstructed);
  EXPECT_FLOAT_EQ(reconstructed[0], 1.0F);
  EXPECT_FLOAT_EQ(reconstructed[1], 0.0F);
}

TEST(IndexIVFFlat, ScannerPreservesSimilarityKnnOrdering) {
  const std::vector<float> vectors = {
      1.0F, 0.0F, 2.0F, 0.0F, -1.0F, 0.0F,
  };
  const hypervec::idx_t ids[] = {200, 201, 202};
  const float query[] = {1.0F, 0.0F};

  hypervec::IndexIVFFlat index(2, 1, hypervec::kMetricInnerProduct);
  index.Train(3, vectors.data());
  index.AddWithIds(3, vectors.data(), ids);

  float distances[3];
  hypervec::idx_t labels[3];
  index.Search(1, query, 3, distances, labels);
  EXPECT_EQ((std::vector<hypervec::idx_t>(labels, labels + 3)),
            (std::vector<hypervec::idx_t>{201, 200, 202}));
  EXPECT_EQ((std::vector<float>(distances, distances + 3)),
            (std::vector<float>{2.0F, 1.0F, -1.0F}));
}

TEST(IndexIVFFlat, ScannerRangeSearchUsesMetricAndSelector) {
  const std::vector<float> vectors = {
      0.0F, 0.0F, 1.0F, 0.0F, 3.0F, 0.0F,
  };
  const hypervec::idx_t ids[] = {300, 301, 302};
  const float query[] = {0.0F, 0.0F};

  hypervec::IndexIVFFlat index(2, 1, hypervec::kMetricL2);
  index.Train(3, vectors.data());
  index.AddWithIds(3, vectors.data(), ids);

  hypervec::IDSelectorRange selector(301, 303);
  hypervec::IVFSearchParameters params;
  params.nprobe = 1;
  params.sel = &selector;
  hypervec::RangeSearchResult result(1);
  index.RangeSearch(1, query, 1.0F, &result, &params);

  ASSERT_EQ(result.lims[1] - result.lims[0], 1U);
  EXPECT_EQ(result.labels[0], 301);
  EXPECT_FLOAT_EQ(result.distances[0], 1.0F);
}

TEST(IndexIVFFlat, CommonPreassignedPathRejectsInvalidList) {
  const float vector[] = {0.0F, 0.0F};
  hypervec::IndexIVFFlat index(2, 1);
  index.Train(1, vector);
  index.Add(1, vector);

  const hypervec::idx_t invalid_list = 1;
  float distance;
  hypervec::idx_t label;
  EXPECT_THROW(index.SearchPreassigned(1, vector, 1, &invalid_list, nullptr,
                                       &distance, &label, 1, nullptr),
               hypervec::HypervecException);
}

TEST(IndexIVFFlat, PersistenceRoundtripPreservesSearch) {
  const hypervec::idx_t d = 8;
  const hypervec::idx_t nb = 512;
  const hypervec::idx_t nq = 12;
  const hypervec::idx_t k = 6;
  const hypervec::idx_t nlist = 16;
  const auto base = RandomVectors(nb, d, 101);
  const auto queries = RandomVectors(nq, d, 102);

  hypervec::IndexIVFFlat source(d, nlist);
  source.nprobe = 5;
  source.Train(nb, base.data());
  source.Add(nb, base.data());

  std::vector<float> source_distances(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> source_labels(static_cast<size_t>(nq) * k);
  source.Search(nq, queries.data(), k, source_distances.data(),
                source_labels.data());

  TempFile file;
  hypervec::WriteIndex(&source, file.path.c_str());
  std::unique_ptr<hypervec::Index> loaded(
      hypervec::ReadIndex(file.path.c_str()));
  auto* restored = dynamic_cast<hypervec::IndexIVFFlat*>(loaded.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->d, source.d);
  EXPECT_EQ(restored->n_total, source.n_total);
  EXPECT_EQ(restored->nlist, source.nlist);
  EXPECT_EQ(restored->nprobe, source.nprobe);
  EXPECT_EQ(restored->centroids, source.centroids);

  std::vector<float> restored_distances(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> restored_labels(static_cast<size_t>(nq) * k);
  restored->Search(nq, queries.data(), k, restored_distances.data(),
                   restored_labels.data());

  EXPECT_EQ(restored_distances, source_distances);
  EXPECT_EQ(restored_labels, source_labels);
}
