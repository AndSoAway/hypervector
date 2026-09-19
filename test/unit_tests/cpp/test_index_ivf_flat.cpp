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
