/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <index/flat/index_flat.h>
#include <index/kdtree/index_kdtree.h>
#include <persistence/index_io.h>
#include <utils/structures/random.h>

#include <cstdio>
#include <algorithm>
#include <cmath>
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

// Reference brute-force k-NN via IndexFlatL2.
void BruteForce(const std::vector<float>& base, hypervec::idx_t nb,
                const std::vector<float>& queries, hypervec::idx_t nq,
                hypervec::idx_t d, hypervec::idx_t k, std::vector<float>& dis,
                std::vector<hypervec::idx_t>& lab) {
  hypervec::IndexFlatL2 flat(d);
  flat.Add(nb, base.data());
  dis.resize(static_cast<size_t>(nq) * k);
  lab.resize(static_cast<size_t>(nq) * k);
  flat.Search(nq, queries.data(), k, dis.data(), lab.data());
}

}  // namespace

// Core guarantee: KD-tree returns exactly the same neighbours as brute force.
TEST(IndexKDTree, ExactMatchesBruteForce) {
  for (hypervec::idx_t d : {2, 8, 16}) {
    const hypervec::idx_t nb = 1000;
    const hypervec::idx_t nq = 20;
    const hypervec::idx_t k = 10;
    const auto base = RandomVectors(nb, d, 1000 + d);
    const auto queries = RandomVectors(nq, d, 2000 + d);

    std::vector<float> ref_dis;
    std::vector<hypervec::idx_t> ref_lab;
    BruteForce(base, nb, queries, nq, d, k, ref_dis, ref_lab);

    hypervec::IndexKDTree kd(d);
    kd.Add(nb, base.data());
    kd.BuildTree();

    std::vector<float> dis(static_cast<size_t>(nq) * k);
    std::vector<hypervec::idx_t> lab(static_cast<size_t>(nq) * k);
    kd.Search(nq, queries.data(), k, dis.data(), lab.data());

    // Exact k-NN: per query the returned neighbour *set* must match brute
    // force. We compare sets (sorted ids) rather than ordered rows: the two
    // code paths compute L2 with different summation orders (IndexFlat uses
    // SIMD-dispatched reductions, this index uses scalar fvec_L2sqr), so for
    // two points at near-identical distance the rounding can flip their order.
    // The sorted-distance vector must still agree within tolerance.
    for (hypervec::idx_t q = 0; q < nq; q++) {
      const size_t off = static_cast<size_t>(q) * k;
      std::vector<hypervec::idx_t> got(lab.begin() + off, lab.begin() + off + k);
      std::vector<hypervec::idx_t> want(ref_lab.begin() + off,
                                        ref_lab.begin() + off + k);
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      EXPECT_EQ(got, want) << "neighbour set differs at d=" << d
                           << ", query " << q;
      for (hypervec::idx_t j = 0; j < k; j++) {
        EXPECT_NEAR(dis[off + j], ref_dis[off + j],
                    1e-4f * (1.0f + std::abs(ref_dis[off + j])))
          << "distance differs at d=" << d << ", query " << q << ", slot " << j;
      }
    }
  }
}

// High dimension still returns exact results (even if pruning is weak).
TEST(IndexKDTree, ExactInHighDimension) {
  const hypervec::idx_t d = 128;
  const hypervec::idx_t nb = 500;
  const hypervec::idx_t nq = 8;
  const hypervec::idx_t k = 5;
  const auto base = RandomVectors(nb, d, 42);
  const auto queries = RandomVectors(nq, d, 43);

  std::vector<float> ref_dis;
  std::vector<hypervec::idx_t> ref_lab;
  BruteForce(base, nb, queries, nq, d, k, ref_dis, ref_lab);

  hypervec::IndexKDTree kd(d);
  kd.Add(nb, base.data());
  kd.BuildTree();

  std::vector<float> dis(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> lab(static_cast<size_t>(nq) * k);
  kd.Search(nq, queries.data(), k, dis.data(), lab.data());

  for (hypervec::idx_t q = 0; q < nq; q++) {
    const size_t off = static_cast<size_t>(q) * k;
    std::vector<hypervec::idx_t> got(lab.begin() + off, lab.begin() + off + k);
    std::vector<hypervec::idx_t> want(ref_lab.begin() + off,
                                      ref_lab.begin() + off + k);
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    EXPECT_EQ(got, want) << "neighbour set differs at query " << q;
    for (hypervec::idx_t j = 0; j < k; j++) {
      EXPECT_NEAR(dis[off + j], ref_dis[off + j],
                  1e-3f * (1.0f + std::abs(ref_dis[off + j])));
    }
  }
}

TEST(IndexKDTree, SearchRequiresBuiltTree) {
  const hypervec::idx_t d = 4;
  const hypervec::idx_t nb = 200;
  const hypervec::idx_t k = 3;
  const auto base = RandomVectors(nb, d, 7);
  const auto queries = RandomVectors(1, d, 8);

  hypervec::IndexKDTree kd(d);
  kd.Add(nb, base.data());

  std::vector<float> dis(k);
  std::vector<hypervec::idx_t> lab(k);
  // Searching before BuildTree() must throw (mirrors IndexFlat1D's
  // UpdatePermutation() contract).
  EXPECT_ANY_THROW(
    kd.Search(1, queries.data(), k, dis.data(), lab.data()));

  kd.BuildTree();
  kd.Search(1, queries.data(), k, dis.data(), lab.data());
  EXPECT_GE(lab[0], 0);
  EXPECT_LT(lab[0], nb);
}

// Fewer points than k: trailing slots padded with -1 / +inf.
TEST(IndexKDTree, FewerPointsThanK) {
  const hypervec::idx_t d = 3;
  const hypervec::idx_t nb = 4;
  const hypervec::idx_t k = 10;
  const auto base = RandomVectors(nb, d, 99);
  const auto queries = RandomVectors(1, d, 100);

  hypervec::IndexKDTree kd(d);
  kd.Add(nb, base.data());
  kd.BuildTree();

  std::vector<float> dis(k);
  std::vector<hypervec::idx_t> lab(k);
  kd.Search(1, queries.data(), k, dis.data(), lab.data());

  for (hypervec::idx_t j = 0; j < nb; j++) {
    EXPECT_GE(lab[j], 0);
  }
  for (hypervec::idx_t j = nb; j < k; j++) {
    EXPECT_EQ(lab[j], -1);
  }
}

TEST(IndexKDTree, EmptyIndexReturnsPadding) {
  const hypervec::idx_t d = 5;
  const hypervec::idx_t k = 4;
  const auto queries = RandomVectors(2, d, 11);

  hypervec::IndexKDTree kd(d);
  kd.BuildTree();

  std::vector<float> dis(static_cast<size_t>(2) * k);
  std::vector<hypervec::idx_t> lab(static_cast<size_t>(2) * k);
  kd.Search(2, queries.data(), k, dis.data(), lab.data());

  for (auto id : lab) {
    EXPECT_EQ(id, -1);
  }
}

TEST(IndexKDTree, PersistenceRoundtripPreservesSearch) {
  const hypervec::idx_t d = 8;
  const hypervec::idx_t nb = 512;
  const hypervec::idx_t nq = 12;
  const hypervec::idx_t k = 6;
  const auto base = RandomVectors(nb, d, 101);
  const auto queries = RandomVectors(nq, d, 102);

  hypervec::IndexKDTree source(d);
  source.Add(nb, base.data());
  source.BuildTree();

  std::vector<float> source_dis(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> source_lab(static_cast<size_t>(nq) * k);
  source.Search(nq, queries.data(), k, source_dis.data(), source_lab.data());

  TempFile file;
  hypervec::WriteIndex(&source, file.path.c_str());
  std::unique_ptr<hypervec::Index> loaded(
    hypervec::ReadIndex(file.path.c_str()));
  auto* restored = dynamic_cast<hypervec::IndexKDTree*>(loaded.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->d, source.d);
  EXPECT_EQ(restored->n_total, source.n_total);
  EXPECT_EQ(restored->leaf_size, source.leaf_size);

  std::vector<float> restored_dis(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> restored_lab(static_cast<size_t>(nq) * k);
  restored->Search(nq, queries.data(), k, restored_dis.data(),
                   restored_lab.data());

  EXPECT_EQ(restored_dis, source_dis);
  EXPECT_EQ(restored_lab, source_lab);
}
