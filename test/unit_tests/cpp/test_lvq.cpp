/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <index/hnsw/index_hnsw_lvq.h>
#include <persistence/index_io.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/lvq.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<float> RandomVectors(hypervec::idx_t n, hypervec::idx_t d,
                                 int64_t seed, float scale = 1.0f) {
  hypervec::RandomGenerator rng(seed);
  std::vector<float> v(static_cast<size_t>(n) * d);
  for (auto& vi : v) {
    vi = scale * rng.rand_float();
  }
  return v;
}

void ExpectSortedValid(const std::vector<float>& distances,
                       const std::vector<hypervec::idx_t>& labels,
                       hypervec::idx_t nq, hypervec::idx_t k,
                       hypervec::idx_t nb) {
  for (hypervec::idx_t i = 0; i < nq; i++) {
    for (hypervec::idx_t j = 0; j < k; j++) {
      EXPECT_GE(labels[i * k + j], 0);
      EXPECT_LT(labels[i * k + j], nb);
      if (j > 0) {
        EXPECT_GE(distances[i * k + j], distances[i * k + j - 1]);
      }
    }
  }
}

struct TempFile {
  std::string path;
  TempFile() {
    char buf[L_tmpnam];
    std::tmpnam(buf);
    path = buf;
  }
  ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST(LocalVectorQuantizer, TrainEncodeDecodeSmoke) {
  const hypervec::idx_t d = 8, n = 400;
  const auto x = RandomVectors(n, d, 11, 5.0f);

  hypervec::LocalVectorQuantizer lvq(d, 8, 4);
  lvq.Train(n, x.data());
  EXPECT_TRUE(lvq.is_trained);
  EXPECT_GT(lvq.code_size, 0);

  std::vector<uint8_t> code(lvq.code_size);
  std::vector<float> decoded(d);
  lvq.ComputeCode(x.data(), code.data());
  lvq.Decode(code.data(), decoded.data());
  for (float v : decoded) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(IndexLVQ, TrainAddSearchSmoke) {
  const hypervec::idx_t d = 12, nb = 800, nq = 20, k = 5;
  const auto base = RandomVectors(nb, d, 21, 4.0f);
  const auto query = RandomVectors(nq, d, 22, 4.0f);

  hypervec::IndexLVQ idx(d, 8, 5);
  idx.Train(nb, base.data());
  idx.Add(nb, base.data());
  EXPECT_EQ(idx.n_total, nb);

  std::vector<float> distances(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);
  idx.Search(nq, query.data(), k, distances.data(), labels.data());
  ExpectSortedValid(distances, labels, nq, k, nb);

  std::vector<float> recons(d);
  idx.Reconstruct(3, recons.data());
  for (float v : recons) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(IndexIVFLVQ, TrainAddSearchSmoke) {
  const hypervec::idx_t d = 12, nb = 1000, nq = 20, k = 5;
  const auto base = RandomVectors(nb, d, 31, 4.0f);
  const auto query = RandomVectors(nq, d, 32, 4.0f);

  hypervec::IndexIVFLVQ idx(d, 16, 8, 4);
  idx.nprobe = 4;
  idx.Train(nb, base.data());
  idx.Add(nb, base.data());
  EXPECT_EQ(idx.n_total, nb);

  std::vector<float> distances(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);
  idx.Search(nq, query.data(), k, distances.data(), labels.data());
  ExpectSortedValid(distances, labels, nq, k, nb);
}

TEST(IndexHNSWLVQ, TrainAddSearchSmoke) {
  const hypervec::idx_t d = 12, nb = 700, nq = 10, k = 5;
  const auto base = RandomVectors(nb, d, 41, 4.0f);
  const auto query = RandomVectors(nq, d, 42, 4.0f);

  hypervec::IndexHNSWLVQ idx(d, 8, 4, 16);
  idx.hnsw.ef_search = 32;
  idx.Train(nb, base.data());
  idx.Add(nb, base.data());
  EXPECT_EQ(idx.n_total, nb);

  std::vector<float> distances(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);
  idx.Search(nq, query.data(), k, distances.data(), labels.data());
  ExpectSortedValid(distances, labels, nq, k, nb);

  idx.Freeze();
  EXPECT_THROW(idx.Add(1, base.data()), hypervec::HypervecException);
}

TEST(IndexLVQ, PersistenceRoundtrip) {
  const hypervec::idx_t d = 10, nb = 500, nq = 8, k = 4;
  const auto base = RandomVectors(nb, d, 51, 4.0f);
  const auto query = RandomVectors(nq, d, 52, 4.0f);

  hypervec::IndexLVQ src(d, 8, 4);
  src.Train(nb, base.data());
  src.Add(nb, base.data());

  TempFile tf;
  hypervec::WriteIndex(&src, tf.path.c_str());
  std::unique_ptr<hypervec::Index> loaded(
    hypervec::ReadIndex(tf.path.c_str()));
  auto* dst = dynamic_cast<hypervec::IndexLVQ*>(loaded.get());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(dst->d, src.d);
  EXPECT_EQ(dst->n_total, src.n_total);
  EXPECT_EQ(dst->lvq.nlocal, src.lvq.nlocal);
  EXPECT_EQ(dst->lvq.nbits, src.lvq.nbits);
  EXPECT_EQ(dst->lvq.local_centroids, src.lvq.local_centroids);
  EXPECT_EQ(dst->lvq.residual_codebooks, src.lvq.residual_codebooks);

  std::vector<float> ds(static_cast<size_t>(nq) * k);
  std::vector<float> dl(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> ls(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> ll(static_cast<size_t>(nq) * k);
  src.Search(nq, query.data(), k, ds.data(), ls.data());
  dst->Search(nq, query.data(), k, dl.data(), ll.data());
  EXPECT_EQ(ds, dl);
  EXPECT_EQ(ls, ll);
}

TEST(LocalVectorQuantizer, StandaloneIORoundtrip) {
  const hypervec::idx_t d = 8, n = 400;
  const auto x = RandomVectors(n, d, 61, 4.0f);
  hypervec::LocalVectorQuantizer src(d, 8, 4);
  src.Train(n, x.data());

  TempFile tf;
  hypervec::write_LocalVectorQuantizer(&src, tf.path.c_str());
  auto dst = hypervec::read_LocalVectorQuantizer_up(tf.path.c_str());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(dst->d, src.d);
  EXPECT_EQ(dst->nlocal, src.nlocal);
  EXPECT_EQ(dst->nbits, src.nbits);
  EXPECT_EQ(dst->local_centroids, src.local_centroids);
  EXPECT_EQ(dst->residual_codebooks, src.residual_codebooks);
}
