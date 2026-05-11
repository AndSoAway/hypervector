/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <index/flat/index_flat.h>
#include <persistence/index_io.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

float Recall(const std::vector<hypervec::idx_t>& result,
             const std::vector<hypervec::idx_t>& gt, int nq, int k) {
  int hits = 0;
  for (int i = 0; i < nq; i++) {
    for (int j = 0; j < k; j++) {
      const hypervec::idx_t want = gt[i * k + j];
      for (int l = 0; l < k; l++) {
        if (result[i * k + l] == want) {
          hits++;
          break;
        }
      }
    }
  }
  return static_cast<float>(hits) / static_cast<float>(nq * k);
}

// Generate `n` d-dim vectors uniformly in [0, scale). Reproducible from seed.
std::vector<float> RandomVectors(hypervec::idx_t n, hypervec::idx_t d,
                                 int64_t seed, float scale = 1.0f) {
  hypervec::RandomGenerator rng(seed);
  std::vector<float> v(static_cast<size_t>(n) * d);
  for (auto& vi : v) {
    vi = scale * rng.rand_float();
  }
  return v;
}

// Cross-platform tmpfile path with cleanup helper.
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

TEST(IndexPQ, TrainAddSearchSmoke) {
  const hypervec::idx_t d = 16, nb = 2000, nq = 50;
  const hypervec::idx_t M = 4;
  const int nbits = 8;
  const hypervec::idx_t k = 5;

  const auto base = RandomVectors(nb, d, 1234, 5.0f);
  const auto query = RandomVectors(nq, d, 5678, 5.0f);

  hypervec::IndexPQ idx(d, M, nbits);
  EXPECT_FALSE(idx.is_trained);
  idx.Train(nb, base.data());
  EXPECT_TRUE(idx.is_trained);
  idx.Add(nb, base.data());
  EXPECT_EQ(idx.n_total, nb);

  std::vector<float> dists(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);
  idx.Search(nq, query.data(), k, dists.data(), labels.data());

  // Every label must be a valid id and distances must be non-decreasing.
  for (hypervec::idx_t i = 0; i < nq; i++) {
    for (hypervec::idx_t j = 0; j < k; j++) {
      const auto id = labels[i * k + j];
      EXPECT_GE(id, 0);
      EXPECT_LT(id, nb);
      if (j > 0) {
        EXPECT_GE(dists[i * k + j], dists[i * k + j - 1])
          << "distances not sorted at i=" << i << " j=" << j;
      }
    }
  }
}

TEST(IndexPQ, RecallAgainstFlatL2) {
  const hypervec::idx_t d = 16, nb = 3000, nq = 100;
  const hypervec::idx_t M = 8;
  const int nbits = 8;
  const hypervec::idx_t k = 10;

  const auto base = RandomVectors(nb, d, 1, 5.0f);
  const auto query = RandomVectors(nq, d, 2, 5.0f);

  hypervec::IndexFlatL2 gt(d);
  gt.Add(nb, base.data());
  std::vector<float> gt_d(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> gt_l(static_cast<size_t>(nq) * k);
  gt.Search(nq, query.data(), k, gt_d.data(), gt_l.data());

  hypervec::IndexPQ pq(d, M, nbits);
  pq.Train(nb, base.data());
  pq.Add(nb, base.data());

  std::vector<float> pq_d(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> pq_l(static_cast<size_t>(nq) * k);
  pq.Search(nq, query.data(), k, pq_d.data(), pq_l.data());

  const float recall = Recall(pq_l, gt_l, nq, k);
  EXPECT_GT(recall, 0.5f) << "IndexPQ recall@" << k << " too low: " << recall;
}

TEST(IndexPQ, ReconstructAndSaCodec) {
  const hypervec::idx_t d = 8, nb = 500;
  const hypervec::idx_t M = 4;
  const int nbits = 6;

  const auto base = RandomVectors(nb, d, 99, 3.0f);
  hypervec::IndexPQ idx(d, M, nbits);
  idx.Train(nb, base.data());
  idx.Add(nb, base.data());

  // Reconstruct(key) must equal SaDecode(SaEncode(x_key)).
  std::vector<uint8_t> code(idx.SaCodeSize());
  std::vector<float> via_codec(d), via_recon(d);
  for (hypervec::idx_t i : {hypervec::idx_t(0), hypervec::idx_t(7),
                            hypervec::idx_t(123), hypervec::idx_t(499)}) {
    idx.SaEncode(1, base.data() + i * d, code.data());
    idx.SaDecode(1, code.data(), via_codec.data());
    idx.Reconstruct(i, via_recon.data());
    for (hypervec::idx_t j = 0; j < d; j++) {
      EXPECT_FLOAT_EQ(via_recon[j], via_codec[j])
        << "i=" << i << " j=" << j;
    }
  }

  EXPECT_THROW(idx.Reconstruct(-1, via_recon.data()),
               hypervec::HypervecException);
  EXPECT_THROW(idx.Reconstruct(nb, via_recon.data()),
               hypervec::HypervecException);
}

TEST(IndexPQ, RejectsNonL2Metric) {
  EXPECT_THROW(hypervec::IndexPQ(8, 4, 8, hypervec::kMetricInnerProduct),
               hypervec::HypervecException);
}

TEST(IndexPQ, IDSelectorParamThrows) {
  // T1 doesn't honour an IDSelector and rejects rather than ignoring it.
  hypervec::IndexPQ idx(8, 4, 4);
  const auto x = RandomVectors(200, 8, 3);
  idx.Train(200, x.data());
  idx.Add(200, x.data());

  hypervec::SearchParameters params;
  // Construct a dummy non-null sel pointer; we never dereference it because
  // the throw fires first.
  params.sel = reinterpret_cast<hypervec::IDSelector*>(0xdeadbeefULL);
  std::vector<float> d_(5);
  std::vector<hypervec::idx_t> l_(5);
  EXPECT_THROW(idx.Search(1, x.data(), 5, d_.data(), l_.data(), &params),
               hypervec::HypervecException);
}

TEST(IndexPQ, PersistenceRoundtrip) {
  const hypervec::idx_t d = 12, nb = 1000;
  const hypervec::idx_t M = 4;
  const int nbits = 7;

  const auto base = RandomVectors(nb, d, 17, 4.0f);

  hypervec::IndexPQ src(d, M, nbits);
  src.Train(nb, base.data());
  src.Add(nb, base.data());

  TempFile tf;
  hypervec::WriteIndex(&src, tf.path.c_str());

  std::unique_ptr<hypervec::Index> loaded(
    hypervec::ReadIndex(tf.path.c_str()));
  auto* dst = dynamic_cast<hypervec::IndexPQ*>(loaded.get());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(dst->d, src.d);
  EXPECT_EQ(dst->n_total, src.n_total);
  EXPECT_EQ(dst->pq.M, src.pq.M);
  EXPECT_EQ(dst->pq.nbits, src.pq.nbits);
  EXPECT_EQ(dst->pq.dsub, src.pq.dsub);
  EXPECT_EQ(dst->pq.ksub, src.pq.ksub);
  EXPECT_EQ(dst->pq.code_size, src.pq.code_size);
  EXPECT_TRUE(dst->is_trained);
  EXPECT_TRUE(dst->pq.is_trained);
  EXPECT_EQ(dst->pq.centroids, src.pq.centroids);

  // Re-search the loaded index with the same query — labels must match.
  const auto query = RandomVectors(20, d, 18);
  std::vector<float> ds(20 * 5), dl(20 * 5);
  std::vector<hypervec::idx_t> ls(20 * 5), ll(20 * 5);
  src.Search(20, query.data(), 5, ds.data(), ls.data());
  dst->Search(20, query.data(), 5, dl.data(), ll.data());
  EXPECT_EQ(ls, ll);
  EXPECT_EQ(ds, dl);
}

TEST(ProductQuantizerStandaloneIO, Roundtrip) {
  hypervec::ProductQuantizer src(8, 4, 5);
  const auto x = RandomVectors(500, 8, 31);
  src.Train(500, x.data());

  TempFile tf;
  hypervec::write_ProductQuantizer(&src, tf.path.c_str());

  auto dst = hypervec::read_ProductQuantizer_up(tf.path.c_str());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(dst->d, src.d);
  EXPECT_EQ(dst->M, src.M);
  EXPECT_EQ(dst->nbits, src.nbits);
  EXPECT_EQ(dst->dsub, src.dsub);
  EXPECT_EQ(dst->ksub, src.ksub);
  EXPECT_EQ(dst->code_size, src.code_size);
  EXPECT_EQ(dst->centroids, src.centroids);
  EXPECT_TRUE(dst->is_trained);

  // Encoding the same vector through both should yield identical codes.
  std::vector<uint8_t> code1(src.code_size), code2(src.code_size);
  src.ComputeCode(x.data(), code1.data());
  dst->ComputeCode(x.data(), code2.data());
  EXPECT_EQ(code1, code2);
}
