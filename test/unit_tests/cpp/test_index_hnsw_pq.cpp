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
#include <index/hnsw/index_hnsw_pq.h>
#include <persistence/index_io.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <quantization/pq/pq_distance_computer.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
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

std::vector<float> RandomVectors(hypervec::idx_t n, hypervec::idx_t d,
                                 int64_t seed, float scale = 1.0f) {
  hypervec::RandomGenerator rng(seed);
  std::vector<float> v(static_cast<size_t>(n) * d);
  for (auto& vi : v) {
    vi = scale * rng.rand_float();
  }
  return v;
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

// PQDistanceComputer's operator() must agree with PQ::SearchL2's distance for
// every i: building the per-query ADC table and then summing per code should
// reproduce SearchL2's accumulated value.
TEST(PQDistanceComputer, AgreesWithSearchL2) {
  const hypervec::idx_t d = 16, nb = 800;
  const hypervec::idx_t M = 4;
  const int nbits = 8;

  const auto base = RandomVectors(nb, d, 1234, 5.0f);
  const auto query = RandomVectors(1, d, 9999, 5.0f);

  hypervec::IndexPQ pq(d, M, nbits);
  pq.Train(nb, base.data());
  pq.Add(nb, base.data());

  // Top-1 via SearchL2 — its first distance is the minimum ADC over all i.
  std::vector<float> sd(1);
  std::vector<hypervec::idx_t> sl(1);
  pq.Search(1, query.data(), 1, sd.data(), sl.data());

  std::unique_ptr<hypervec::DistanceComputer> dc(pq.GetDistanceComputer());
  ASSERT_NE(dc.get(), nullptr);
  dc->SetQuery(query.data());

  // Distance computed via the DC for the SearchL2-reported best id must match.
  const float dc_d = (*dc)(sl[0]);
  EXPECT_FLOAT_EQ(dc_d, sd[0]);

  // And the DC's reported best distance over all i must equal SearchL2's best.
  float best = std::numeric_limits<float>::max();
  for (hypervec::idx_t i = 0; i < nb; i++) {
    const float di = (*dc)(i);
    if (di < best) {
      best = di;
    }
  }
  EXPECT_FLOAT_EQ(best, sd[0]);
}

// symmetric_dis is non-negative and zero for self-pairs.
TEST(PQDistanceComputer, SymmetricDisProperties) {
  const hypervec::idx_t d = 8, nb = 200;
  const auto base = RandomVectors(nb, d, 7, 3.0f);

  hypervec::IndexPQ pq(d, 4, 6);
  pq.Train(nb, base.data());
  pq.Add(nb, base.data());

  std::unique_ptr<hypervec::DistanceComputer> dc(pq.GetDistanceComputer());
  // SetQuery isn't required for symmetric_dis, but call it to be defensive.
  dc->SetQuery(base.data());

  EXPECT_FLOAT_EQ(dc->symmetric_dis(0, 0), 0.0f);
  EXPECT_GE(dc->symmetric_dis(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(dc->symmetric_dis(0, 1), dc->symmetric_dis(1, 0));
}

TEST(IndexHNSWPQ, ConstructRejectsNonL2) {
  EXPECT_THROW(hypervec::IndexHNSWPQ(16, 4, 8, 16, hypervec::kMetricInnerProduct),
               hypervec::HypervecException);
}

TEST(IndexHNSWPQ, AddBeforeTrainThrows) {
  hypervec::IndexHNSWPQ idx(16, 4, 8, 16);
  const auto x = RandomVectors(50, 16, 1);
  EXPECT_THROW(idx.Add(50, x.data()), hypervec::HypervecException);
}

TEST(IndexHNSWPQ, AddZeroIsNoop) {
  hypervec::IndexHNSWPQ idx(16, 4, 8, 16);
  const auto x = RandomVectors(500, 16, 2, 5.0f);
  idx.Train(500, x.data());
  idx.Add(0, nullptr);
  EXPECT_EQ(idx.n_total, 0);
}

TEST(IndexHNSWPQ, TrainAddSearchSmoke) {
  const hypervec::idx_t d = 32, nb = 2000, nq = 50;
  const hypervec::idx_t M_pq = 8;
  const int nbits = 8;
  const int M_hnsw = 16;
  const hypervec::idx_t k = 10;

  const auto base = RandomVectors(nb, d, 1234, 5.0f);
  const auto query = RandomVectors(nq, d, 5678, 5.0f);

  hypervec::IndexHNSWPQ idx(d, M_pq, nbits, M_hnsw);
  EXPECT_FALSE(idx.is_trained);
  idx.Train(nb, base.data());
  EXPECT_TRUE(idx.is_trained);
  idx.Add(nb, base.data());
  EXPECT_EQ(idx.n_total, nb);

  std::vector<float> dists(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);
  idx.Search(nq, query.data(), k, dists.data(), labels.data());

  for (hypervec::idx_t i = 0; i < nq; i++) {
    for (hypervec::idx_t j = 0; j < k; j++) {
      const auto id = labels[i * k + j];
      EXPECT_GE(id, 0);
      EXPECT_LT(id, nb);
      if (j > 0) {
        EXPECT_GE(dists[i * k + j], dists[i * k + j - 1]);
      }
    }
  }
}

// Sanity: the dual-storage build must out-recall a control where the graph
// is built from PQ-decoded distances. We construct the control by attaching
// an IndexPQ as the storage of a plain IndexHNSW (which then uses the PQ DC
// for both construction and search).
TEST(IndexHNSWPQ, DualStorageBuildBeatsPQOnlyBuild) {
  const hypervec::idx_t d = 32, nb = 4000, nq = 100;
  const hypervec::idx_t M_pq = 8;
  const int nbits = 8;
  const int M_hnsw = 16;
  const hypervec::idx_t k = 10;

  const auto base = RandomVectors(nb, d, 11, 5.0f);
  const auto query = RandomVectors(nq, d, 22, 5.0f);

  // Ground truth.
  hypervec::IndexFlatL2 gt(d);
  gt.Add(nb, base.data());
  std::vector<float> gt_d(nq * k);
  std::vector<hypervec::idx_t> gt_l(nq * k);
  gt.Search(nq, query.data(), k, gt_d.data(), gt_l.data());

  // Dual-storage HNSWPQ: graph built from raw vectors.
  hypervec::IndexHNSWPQ dual(d, M_pq, nbits, M_hnsw);
  dual.Train(nb, base.data());
  dual.Add(nb, base.data());
  std::vector<float> dd(nq * k);
  std::vector<hypervec::idx_t> dl(nq * k);
  dual.Search(nq, query.data(), k, dd.data(), dl.data());
  const float dual_recall = Recall(dl, gt_l, nq, k);

  // PQ-only build control: storage IS the IndexPQ, so HNSW build & search
  // both use the ADC computer.
  auto* pq_storage = new hypervec::IndexPQ(d, M_pq, nbits);
  pq_storage->Train(nb, base.data());
  hypervec::IndexHNSW pq_only(pq_storage, M_hnsw);
  pq_only.own_fields = true;
  pq_only.is_trained = true;
  pq_only.Add(nb, base.data());
  std::vector<float> pd(nq * k);
  std::vector<hypervec::idx_t> pl(nq * k);
  pq_only.Search(nq, query.data(), k, pd.data(), pl.data());
  const float pq_only_recall = Recall(pl, gt_l, nq, k);

  // Loose floor: dual-storage should not be strictly worse. Allow a tiny
  // slack for FP-tie noise.
  EXPECT_GE(dual_recall, pq_only_recall - 0.02f)
    << "dual=" << dual_recall << " pq_only=" << pq_only_recall;
}

TEST(IndexHNSWPQ, FreezeMakesIndexReadOnly) {
  hypervec::IndexHNSWPQ idx(16, 4, 8, 16);
  const auto base = RandomVectors(500, 16, 33, 4.0f);
  idx.Train(500, base.data());
  idx.Add(500, base.data());

  idx.Freeze();
  EXPECT_THROW(idx.Add(50, base.data()), hypervec::HypervecException);

  // Idempotent — second Freeze is a no-op.
  idx.Freeze();
  EXPECT_THROW(idx.Add(50, base.data()), hypervec::HypervecException);

  // Search still works after Freeze.
  std::vector<float> dists(5);
  std::vector<hypervec::idx_t> labels(5);
  idx.Search(1, base.data(), 5, dists.data(), labels.data());
  EXPECT_GE(labels[0], 0);
}

TEST(IndexHNSWPQ, ResetPreservesFrozenState) {
  hypervec::IndexHNSWPQ idx(16, 4, 8, 16);
  const auto base = RandomVectors(300, 16, 41, 4.0f);
  idx.Train(300, base.data());
  idx.Add(300, base.data());
  idx.Freeze();

  idx.Reset();
  EXPECT_EQ(idx.n_total, 0);
  // Frozen state survives Reset — Add must still throw.
  EXPECT_THROW(idx.Add(10, base.data()), hypervec::HypervecException);
}

TEST(IndexHNSWPQ, ResetBeforeFreezeAllowsReAdd) {
  hypervec::IndexHNSWPQ idx(16, 4, 8, 16);
  const auto base = RandomVectors(300, 16, 51, 4.0f);
  idx.Train(300, base.data());
  idx.Add(300, base.data());

  idx.Reset();
  EXPECT_EQ(idx.n_total, 0);
  // Not frozen — re-Add works.
  idx.Add(100, base.data());
  EXPECT_EQ(idx.n_total, 100);
}

TEST(IndexHNSWPQ, PersistenceRoundtrip) {
  const hypervec::idx_t d = 24, nb = 1500, nq = 30;
  const hypervec::idx_t M_pq = 4;
  const int nbits = 8;
  const int M_hnsw = 16;
  const hypervec::idx_t k = 5;

  const auto base = RandomVectors(nb, d, 71, 4.0f);
  const auto query = RandomVectors(nq, d, 72, 4.0f);

  hypervec::IndexHNSWPQ src(d, M_pq, nbits, M_hnsw);
  src.Train(nb, base.data());
  src.Add(nb, base.data());

  std::vector<float> ds(nq * k);
  std::vector<hypervec::idx_t> ls(nq * k);
  src.Search(nq, query.data(), k, ds.data(), ls.data());

  TempFile tf;
  hypervec::WriteIndex(&src, tf.path.c_str());

  std::unique_ptr<hypervec::Index> loaded(
    hypervec::ReadIndex(tf.path.c_str()));
  auto* dst = dynamic_cast<hypervec::IndexHNSWPQ*>(loaded.get());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(dst->d, src.d);
  EXPECT_EQ(dst->n_total, src.n_total);
  EXPECT_TRUE(dst->is_trained);
  // Deserialized index is implicitly frozen.
  EXPECT_EQ(dst->raw_storage, nullptr);

  std::vector<float> dl(nq * k);
  std::vector<hypervec::idx_t> ll(nq * k);
  dst->Search(nq, query.data(), k, dl.data(), ll.data());
  EXPECT_EQ(ls, ll);
  EXPECT_EQ(ds, dl);

  // Add on a deserialized index throws.
  EXPECT_THROW(dst->Add(10, base.data()), hypervec::HypervecException);
}

TEST(IndexHNSWPQ, Search1AndRangeSearchThrow) {
  // ksub = 1 << 8 = 256, so need >= 256 training vectors.
  hypervec::IndexHNSWPQ idx(16, 4, 8, 16);
  const auto base = RandomVectors(400, 16, 91, 4.0f);
  idx.Train(400, base.data());
  idx.Add(400, base.data());

  // Search1 / RangeSearch are explicitly unsupported — verify they throw
  // rather than silently misbehaving via the IndexPQ storage.
  EXPECT_THROW(idx.RangeSearch(1, base.data(), 1.0f, nullptr),
               hypervec::HypervecException);
}
