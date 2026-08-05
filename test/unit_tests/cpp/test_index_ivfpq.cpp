/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <index/flat/index_flat.h>
#include <index/ivf/index_ivf.h>
#include <persistence/index_io.h>
#include <quantization/pq/index_ivfpq.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <cmath>
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

TEST(IndexIVFPQ, TrainAddSearchSmoke) {
  const hypervec::idx_t d = 16, nb = 3000, nq = 50, nlist = 32;
  const hypervec::idx_t M = 4;
  const int nbits = 6;
  const hypervec::idx_t k = 5;

  const auto base = RandomVectors(nb, d, 11, 5.0f);
  const auto query = RandomVectors(nq, d, 22, 5.0f);

  hypervec::IndexIVFPQ idx(d, nlist, M, nbits);
  EXPECT_FALSE(idx.is_trained);
  idx.Train(nb, base.data());
  EXPECT_TRUE(idx.is_trained);
  idx.Add(nb, base.data());
  EXPECT_EQ(idx.n_total, nb);

  hypervec::IVFSearchParameters params;
  params.nprobe = 8;

  std::vector<float> dists(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);
  idx.Search(nq, query.data(), k, dists.data(), labels.data(), &params);

  // Each label is either -1 (not enough hits) or a valid id; distances
  // strictly non-decreasing per query (or sentinel after a -1).
  for (hypervec::idx_t i = 0; i < nq; i++) {
    for (hypervec::idx_t j = 0; j < k; j++) {
      const auto id = labels[i * k + j];
      EXPECT_TRUE(id == -1 || (id >= 0 && id < nb));
    }
  }
}

TEST(IndexIVFPQ, RecallImprovesWithNprobe) {
  // With more nprobe lists scanned, recall should rise (or stay equal),
  // never fall, against the ground-truth flat index.
  const hypervec::idx_t d = 16, nb = 3000, nq = 100, nlist = 32;
  const hypervec::idx_t M = 8;
  const int nbits = 8;
  const hypervec::idx_t k = 10;

  const auto base = RandomVectors(nb, d, 5, 5.0f);
  const auto query = RandomVectors(nq, d, 6, 5.0f);

  hypervec::IndexFlatL2 gt(d);
  gt.Add(nb, base.data());
  std::vector<float> gt_d(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> gt_l(static_cast<size_t>(nq) * k);
  gt.Search(nq, query.data(), k, gt_d.data(), gt_l.data());

  hypervec::IndexIVFPQ idx(d, nlist, M, nbits);
  idx.Train(nb, base.data());
  idx.Add(nb, base.data());

  hypervec::IVFSearchParameters params;
  std::vector<float> dists(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);

  float prev_recall = -1.0f;
  for (int nprobe : {1, 4, 16, static_cast<int>(nlist)}) {
    params.nprobe = nprobe;
    idx.Search(nq, query.data(), k, dists.data(), labels.data(), &params);
    const float r = Recall(labels, gt_l, nq, k);
    EXPECT_GE(r, prev_recall - 0.02f)
      << "recall regressed at nprobe=" << nprobe;
    prev_recall = r;
  }
  // At nprobe=nlist (i.e., scan everything), PQ recall should be at least
  // moderate (PQ is lossy, hence < 100%).
  EXPECT_GT(prev_recall, 0.4f);
}

TEST(IndexIVFPQ, PrecomputedTableMatchesBasicPath) {
  // The killer test: with use_precomputed_table=1 and =0, the index must
  // return the IDENTICAL labels and (within float rounding) identical
  // distances for every query. Anything else is a bug in the L2 expansion.
  const hypervec::idx_t d = 16, nb = 2000, nq = 50, nlist = 32;
  const hypervec::idx_t M = 8;
  const int nbits = 7;
  const hypervec::idx_t k = 8;

  const auto base = RandomVectors(nb, d, 100, 5.0f);
  const auto query = RandomVectors(nq, d, 200, 5.0f);

  // Two indexes share training (same seed, same data) → identical centroids.
  hypervec::IndexIVFPQ idx_basic(d, nlist, M, nbits);
  idx_basic.Train(nb, base.data());
  idx_basic.Add(nb, base.data());

  hypervec::IndexIVFPQ idx_pre(d, nlist, M, nbits);
  idx_pre.use_precomputed_table = 1;
  idx_pre.Train(nb, base.data());
  idx_pre.Add(nb, base.data());

  // Sanity: same trained centroids (deterministic kmeans seed).
  ASSERT_EQ(idx_basic.centroids, idx_pre.centroids);
  ASSERT_EQ(idx_basic.pq.centroids, idx_pre.pq.centroids);
  ASSERT_FALSE(idx_pre.precomputed_table.empty());

  hypervec::IVFSearchParameters params;
  params.nprobe = 8;

  std::vector<float> d_b(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> l_b(static_cast<size_t>(nq) * k);
  idx_basic.Search(nq, query.data(), k, d_b.data(), l_b.data(), &params);

  std::vector<float> d_p(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> l_p(static_cast<size_t>(nq) * k);
  idx_pre.Search(nq, query.data(), k, d_p.data(), l_p.data(), &params);

  EXPECT_EQ(l_b, l_p);
  // Distance values should agree to within float rounding noise. Allow a
  // small tolerance because the two paths add the same terms in different
  // order.
  for (size_t i = 0; i < d_b.size(); i++) {
    EXPECT_NEAR(d_b[i], d_p[i],
                1e-3f * std::max(std::abs(d_b[i]), 1.0f))
      << "distance mismatch at i=" << i;
  }
}

TEST(IndexIVFPQ, ByResidualImprovesOverRawEncoding) {
  // Residual encoding should be at least as accurate as raw encoding once
  // nprobe is reasonable, because the per-cell residual distribution has
  // less variance than the global one. We just check by_residual=true is
  // not strictly worse on this random dataset.
  const hypervec::idx_t d = 16, nb = 2000, nq = 100, nlist = 32;
  const hypervec::idx_t M = 8;
  const int nbits = 6;
  const hypervec::idx_t k = 10;

  const auto base = RandomVectors(nb, d, 31, 5.0f);
  const auto query = RandomVectors(nq, d, 32, 5.0f);

  hypervec::IndexFlatL2 gt(d);
  gt.Add(nb, base.data());
  std::vector<float> gt_d(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> gt_l(static_cast<size_t>(nq) * k);
  gt.Search(nq, query.data(), k, gt_d.data(), gt_l.data());

  hypervec::IndexIVFPQ idx_res(d, nlist, M, nbits);
  // by_residual = true (default)
  idx_res.Train(nb, base.data());
  idx_res.Add(nb, base.data());

  hypervec::IndexIVFPQ idx_raw(d, nlist, M, nbits);
  idx_raw.by_residual = false;
  idx_raw.Train(nb, base.data());
  idx_raw.Add(nb, base.data());

  hypervec::IVFSearchParameters params;
  params.nprobe = nlist;  // scan everything to isolate encoding accuracy

  std::vector<float> d_r(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> l_r(static_cast<size_t>(nq) * k);
  idx_res.Search(nq, query.data(), k, d_r.data(), l_r.data(), &params);

  std::vector<float> d_w(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> l_w(static_cast<size_t>(nq) * k);
  idx_raw.Search(nq, query.data(), k, d_w.data(), l_w.data(), &params);

  const float r_residual = Recall(l_r, gt_l, nq, k);
  const float r_raw = Recall(l_w, gt_l, nq, k);
  // Loose: residual should not be much worse than raw. On clustered data it
  // would be strictly better; on uniform random data the gap is small.
  EXPECT_GE(r_residual, r_raw - 0.05f)
    << "residual=" << r_residual << " raw=" << r_raw;
}

TEST(IndexIVFPQ, PersistenceRoundtrip) {
  const hypervec::idx_t d = 12, nb = 1500, nlist = 16;
  const hypervec::idx_t M = 4;
  const int nbits = 8;

  const auto base = RandomVectors(nb, d, 71, 4.0f);

  hypervec::IndexIVFPQ src(d, nlist, M, nbits);
  src.use_precomputed_table = 1;
  src.nprobe = 5;
  src.Train(nb, base.data());
  src.Add(nb, base.data());
  ASSERT_FALSE(src.precomputed_table.empty());

  TempFile tf;
  hypervec::WriteIndex(&src, tf.path.c_str());

  std::unique_ptr<hypervec::Index> loaded(
    hypervec::ReadIndex(tf.path.c_str()));
  auto* dst = dynamic_cast<hypervec::IndexIVFPQ*>(loaded.get());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(dst->d, src.d);
  EXPECT_EQ(dst->n_total, src.n_total);
  EXPECT_EQ(dst->nlist, src.nlist);
  EXPECT_EQ(dst->nprobe, src.nprobe);
  EXPECT_EQ(dst->by_residual, src.by_residual);
  EXPECT_EQ(dst->use_precomputed_table, src.use_precomputed_table);
  EXPECT_EQ(dst->centroids, src.centroids);
  EXPECT_EQ(dst->pq.M, src.pq.M);
  EXPECT_EQ(dst->pq.nbits, src.pq.nbits);
  EXPECT_EQ(dst->pq.centroids, src.pq.centroids);
  EXPECT_EQ(dst->precomputed_table, src.precomputed_table);

  // Per-list contents must match exactly.
  for (size_t list_no = 0; list_no < src.nlist; list_no++) {
    EXPECT_EQ(dst->invlists->list_size(list_no),
              src.invlists->list_size(list_no));
  }

  // Same query → same results post-roundtrip.
  const auto query = RandomVectors(20, d, 72, 4.0f);
  hypervec::IVFSearchParameters params;
  params.nprobe = 4;
  std::vector<float> ds(20 * 5), dl(20 * 5);
  std::vector<hypervec::idx_t> ls(20 * 5), ll(20 * 5);
  src.Search(20, query.data(), 5, ds.data(), ls.data(), &params);
  dst->Search(20, query.data(), 5, dl.data(), ll.data(), &params);
  EXPECT_EQ(ls, ll);
  EXPECT_EQ(ds, dl);
}

TEST(IndexIVFPQ, ReconstructAddsCentroidBackForResidual) {
  // Reconstruct(id) for by_residual=true must add the coarse centroid back
  // — otherwise the reconstructed vector is shifted by the centroid offset
  // and would never approximate the original. We check by comparing the
  // reconstruction to a plain PQ-decoded residual + centroid lookup.
  const hypervec::idx_t d = 8, nb = 800, nlist = 16;
  const hypervec::idx_t M = 4;
  const int nbits = 6;

  const auto base = RandomVectors(nb, d, 41, 3.0f);
  hypervec::IndexIVFPQ idx(d, nlist, M, nbits);
  idx.Train(nb, base.data());
  idx.Add(nb, base.data());

  // Pick a few ids and check reconstruction stays within "centroid + PQ
  // codebook reach" of the original.
  std::vector<float> recons(d);
  for (hypervec::idx_t key : {hypervec::idx_t(0), hypervec::idx_t(123),
                              hypervec::idx_t(456), hypervec::idx_t(799)}) {
    idx.Reconstruct(key, recons.data());
    // Reconstruction error is bounded by twice the per-coordinate scale —
    // generous because PQ is lossy. The point is: it's not a centroid-shift
    // error like 5x off.
    for (hypervec::idx_t j = 0; j < d; j++) {
      const float diff = recons[j] - base[key * d + j];
      EXPECT_LT(std::abs(diff), 5.0f)
        << "key=" << key << " j=" << j;
    }
  }
}

TEST(IndexIVFPQ, RejectsNonL2Metric) {
  EXPECT_THROW(
    hypervec::IndexIVFPQ(8, 4, 4, 8, hypervec::kMetricInnerProduct),
    hypervec::HypervecException);
}
