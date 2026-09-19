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
#include <utils/common/range_search_result.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/random.h>

#include <cmath>
#include <cstdio>
#include <limits>
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
  EXPECT_EQ(lvq.decoded_codebooks.size(),
            static_cast<size_t>(lvq.nlocal * lvq.ksub * lvq.d));

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

TEST(IndexLVQ, InvalidAddDoesNotMutateCodes) {
  const auto x = RandomVectors(32, 4, 23);
  hypervec::IndexLVQ index(4, 2, 2);
  index.Train(32, x.data());
  index.Add(1, x.data());
  const auto original_codes = index.codes.owned_data;

  EXPECT_THROW(index.Add(-1, x.data()), hypervec::HypervecException);
  EXPECT_THROW(index.Add(1, nullptr), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 1);
  EXPECT_EQ(index.codes.owned_data, original_codes);

  std::unique_ptr<hypervec::DistanceComputer> distance(
      index.GetDistanceComputer());
  distance->SetQuery(x.data());
  EXPECT_THROW((*distance)(1), hypervec::HypervecException);
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

TEST(IndexIVFLVQ, ScannerRangeSearchSupportsResidualAndRawCodes) {
  constexpr hypervec::idx_t kDimension = 4;
  constexpr hypervec::idx_t kCount = 32;
  constexpr hypervec::idx_t kTarget = 7;
  const auto base = RandomVectors(kCount, kDimension, 35, 4.0F);
  std::vector<hypervec::idx_t> ids(static_cast<size_t>(kCount));
  for (hypervec::idx_t i = 0; i < kCount; ++i) {
    ids[static_cast<size_t>(i)] = 1000 + i;
  }

  for (const bool by_residual : {true, false}) {
    hypervec::IndexIVFLVQ index(kDimension, 2, 2, 2);
    index.by_residual = by_residual;
    index.Train(kCount, base.data());
    index.AddWithIds(kCount, base.data(), ids.data());

    hypervec::IDSelectorRange selector(ids[kTarget], ids[kTarget] + 1);
    hypervec::IVFSearchParameters params;
    params.nprobe = 2;
    params.sel = &selector;
    hypervec::RangeSearchResult result(1);
    const float* query = base.data() + kTarget * kDimension;
    index.RangeSearch(1, query, (std::numeric_limits<float>::infinity)(),
                      &result, &params);

    ASSERT_EQ(result.lims[1] - result.lims[0], 1U) << by_residual;
    EXPECT_EQ(result.labels[0], ids[kTarget]) << by_residual;

    std::vector<float> distances(static_cast<size_t>(kCount));
    std::vector<hypervec::idx_t> labels(static_cast<size_t>(kCount));
    params.sel = nullptr;
    index.Search(1, query, kCount, distances.data(), labels.data(), &params);
    bool found = false;
    for (hypervec::idx_t i = 0; i < kCount; ++i) {
      if (labels[static_cast<size_t>(i)] == ids[kTarget]) {
        EXPECT_FLOAT_EQ(result.distances[0], distances[static_cast<size_t>(i)])
            << by_residual;
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << by_residual;
  }
}

TEST(IndexIVFLVQ, PersistenceRoundtripUsesScanner) {
  constexpr hypervec::idx_t kDimension = 8;
  constexpr hypervec::idx_t kBaseCount = 256;
  constexpr hypervec::idx_t kQueryCount = 6;
  constexpr hypervec::idx_t kNeighbors = 4;
  const auto base = RandomVectors(kBaseCount, kDimension, 36, 4.0F);
  const auto queries = RandomVectors(kQueryCount, kDimension, 37, 4.0F);

  hypervec::IndexIVFLVQ source(kDimension, 8, 4, 3);
  source.nprobe = 3;
  source.Train(kBaseCount, base.data());
  source.Add(kBaseCount, base.data());

  TempFile file;
  hypervec::WriteIndex(&source, file.path.c_str());
  std::unique_ptr<hypervec::Index> loaded(
      hypervec::ReadIndex(file.path.c_str()));
  auto* restored = dynamic_cast<hypervec::IndexIVFLVQ*>(loaded.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_TRUE(restored->GetCapabilities().supports_range_search);

  std::vector<float> source_distances(
      static_cast<size_t>(kQueryCount * kNeighbors));
  std::vector<float> restored_distances(source_distances.size());
  std::vector<hypervec::idx_t> source_labels(source_distances.size());
  std::vector<hypervec::idx_t> restored_labels(source_distances.size());
  source.Search(kQueryCount, queries.data(), kNeighbors,
                source_distances.data(), source_labels.data());
  restored->Search(kQueryCount, queries.data(), kNeighbors,
                   restored_distances.data(), restored_labels.data());
  EXPECT_EQ(restored_distances, source_distances);
  EXPECT_EQ(restored_labels, source_labels);
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
  std::unique_ptr<hypervec::Index> loaded(hypervec::ReadIndex(tf.path.c_str()));
  auto* dst = dynamic_cast<hypervec::IndexLVQ*>(loaded.get());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(dst->d, src.d);
  EXPECT_EQ(dst->n_total, src.n_total);
  EXPECT_EQ(dst->lvq.nlocal, src.lvq.nlocal);
  EXPECT_EQ(dst->lvq.nbits, src.lvq.nbits);
  EXPECT_EQ(dst->lvq.local_centroids, src.lvq.local_centroids);
  EXPECT_EQ(dst->lvq.residual_codebooks, src.lvq.residual_codebooks);
  EXPECT_EQ(dst->lvq.decoded_codebooks, src.lvq.decoded_codebooks);

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
  EXPECT_EQ(dst->decoded_codebooks, src.decoded_codebooks);
}
