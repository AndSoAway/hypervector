/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// Unit tests for BM25InvertedIndex (task T1, step 3/3).
//
// Documents are built directly as TF SparseRows (index = term id, value = raw
// term frequency); queries as IDF SparseRows.  Golden BM25 scores are computed
// by hand in comments using the same formula as SparseRow::dot_bm25:
//
//   score(doc) = Σ_{t ∈ q∩doc}  idf(t) · tf · (k1+1) / (tf + norm)
//   norm       = k1 · (1 − b + b · doc_len / avgdl)      (avgdl clamped ≥ 1)

#include <utils/algo/bm25/inverted_index.h>

#include <utils/algo/bm25/bm25.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace {

hypervec::SparseRow MakeRow(std::vector<uint32_t> idx,
                            std::vector<float> val) {
  return hypervec::SparseRow(std::move(idx), std::move(val));
}

constexpr float kEps = 1e-4f;

// BM25 term contribution helper mirroring dot_bm25 (for golden values).
double Bm25Term(double idf, double tf, double doc_len, double k1, double b,
                double avgdl) {
  const double norm = k1 * (1.0 - b + b * doc_len / std::max(avgdl, 1.0));
  return idf * tf * (k1 + 1.0) / (tf + norm);
}

}  // namespace

// ---------------------------------------------------------------------------
// Empty index
// ---------------------------------------------------------------------------

TEST(BM25InvertedIndex, EmptyIndexReturnsSentinels) {
  hypervec::BM25InvertedIndex idx;
  EXPECT_EQ(idx.NumDocs(), 0);
  EXPECT_EQ(idx.Dim(), 0);

  auto query = MakeRow({0}, {1.0f});
  float dist[3];
  hypervec::idx_t lbl[3];
  idx.Search(query, 3, dist, lbl);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(lbl[i], -1) << "slot " << i;
    EXPECT_TRUE(std::isinf(dist[i]));
  }
}

// ---------------------------------------------------------------------------
// Add bookkeeping
// ---------------------------------------------------------------------------

TEST(BM25InvertedIndex, AddTracksDimAndRowSum) {
  hypervec::BM25InvertedIndex idx;
  std::vector<hypervec::SparseRow> docs = {
      MakeRow({0, 2}, {1.0f, 3.0f}),  // doc0: len = 4, max term id 2
      MakeRow({1, 5}, {2.0f, 2.0f}),  // doc1: len = 4, max term id 5
  };
  idx.Add(2, docs.data(), 0);

  EXPECT_EQ(idx.NumDocs(), 2);
  EXPECT_EQ(idx.Dim(), 6) << "highest term id (5) + 1";
  EXPECT_NEAR(idx.RowSum(0), 4.0f, kEps);
  EXPECT_NEAR(idx.RowSum(1), 4.0f, kEps);
}

// ---------------------------------------------------------------------------
// Golden single-term BM25 score
// ---------------------------------------------------------------------------

// One doc: term 0 with tf=2, doc_len=2.  Query: term 0 with idf=1.5.
// k1=1.2, b=0.75, avgdl=2.0
//   norm = 1.2*(1 - 0.75 + 0.75*2/2) = 1.2*(0.25+0.75) = 1.2
//   score = 1.5 * 2*(1.2+1)/(2+1.2) = 1.5 * 4.4/3.2 = 1.5 * 1.375 = 2.0625
TEST(BM25InvertedIndex, GoldenSingleTerm) {
  hypervec::BM25InvertedIndex idx;
  auto doc = MakeRow({0}, {2.0f});
  idx.Add(1, &doc, 0);

  auto query = MakeRow({0}, {1.5f});
  hypervec::SparseSearchParameters params;
  params.bm25 = {1.2f, 0.75f, 2.0f};

  float dist[1];
  hypervec::idx_t lbl[1];
  idx.Search(query, 1, dist, lbl, &params);

  EXPECT_EQ(lbl[0], 0);
  const double golden = Bm25Term(1.5, 2.0, 2.0, 1.2, 0.75, 2.0);
  EXPECT_NEAR(dist[0], static_cast<float>(golden), kEps);
  EXPECT_NEAR(dist[0], 2.0625f, kEps);
}

// ---------------------------------------------------------------------------
// Ranking across multiple documents
// ---------------------------------------------------------------------------

// Corpus (term ids): doc0=[t0:3], doc1=[t0:1, t1:1], doc2=[t2:5]
// Query = [t0:idf=1.0]
// Only doc0 and doc1 match t0.  avgdl = (3+2+5)/3 = 10/3.
// doc0 has higher tf for t0, but longer doc_len penalizes it; verify ordering
// empirically against the golden helper rather than hard-coding the winner.
TEST(BM25InvertedIndex, RanksMatchingDocs) {
  hypervec::BM25InvertedIndex idx;
  std::vector<hypervec::SparseRow> docs = {
      MakeRow({0}, {3.0f}),          // doc0
      MakeRow({0, 1}, {1.0f, 1.0f}), // doc1
      MakeRow({2}, {5.0f}),          // doc2 (no shared term)
  };
  idx.Add(3, docs.data(), 0);

  auto query = MakeRow({0}, {1.0f});
  hypervec::SparseSearchParameters params;
  params.bm25 = {1.2f, 0.75f, 10.0f / 3.0f};

  float dist[3];
  hypervec::idx_t lbl[3];
  idx.Search(query, 3, dist, lbl, &params);

  // doc2 shares no term → must not appear; slot 2 is a sentinel.
  EXPECT_EQ(lbl[2], -1);
  EXPECT_TRUE(std::isinf(dist[2]));

  // The two matching docs occupy slots 0 and 1, sorted by score desc.
  EXPECT_GE(dist[0], dist[1]);

  const double avgdl = 10.0 / 3.0;
  const double s0 = Bm25Term(1.0, 3.0, 3.0, 1.2, 0.75, avgdl);
  const double s1 = Bm25Term(1.0, 1.0, 2.0, 1.2, 0.75, avgdl);
  const hypervec::idx_t winner = (s0 >= s1) ? 0 : 1;
  const hypervec::idx_t loser  = (s0 >= s1) ? 1 : 0;
  EXPECT_EQ(lbl[0], winner);
  EXPECT_EQ(lbl[1], loser);
  EXPECT_NEAR(dist[0], static_cast<float>(std::max(s0, s1)), kEps);
  EXPECT_NEAR(dist[1], static_cast<float>(std::min(s0, s1)), kEps);
}

// ---------------------------------------------------------------------------
// Multi-term query accumulates across terms
// ---------------------------------------------------------------------------

// doc0 = [t0:2, t1:1], doc_len=3.  Query = [t0:idf=1.0, t1:idf=2.0].
// avgdl=3, k1=1.2, b=0.75 → norm = 1.2*(1-0.75+0.75*1) = 1.2
//   score = 1.0*2*2.2/(2+1.2) + 2.0*1*2.2/(1+1.2)
//         = 1.0*4.4/3.2 + 2.0*2.2/2.2 = 1.375 + 2.0 = 3.375
TEST(BM25InvertedIndex, MultiTermAccumulates) {
  hypervec::BM25InvertedIndex idx;
  auto doc = MakeRow({0, 1}, {2.0f, 1.0f});
  idx.Add(1, &doc, 0);

  auto query = MakeRow({0, 1}, {1.0f, 2.0f});
  hypervec::SparseSearchParameters params;
  params.bm25 = {1.2f, 0.75f, 3.0f};

  float dist[1];
  hypervec::idx_t lbl[1];
  idx.Search(query, 1, dist, lbl, &params);

  EXPECT_EQ(lbl[0], 0);
  const double golden = Bm25Term(1.0, 2.0, 3.0, 1.2, 0.75, 3.0) +
                        Bm25Term(2.0, 1.0, 3.0, 1.2, 0.75, 3.0);
  EXPECT_NEAR(dist[0], static_cast<float>(golden), kEps);
  EXPECT_NEAR(dist[0], 3.375f, kEps);
}

// ---------------------------------------------------------------------------
// Sequential Add assigns continuous doc ids
// ---------------------------------------------------------------------------

TEST(BM25InvertedIndex, SequentialAddContinuesDocIds) {
  hypervec::BM25InvertedIndex idx;
  auto d0 = MakeRow({0}, {1.0f});
  idx.Add(1, &d0, 0);
  auto d1 = MakeRow({0}, {1.0f});
  idx.Add(1, &d1, 0);

  EXPECT_EQ(idx.NumDocs(), 2);

  auto query = MakeRow({0}, {1.0f});
  float dist[2];
  hypervec::idx_t lbl[2];
  idx.Search(query, 2, dist, lbl);

  // Both docs match; ids 0 and 1 must both be present.
  std::vector<hypervec::idx_t> got = {lbl[0], lbl[1]};
  EXPECT_NE(std::find(got.begin(), got.end(), 0), got.end());
  EXPECT_NE(std::find(got.begin(), got.end(), 1), got.end());
}

// ---------------------------------------------------------------------------
// k larger than result count pads with sentinels
// ---------------------------------------------------------------------------

TEST(BM25InvertedIndex, KLargerThanResultsPadsSentinels) {
  hypervec::BM25InvertedIndex idx;
  auto doc = MakeRow({0}, {1.0f});
  idx.Add(1, &doc, 0);

  auto query = MakeRow({0}, {1.0f});
  float dist[5];
  hypervec::idx_t lbl[5];
  idx.Search(query, 5, dist, lbl);

  EXPECT_NE(lbl[0], -1);
  for (int i = 1; i < 5; ++i) {
    EXPECT_EQ(lbl[i], -1) << "slot " << i;
    EXPECT_TRUE(std::isinf(dist[i]));
  }
}

// ---------------------------------------------------------------------------
// Query term absent from index yields no results
// ---------------------------------------------------------------------------

TEST(BM25InvertedIndex, QueryTermNotIndexedYieldsNothing) {
  hypervec::BM25InvertedIndex idx;
  auto doc = MakeRow({0}, {1.0f});
  idx.Add(1, &doc, 0);

  auto query = MakeRow({99}, {1.0f});  // term 99 never indexed
  float dist[1];
  hypervec::idx_t lbl[1];
  idx.Search(query, 1, dist, lbl);

  EXPECT_EQ(lbl[0], -1);
  EXPECT_TRUE(std::isinf(dist[0]));
}

// ---------------------------------------------------------------------------
// BM25Score free function (pure, index-independent)
// ---------------------------------------------------------------------------

// doc TF = [t0:2, t1:1], query IDF = [t0:1.0, t1:2.0], doc_len=3, avgdl=3
//   norm = 1.2*(1-0.75+0.75*1) = 1.2
//   score = 1.0*2*2.2/(2+1.2) + 2.0*1*2.2/(1+1.2) = 1.375 + 2.0 = 3.375
TEST(BM25Score, GoldenMultiTerm) {
  auto doc_tf    = MakeRow({0, 1}, {2.0f, 1.0f});
  auto query_idf = MakeRow({0, 1}, {1.0f, 2.0f});
  hypervec::BM25Params params{1.2f, 0.75f, 3.0f};

  const float score = hypervec::BM25Score(query_idf, doc_tf, params, 3.0f);
  EXPECT_NEAR(score, 3.375f, kEps);
}

// No shared term → score 0.
TEST(BM25Score, DisjointTermsScoreZero) {
  auto doc_tf    = MakeRow({0}, {5.0f});
  auto query_idf = MakeRow({7}, {1.0f});
  hypervec::BM25Params params{1.2f, 0.75f, 5.0f};

  EXPECT_NEAR(hypervec::BM25Score(query_idf, doc_tf, params, 5.0f), 0.0f, kEps);
}

// BM25Score and SparseRow::dot_bm25 must agree (dot_bm25 delegates to it).
TEST(BM25Score, AgreesWithDotBm25Member) {
  auto doc_tf    = MakeRow({0, 2, 5}, {3.0f, 1.0f, 2.0f});
  auto query_idf = MakeRow({0, 5},    {1.5f, 0.8f});
  hypervec::BM25Params params{1.2f, 0.75f, 4.0f};
  const float doc_len = 6.0f;

  const float via_free   = hypervec::BM25Score(query_idf, doc_tf, params, doc_len);
  const float via_member = doc_tf.dot_bm25(query_idf, params, doc_len);
  EXPECT_NEAR(via_free, via_member, kEps);
}

// avgdl below 1.0 is clamped to 1.0 (matches bm25_term_score contract).
TEST(BM25Score, AvgdlClampedToOne) {
  auto doc_tf    = MakeRow({0}, {1.0f});
  auto query_idf = MakeRow({0}, {1.0f});
  hypervec::BM25Params low{1.2f, 0.75f, 0.1f};   // avgdl < 1 → clamp to 1
  hypervec::BM25Params one{1.2f, 0.75f, 1.0f};

  EXPECT_NEAR(hypervec::BM25Score(query_idf, doc_tf, low, 1.0f),
              hypervec::BM25Score(query_idf, doc_tf, one, 1.0f), kEps);
}

// ---------------------------------------------------------------------------
// Document frequency (DF) and avgdl from the inverted index (§六.4)
// ---------------------------------------------------------------------------

// Corpus: doc0=[t0, t1], doc1=[t1, t2], doc2=[t1]
//   DF(t0)=1, DF(t1)=3, DF(t2)=1, DF(unseen)=0
//   lengths 2,2,1 → avgdl = 5/3
TEST(BM25InvertedIndex, DocFreqAndAvgdl) {
  hypervec::BM25InvertedIndex idx;
  std::vector<hypervec::SparseRow> docs = {
      MakeRow({0, 1}, {1.0f, 1.0f}),
      MakeRow({1, 2}, {1.0f, 1.0f}),
      MakeRow({1},    {1.0f}),
  };
  idx.Add(3, docs.data(), 0);

  EXPECT_EQ(idx.DocFreq(0), 1);
  EXPECT_EQ(idx.DocFreq(1), 3);
  EXPECT_EQ(idx.DocFreq(2), 1);
  EXPECT_EQ(idx.DocFreq(999), 0) << "unseen term DF must be 0";

  EXPECT_NEAR(idx.Avgdl(), 5.0f / 3.0f, kEps);
}

// DF counts documents, not term occurrences: a repeated term within one doc
// still contributes DF=1 for that doc.
TEST(BM25InvertedIndex, DocFreqCountsDocsNotOccurrences) {
  hypervec::BM25InvertedIndex idx;
  // Single doc where term 0 has tf=5 (one posting entry, tf aggregated).
  auto doc = MakeRow({0}, {5.0f});
  idx.Add(1, &doc, 0);

  EXPECT_EQ(idx.DocFreq(0), 1) << "one document → DF=1 regardless of tf";
}

// ---------------------------------------------------------------------------
// Concurrent read-only Search (§六.6: no data race under parallel queries)
// ---------------------------------------------------------------------------

// After Add completes, Search is read-only and must be safe to call from many
// threads at once.  Run the same query on N threads and require identical
// results — a data race would surface as a mismatch or a crash under TSan.
TEST(BM25InvertedIndex, ConcurrentSearchIsConsistent) {
  hypervec::BM25InvertedIndex idx;
  std::vector<hypervec::SparseRow> docs = {
      MakeRow({0, 1}, {2.0f, 1.0f}),
      MakeRow({1, 2}, {1.0f, 3.0f}),
      MakeRow({0, 2}, {1.0f, 1.0f}),
      MakeRow({1},    {4.0f}),
  };
  idx.Add(4, docs.data(), 0);

  auto query = MakeRow({0, 1, 2}, {1.0f, 1.5f, 0.8f});
  hypervec::SparseSearchParameters params;
  params.bm25 = {1.2f, 0.75f, 3.0f};

  // Reference result from a single-threaded call.
  constexpr int kK = 4;
  float ref_dist[kK];
  hypervec::idx_t ref_lbl[kK];
  idx.Search(query, kK, ref_dist, ref_lbl, &params);

  constexpr int kThreads = 8;
  std::vector<std::thread> workers;
  std::vector<int> mismatches(kThreads, 0);

  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t]() {
      for (int iter = 0; iter < 200; ++iter) {
        float d[kK];
        hypervec::idx_t l[kK];
        idx.Search(query, kK, d, l, &params);
        for (int i = 0; i < kK; ++i) {
          if (l[i] != ref_lbl[i] || std::fabs(d[i] - ref_dist[i]) > kEps) {
            ++mismatches[t];
          }
        }
      }
    });
  }
  for (auto& w : workers) w.join();

  for (int t = 0; t < kThreads; ++t) {
    EXPECT_EQ(mismatches[t], 0) << "thread " << t << " saw inconsistent results";
  }
}
