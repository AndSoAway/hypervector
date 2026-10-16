/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// Unit tests for TextToSparse (task T1, step 2/3).
//
// No dict files or Analyzer dependency — tokens are passed directly as
// vector<string>.  Golden TF/IDF values are computed by hand in comments.

#include <utils/algo/bm25/text_to_sparse.h>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::unordered_map<uint32_t, float> AsMap(const hypervec::SparseRow& row) {
  std::unordered_map<uint32_t, float> m;
  for (size_t i = 0; i < row.nnz(); ++i) m[row.index_at(i)] = row.value_at(i);
  return m;
}

constexpr float kEps = 1e-5f;

}  // namespace

// ---------------------------------------------------------------------------
// Empty input
// ---------------------------------------------------------------------------

TEST(TextToSparse, EmptyDocumentProducesEmptyRow) {
  hypervec::TextToSparse tts;
  EXPECT_EQ(tts.AddDocument({}).nnz(), 0u);
  EXPECT_EQ(tts.NumDocs(), 0u);
}

TEST(TextToSparse, EmptyQueryProducesEmptyRow) {
  hypervec::TextToSparse tts;
  tts.AddDocument({"hello", "world"});
  EXPECT_EQ(tts.QueryToSparse({}).nnz(), 0u);
}

// ---------------------------------------------------------------------------
// TF golden values — single document
// ---------------------------------------------------------------------------

// Doc: [a, b, a]  →  TF(a)=2, TF(b)=1
// NumDocs=1, total_tokens=3, avgdl=3.0
TEST(TextToSparse, SingleDocTF) {
  hypervec::TextToSparse tts;
  const auto row = tts.AddDocument({"a", "b", "a"});

  EXPECT_EQ(row.nnz(), 2u);
  EXPECT_EQ(tts.NumDocs(), 1u);
  EXPECT_NEAR(tts.Avgdl(), 3.0f, kEps);

  auto m = AsMap(row);
  // ids assigned first-seen: a→0, b→1
  EXPECT_NEAR(m.at(0), 2.0f, kEps) << "TF(a) should be 2";
  EXPECT_NEAR(m.at(1), 1.0f, kEps) << "TF(b) should be 1";
}

// ---------------------------------------------------------------------------
// avgdl across two documents
// ---------------------------------------------------------------------------

// Doc1: [a,b] len=2   Doc2: [b,c] len=2   avgdl = 4/2 = 2.0
TEST(TextToSparse, TwoDocAvgdl) {
  hypervec::TextToSparse tts;
  tts.AddDocument({"a", "b"});
  tts.AddDocument({"b", "c"});

  EXPECT_EQ(tts.NumDocs(), 2u);
  EXPECT_NEAR(tts.Avgdl(), 2.0f, kEps);
}

// ---------------------------------------------------------------------------
// IDF golden values
// ---------------------------------------------------------------------------

// Corpus: Doc1=[a,b], Doc2=[b,c]
//   N=2, DF(a)=1, DF(b)=2, DF(c)=1
//   IDF(a) = log(1 + (2-1+0.5)/(1+0.5)) = log(1 + 1.5/1.5) = log(2)
//   IDF(b) = log(1 + (2-2+0.5)/(2+0.5)) = log(1 + 0.5/2.5) = log(1.2)
//   IDF(c) = log(1 + (2-1+0.5)/(1+0.5)) = log(2)
TEST(TextToSparse, IdfGoldenValues) {
  hypervec::TextToSparse tts;
  tts.AddDocument({"a", "b"});  // a→0, b→1
  tts.AddDocument({"b", "c"});  // c→2

  const auto qrow = tts.QueryToSparse({"a", "b", "c"});
  EXPECT_EQ(qrow.nnz(), 3u);

  auto m = AsMap(qrow);
  EXPECT_NEAR(m.at(0), static_cast<float>(std::log(2.0)),   kEps) << "IDF(a)";
  EXPECT_NEAR(m.at(1), static_cast<float>(std::log(1.2)),   kEps) << "IDF(b)";
  EXPECT_NEAR(m.at(2), static_cast<float>(std::log(2.0)),   kEps) << "IDF(c)";
}

// ---------------------------------------------------------------------------
// Repeated query token deduplication
// ---------------------------------------------------------------------------

TEST(TextToSparse, RepeatedQueryTokenDeduplicated) {
  hypervec::TextToSparse tts;
  tts.AddDocument({"a", "b"});
  EXPECT_EQ(tts.QueryToSparse({"a", "a", "a"}).nnz(), 1u);
}

// ---------------------------------------------------------------------------
// Unknown query token silently dropped
// ---------------------------------------------------------------------------

TEST(TextToSparse, UnknownQueryTokenDropped) {
  hypervec::TextToSparse tts;
  tts.AddDocument({"a", "b"});
  const auto qrow = tts.QueryToSparse({"a", "z"});  // z never indexed
  EXPECT_EQ(qrow.nnz(), 1u);
  EXPECT_TRUE(AsMap(qrow).count(0)) << "'a' (id=0) should be present";
}

// ---------------------------------------------------------------------------
// Dict continuity across multiple AddDocument calls
// ---------------------------------------------------------------------------

TEST(TextToSparse, DictContinuity) {
  hypervec::TextToSparse tts;
  tts.AddDocument({"x", "y"});  // x→0, y→1
  tts.AddDocument({"y", "z"});  // z→2

  EXPECT_EQ(tts.Dict().id_of("x"), 0u);
  EXPECT_EQ(tts.Dict().id_of("y"), 1u);
  EXPECT_EQ(tts.Dict().id_of("z"), 2u);
}

// ---------------------------------------------------------------------------
// SparseRow is sorted by index ascending
// ---------------------------------------------------------------------------

TEST(TextToSparse, TfRowSortedByIndex) {
  hypervec::TextToSparse tts;
  const auto row = tts.AddDocument({"c", "b", "a"});  // c→0, b→1, a→2
  for (size_t i = 1; i < row.nnz(); ++i) {
    EXPECT_LT(row.index_at(i - 1), row.index_at(i));
  }
}

// ---------------------------------------------------------------------------
// avgdl with varied document lengths
// ---------------------------------------------------------------------------

// Lengths 1,2,3,4,5 → total=15, avgdl=3.0
TEST(TextToSparse, AvgdlMultipleDocs) {
  hypervec::TextToSparse tts;
  tts.AddDocument({"a"});
  tts.AddDocument({"a", "b"});
  tts.AddDocument({"a", "b", "c"});
  tts.AddDocument({"a", "b", "c", "d"});
  tts.AddDocument({"a", "b", "c", "d", "e"});

  EXPECT_EQ(tts.NumDocs(), 5u);
  EXPECT_NEAR(tts.Avgdl(), 3.0f, kEps);
}
