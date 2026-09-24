/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// Unit tests for CppjiebaAnalyzer (task T1, acceptance criteria §六).
//
// Dict files are resolved from CPPJIEBA_DICT_DIR, which is set by
// cmake/thirdparty/fetch_cppjieba.cmake to the FetchContent source directory.
// Tests require the cmake variable to be defined; if it is not (e.g. running
// the binary directly without cmake), they are skipped.

#include <utils/algo/bm25/analyzer.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

// CPPJIEBA_DICT_DIR is injected by CMake as a compile definition (a string
// literal including surrounding quotes, e.g. "/path/to/dict").
#ifndef CPPJIEBA_DICT_DIR
#define CPPJIEBA_DICT_DIR ""
#endif

hypervec::AnalyzerParams DefaultParams() {
  const std::string dir = std::string(CPPJIEBA_DICT_DIR);
  return hypervec::AnalyzerParams{
      dir + "/jieba.dict.utf8",
      dir + "/hmm_model.utf8",
      dir + "/user.dict.utf8",
      dir + "/idf.utf8",
      dir + "/stop_words.utf8",
  };
}

bool DictDirAvailable() {
  return std::string(CPPJIEBA_DICT_DIR) != std::string("");
}

// Check that `expected` is a subsequence of `terms` (all expected tokens
// appear, in order, within the actual output).
bool ContainsInOrder(const std::vector<std::string>& terms,
                     const std::vector<std::string>& expected) {
  size_t ei = 0;
  for (const auto& t : terms) {
    if (ei < expected.size() && t == expected[ei]) ++ei;
  }
  return ei == expected.size();
}

bool Contains(const std::vector<std::string>& terms, const std::string& tok) {
  return std::find(terms.begin(), terms.end(), tok) != terms.end();
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(CppjiebaAnalyzer, ThrowsOnEmptyDictPath) {
  hypervec::AnalyzerParams p;
  // All paths empty — must throw.
  EXPECT_THROW(hypervec::CppjiebaAnalyzer a(p), std::exception);
}

TEST(CppjiebaAnalyzer, ThrowsOnEmptyHmmPath) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::AnalyzerParams p = DefaultParams();
  p.hmm_model_path = "";
  EXPECT_THROW(hypervec::CppjiebaAnalyzer a(p), std::exception);
}

// ---------------------------------------------------------------------------
// Basic segmentation (§六 acceptance criteria)
// ---------------------------------------------------------------------------

TEST(CppjiebaAnalyzer, EmptyInputReturnsEmpty) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  EXPECT_TRUE(a.Analyze("").empty());
}

// 双字词
TEST(CppjiebaAnalyzer, TwoCharWord) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  // "数据" appears as an independent word in "数据库" context.
  auto terms = a.Analyze("数据库查询");
  EXPECT_TRUE(Contains(terms, "数据库") || Contains(terms, "数据"))
      << "expected '数据库' or '数据' in output";
  // "检索" is a clear standalone two-char word.
  auto terms2 = a.Analyze("文本检索");
  EXPECT_TRUE(Contains(terms2, "检索") || Contains(terms2, "文本检索"))
      << "expected '检索' or '文本检索' in output";
}

// 三字词
TEST(CppjiebaAnalyzer, ThreeCharWord) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("电磁场理论");
  // "电磁场" should appear as a unit (in cppjieba's dict).
  EXPECT_TRUE(Contains(terms, "电磁场") || Contains(terms, "电磁"))
      << "expected at least '电磁' in output";
}

// 成语
TEST(CppjiebaAnalyzer, Idiom) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("守株待兔是一个成语");
  EXPECT_TRUE(Contains(terms, "守株待兔")) << "idiom '守株待兔' not segmented as a unit";
}

// 人名 (using HMM mode for OOV)
TEST(CppjiebaAnalyzer, PersonName) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  // "张华" is a common Chinese name; HMM should produce it as a single token.
  auto terms = a.Analyze("张华是一位工程师");
  EXPECT_TRUE(Contains(terms, "张华"))
      << "person name '张华' should be a single token (HMM mode)";
}

// 典型 RAG/领域文本
TEST(CppjiebaAnalyzer, DomainText) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("向量检索是大模型RAG的核心技术");
  EXPECT_TRUE(Contains(terms, "向量")) << "missing '向量'";
  EXPECT_TRUE(Contains(terms, "检索")) << "missing '检索'";
}

// ASCII 内容：全小写化
TEST(CppjiebaAnalyzer, AsciiLowercased) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("RAG System");
  for (const auto& t : terms) {
    for (char c : t) {
      if (static_cast<unsigned char>(c) < 128) {
        EXPECT_EQ(c, static_cast<char>(std::tolower(static_cast<unsigned char>(c))))
            << "ASCII char not lowercased in token '" << t << "'";
      }
    }
  }
}

// 空白字符不作为 token 输出
TEST(CppjiebaAnalyzer, WhitespaceNotInOutput) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("数据  分析");
  for (const auto& t : terms) {
    EXPECT_FALSE(t.empty()) << "empty token in output";
    EXPECT_NE(t.front(), ' ') << "leading space in token '" << t << "'";
    EXPECT_NE(t.back(), ' ')  << "trailing space in token '" << t << "'";
  }
}

// 纯英文文本 — 应正常返回小写 token
TEST(CppjiebaAnalyzer, EnglishText) {
  if (!DictDirAvailable()) GTEST_SKIP() << "CPPJIEBA_DICT_DIR not set";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("hello world");
  EXPECT_TRUE(Contains(terms, "hello"));
  EXPECT_TRUE(Contains(terms, "world"));
}
