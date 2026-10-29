/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// Unit tests for CppjiebaAnalyzer (task T1, acceptance criteria §六).
//
// Dict files are resolved through DefaultAnalyzerParams(): the
// HYPERVEC_DICT_DIR environment variable, then the dict directory installed by
// CMake.  Tests that need real dicts are skipped when neither is reachable.

#include <utils/algo/bm25/analyzer.h>

#include <utils/log/exception.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

namespace {

hypervec::AnalyzerParams DefaultParams() {
  return hypervec::DefaultAnalyzerParams();
}

bool DictsAvailable() {
  const hypervec::AnalyzerParams params = DefaultParams();
  if (params.jieba_dict_path.empty() || params.hmm_model_path.empty() ||
      params.user_dict_path.empty()) {
    return false;
  }
  std::ifstream jieba(params.jieba_dict_path);
  std::ifstream hmm(params.hmm_model_path);
  std::ifstream user(params.user_dict_path);
  return jieba.good() && hmm.good() && user.good();
}

bool Contains(const std::vector<std::string>& terms, const std::string& tok) {
  return std::find(terms.begin(), terms.end(), tok) != terms.end();
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(CppjiebaAnalyzer, ThrowsOnMissingDictFile) {
  hypervec::AnalyzerParams params = DefaultParams();
  params.jieba_dict_path = "nonexistent-dir/jieba.dict.utf8";
  EXPECT_THROW(hypervec::CppjiebaAnalyzer a(params),
               hypervec::HypervecException);
}

TEST(CppjiebaAnalyzer, ThrowsOnMissingHmmModel) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::AnalyzerParams params = DefaultParams();
  params.hmm_model_path = "nonexistent-dir/hmm_model.utf8";
  EXPECT_THROW(hypervec::CppjiebaAnalyzer a(params),
               hypervec::HypervecException);
}

// ---------------------------------------------------------------------------
// Basic segmentation (§六 acceptance criteria)
// ---------------------------------------------------------------------------

TEST(CppjiebaAnalyzer, EmptyInputReturnsEmpty) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  EXPECT_TRUE(a.Analyze("").empty());
}

// 双字词
TEST(CppjiebaAnalyzer, TwoCharWord) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
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
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("电磁场理论");
  // "电磁场" should appear as a unit (in cppjieba's dict).
  EXPECT_TRUE(Contains(terms, "电磁场") || Contains(terms, "电磁"))
      << "expected at least '电磁' in output";
}

// 成语
TEST(CppjiebaAnalyzer, Idiom) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("守株待兔是一个成语");
  EXPECT_TRUE(Contains(terms, "守株待兔")) << "idiom '守株待兔' not segmented as a unit";
}

// 人名 (using HMM mode for OOV)
TEST(CppjiebaAnalyzer, PersonName) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  // "张华" is a common Chinese name; HMM should produce it as a single token.
  auto terms = a.Analyze("张华是一位工程师");
  EXPECT_TRUE(Contains(terms, "张华"))
      << "person name '张华' should be a single token (HMM mode)";
}

// 典型 RAG/领域文本
TEST(CppjiebaAnalyzer, DomainText) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("向量检索是大模型RAG的核心技术");
  EXPECT_TRUE(Contains(terms, "向量")) << "missing '向量'";
  EXPECT_TRUE(Contains(terms, "检索")) << "missing '检索'";
}

// ASCII 内容：全小写化
TEST(CppjiebaAnalyzer, AsciiLowercased) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
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
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("数据  分析");
  for (const auto& t : terms) {
    ASSERT_FALSE(t.empty()) << "empty token in output";
    EXPECT_NE(t.front(), ' ') << "leading space in token '" << t << "'";
    EXPECT_NE(t.back(), ' ')  << "trailing space in token '" << t << "'";
  }
}

// 纯英文文本 — 应正常返回小写 token
TEST(CppjiebaAnalyzer, EnglishText) {
  if (!DictsAvailable()) GTEST_SKIP() << "dict files not found";
  hypervec::CppjiebaAnalyzer a(DefaultParams());
  auto terms = a.Analyze("hello world");
  EXPECT_TRUE(Contains(terms, "hello"));
  EXPECT_TRUE(Contains(terms, "world"));
}
