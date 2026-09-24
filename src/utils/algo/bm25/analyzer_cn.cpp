/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/algo/bm25/analyzer.h>

#include <utils/log/assert.h>

// cppjieba is header-only; it is pulled in via FetchContent and its include
// directories are attached to the hypervec target by fetch_cppjieba.cmake.
#include "cppjieba/Jieba.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hypervec {

namespace {

// Strip leading and trailing ASCII whitespace from s (in-place).
void trim_inplace(std::string& s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

// Lowercase ASCII characters in s (in-place).  Non-ASCII bytes are untouched
// because Chinese characters are multi-byte UTF-8 and ::tolower must not be
// applied to them.
void ascii_lower_inplace(std::string& s) {
  for (char& c : s) {
    if (static_cast<unsigned char>(c) < 128) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }
}

}  // namespace

struct CppjiebaAnalyzer::Impl {
  cppjieba::Jieba jieba;

  explicit Impl(const AnalyzerParams& p)
      : jieba(p.jieba_dict_path, p.hmm_model_path, p.user_dict_path,
              p.idf_path, p.stop_words_path) {}
};

CppjiebaAnalyzer::CppjiebaAnalyzer(const AnalyzerParams& params) {
  HYPERVEC_THROW_IF_NOT_MSG(
      !params.jieba_dict_path.empty(),
      "CppjiebaAnalyzer: jieba_dict_path must not be empty");
  HYPERVEC_THROW_IF_NOT_MSG(
      !params.hmm_model_path.empty(),
      "CppjiebaAnalyzer: hmm_model_path must not be empty");
  HYPERVEC_THROW_IF_NOT_MSG(
      !params.user_dict_path.empty(),
      "CppjiebaAnalyzer: user_dict_path must not be empty");
  impl_ = std::make_unique<Impl>(params);
}

CppjiebaAnalyzer::~CppjiebaAnalyzer() = default;

std::vector<std::string> CppjiebaAnalyzer::Analyze(
    const std::string& text) const {
  std::vector<std::string> raw;
  impl_->jieba.Cut(text, raw, /*hmm=*/true);

  std::vector<std::string> result;
  result.reserve(raw.size());
  for (std::string& tok : raw) {
    trim_inplace(tok);
    ascii_lower_inplace(tok);
    if (!tok.empty()) {
      result.push_back(std::move(tok));
    }
  }
  return result;
}

}  // namespace hypervec
