/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/algo/bm25/analyzer.h>

#include <utils/log/assert.h>

// cppjieba is vendored under third_party/cppjieba; its include paths are
// provided by the cppjieba_iface INTERFACE target.
#include "cppjieba/DictTrie.hpp"
#include "cppjieba/HMMModel.hpp"
#include "cppjieba/MixSegment.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

constexpr const char* kDictDirEnvVar = "HYPERVEC_DICT_DIR";

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

std::string join_path(const std::string& dir, const char* file) {
  if (dir.empty()) {
    return std::string();
  }
  std::string path = dir;
  if (path.back() != '/' && path.back() != '\\') {
    path += '/';
  }
  path += file;
  return path;
}

std::string default_dict_dir() {
  const char* from_env = std::getenv(kDictDirEnvVar);
  if (from_env != nullptr && from_env[0] != '\0') {
    return from_env;
  }
#if defined(HYPERVEC_DEFAULT_DICT_DIR)
  return HYPERVEC_DEFAULT_DICT_DIR;
#else
  return std::string();
#endif
}

AnalyzerParams resolve_params(const AnalyzerParams& params) {
  AnalyzerParams resolved = params;
  if (resolved.jieba_dict_path.empty() || resolved.hmm_model_path.empty() ||
      resolved.user_dict_path.empty()) {
    const AnalyzerParams defaults = DefaultAnalyzerParams();
    if (resolved.jieba_dict_path.empty()) {
      resolved.jieba_dict_path = defaults.jieba_dict_path;
    }
    if (resolved.hmm_model_path.empty()) {
      resolved.hmm_model_path = defaults.hmm_model_path;
    }
    if (resolved.user_dict_path.empty()) {
      resolved.user_dict_path = defaults.user_dict_path;
    }
  }
  return resolved;
}

// cppjieba validates its inputs with limonp's XCHECK, which calls abort() on
// failure.  Check readability first so a bad path becomes a catchable
// exception instead of killing the process.
void check_readable(const std::string& path, const char* what) {
  HYPERVEC_THROW_IF_NOT_FMT(
      !path.empty(),
      "%s is not set; pass AnalyzerParams or set HYPERVEC_DICT_DIR", what);
  std::ifstream stream(path);
  HYPERVEC_THROW_IF_NOT_FMT(stream.good(), "cannot read %s: %s", what,
                            path.c_str());
}

}  // namespace

AnalyzerParams DefaultAnalyzerParams() {
  const std::string dir = default_dict_dir();
  AnalyzerParams params;
  params.jieba_dict_path = join_path(dir, "jieba.dict.utf8");
  params.hmm_model_path = join_path(dir, "hmm_model.utf8");
  params.user_dict_path = join_path(dir, "user.dict.utf8");
  return params;
}

struct CppjiebaAnalyzer::Impl {
  // Declaration order matters: MixSegment keeps non-owning pointers to the
  // trie and the model, so both must outlive it.
  cppjieba::DictTrie dict_trie;
  cppjieba::HMMModel hmm_model;
  cppjieba::MixSegment segment;

  explicit Impl(const AnalyzerParams& params)
      : dict_trie(params.jieba_dict_path, params.user_dict_path),
        hmm_model(params.hmm_model_path),
        segment(&dict_trie, &hmm_model) {}
};

CppjiebaAnalyzer::CppjiebaAnalyzer(const AnalyzerParams& params) {
  const AnalyzerParams resolved = resolve_params(params);
  check_readable(resolved.jieba_dict_path, "jieba_dict_path");
  check_readable(resolved.hmm_model_path, "hmm_model_path");
  check_readable(resolved.user_dict_path, "user_dict_path");
  impl_ = std::make_unique<Impl>(resolved);
}

CppjiebaAnalyzer::~CppjiebaAnalyzer() = default;

CppjiebaAnalyzer::CppjiebaAnalyzer(CppjiebaAnalyzer&&) noexcept = default;

CppjiebaAnalyzer& CppjiebaAnalyzer::operator=(
    CppjiebaAnalyzer&&) noexcept = default;

std::vector<std::string> CppjiebaAnalyzer::Analyze(
    const std::string& text) const {
  std::vector<std::string> raw;
  impl_->segment.Cut(text, raw, /*hmm=*/true);

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
