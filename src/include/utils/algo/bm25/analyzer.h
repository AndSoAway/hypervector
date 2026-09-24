/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace hypervec {

/** Parameters for constructing an Analyzer.
 *
 * Dict paths are only consumed by CppjiebaAnalyzer; a whitespace-only
 * implementation would ignore them.  Passing empty strings causes
 * CppjiebaAnalyzer to throw at construction time.
 */
struct AnalyzerParams {
  std::string jieba_dict_path;       // jieba.dict.utf8
  std::string hmm_model_path;        // hmm_model.utf8
  std::string user_dict_path;        // user.dict.utf8 (may be empty file)
  std::string idf_path;              // idf.utf8 (pass "" to skip)
  std::string stop_words_path;       // stop_words.utf8 (pass "" to skip)
};

/** Text segmentation interface.
 *
 * Implementations are expected to be heavyweight (loading dict files on
 * construction) and cheap to call repeatedly.  Callers should keep a single
 * Analyzer instance per collection and reuse it across Analyze() calls.
 */
struct Analyzer {
  virtual ~Analyzer() = default;

  /** Segment `text` into a list of terms.
   *
   * The returned terms are lowercased and stripped of leading/trailing ASCII
   * whitespace.  Empty terms are omitted.  The order matches the input text.
   *
   * Thread safety: implementations must be safe to call concurrently from
   * multiple threads once constructed.
   */
  virtual std::vector<std::string> Analyze(const std::string& text) const = 0;
};

/** Chinese segmentation using cppjieba (MixSegment mode).
 *
 * Wraps cppjieba::Jieba and exposes it through the Analyzer interface.  The
 * Jieba object is constructed once from the dict files specified in `params`
 * and reused for all Analyze() calls.
 *
 * Dict files must exist at the given paths; construction throws
 * std::runtime_error if any required path is empty or the file cannot be read.
 *
 * Thread safety: cppjieba::Jieba::Cut is const and internally thread-safe for
 * read operations; this wrapper inherits that guarantee.
 */
struct CppjiebaAnalyzer : Analyzer {
  explicit CppjiebaAnalyzer(const AnalyzerParams& params);
  ~CppjiebaAnalyzer() override;

  std::vector<std::string> Analyze(const std::string& text) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hypervec
