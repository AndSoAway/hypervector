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
 * Leave a path empty to accept the default resolution performed by
 * DefaultAnalyzerParams().  An empty path is never interpreted as "skip this
 * file": all three dict files are required by CppjiebaAnalyzer.
 */
struct AnalyzerParams {
  std::string jieba_dict_path;  ///< jieba.dict.utf8
  std::string hmm_model_path;   ///< hmm_model.utf8
  std::string user_dict_path;   ///< user.dict.utf8 (its content may be empty)
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
   * Thread safety: implementations are safe to call concurrently from
   * multiple threads once constructed.
   */
  virtual std::vector<std::string> Analyze(const std::string& text) const = 0;
};

#if defined(HYPERVEC_ENABLE_BM25_ANALYZER)
/** Resolve the default dict paths.
 *
 * Resolution order:
 *   1. the HYPERVEC_DICT_DIR environment variable;
 *   2. the dict directory baked in at build time
 *      (${CMAKE_INSTALL_DATAROOTDIR}/hypervec/dict).
 *
 * When neither is available the returned paths are empty and constructing a
 * CppjiebaAnalyzer throws.
 */
AnalyzerParams DefaultAnalyzerParams();

/** Chinese segmentation using cppjieba (MixSegment mode).
 *
 * Wraps a DictTrie + HMMModel pair through MixSegment.  The dict files are
 * loaded once at construction and reused for all Analyze() calls.
 *
 * Exception contract: construction throws hypervec::HypervecException (derived
 * from std::exception) through HYPERVEC_THROW_IF_NOT_* when a required path is
 * empty or cannot be opened.  Nothing is aborted, so callers may recover from
 * a bad path.
 *
 * Thread safety: Analyze() only reads the dict structures loaded at
 * construction and keeps its working buffers on the stack, so concurrent calls
 * on the same instance are safe.  This relies on cppjieba's Cut() being const
 * and side-effect free; re-check after a cppjieba upgrade.
 */
struct CppjiebaAnalyzer : Analyzer {
  explicit CppjiebaAnalyzer(const AnalyzerParams& params);
  ~CppjiebaAnalyzer() override;

  CppjiebaAnalyzer(CppjiebaAnalyzer&&) noexcept;
  CppjiebaAnalyzer& operator=(CppjiebaAnalyzer&&) noexcept;
  CppjiebaAnalyzer(const CppjiebaAnalyzer&) = delete;
  CppjiebaAnalyzer& operator=(const CppjiebaAnalyzer&) = delete;

  std::vector<std::string> Analyze(const std::string& text) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace hypervec
