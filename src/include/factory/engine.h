/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <factory/index_registry.h>

#include <memory>
#include <string_view>

namespace hypervec {

/** Lightweight C++ creation entry point; this is not a network service.
 *
 * Construction idempotently ensures that built-in descriptors are present in
 * the process-wide IndexRegistry. It does not allocate an index or start any
 * background service.
 */
class Engine {
 public:
  explicit Engine(ResourceContext resources = {});

  IndexConfig NormalizeConfig(const IndexConfig& config) const;

  /** Apply index defaults and validate per-search overrides.
   *
   * Factory validation is strict. Invalid overrides are rejected instead of
   * relying on a lower-level index to clamp them; for example, IVF nprobe must
   * be in [1, build.nlist].
   */
  SearchConfig NormalizeSearchConfig(
      const IndexConfig& index_config,
      const SearchConfig& search_config = {}) const;

  std::unique_ptr<Index> CreateIndex(const IndexConfig& config) const;

  std::unique_ptr<Index> CreateIndex(std::string_view json) const;

  /** Create the concrete SearchParameters for an index configuration. */
  std::unique_ptr<SearchParameters> CreateSearchParameters(
      const IndexConfig& index_config,
      const SearchConfig& search_config = {}) const;

 private:
  ResourceContext resources_;
};

/** Convenience entry point backed by a default Engine. */
std::unique_ptr<Index> CreateIndex(std::string_view json);

}  // namespace hypervec
