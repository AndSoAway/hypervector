/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hypervec {

/** Typed, named runtime options used to construct SearchParameters.
 *
 * Only integers and booleans are currently needed by built-in indexes. Unknown
 * names and type mismatches are rejected by CreateSearchParameters().
 */
class SearchConfig {
 public:
  SearchConfig() = default;

  /** Optional selector remains owned by the caller. */
  IDSelector* selector = nullptr;

  SearchConfig& SetInteger(std::string name, int64_t value);
  SearchConfig& SetBoolean(std::string name, bool value);

  bool HasParameter(std::string_view name) const;
  int64_t GetInteger(std::string_view name, int64_t default_value) const;
  bool GetBoolean(std::string_view name, bool default_value) const;
  std::vector<std::string> ParameterNames() const;

 private:
  enum class ValueType { kInteger, kBoolean };

  struct Value {
    ValueType type = ValueType::kInteger;
    int64_t integer = 0;
    bool boolean = false;
  };

  const Value* Find(std::string_view name) const;
  void Set(std::string name, Value value);

  std::vector<std::pair<std::string, Value>> parameters_;
};

/** Runtime parameter surface exposed by the resolved underlying index. */
struct SearchParameterDescriptor {
  std::string name;
  std::vector<std::string> parameter_names;
};

/** Describe the parameters accepted for an index, unwrapping decorators. */
SearchParameterDescriptor DescribeSearchParameters(const Index& index);

/** Build the correct SearchParameters subtype for an index.
 *
 * IndexIDMap and IndexPreTransform are unwrapped only for type discovery; the
 * returned selector still uses the IDs accepted by the outer index.
 */
std::unique_ptr<SearchParameters> CreateSearchParameters(
    const Index& index, const SearchConfig& config = {});

}  // namespace hypervec
