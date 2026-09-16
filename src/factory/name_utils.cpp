/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "factory/name_utils.h"

#include <utils/log/exception.h>

#include <cstddef>
#include <string>

namespace hypervec::factory_detail {

namespace {

bool IsAsciiAlphaNumeric(unsigned char ch) {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
         (ch >= '0' && ch <= '9');
}

}  // namespace

std::string NormalizeToken(std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const unsigned char ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      normalized.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else if (ch == '-') {
      normalized.push_back('_');
    } else {
      normalized.push_back(static_cast<char>(ch));
    }
  }
  return normalized;
}

std::string NormalizeName(std::string_view value, std::string_view context,
                          bool allow_empty) {
  if (value.empty()) {
    if (allow_empty) {
      return {};
    }
    throw HypervecException(std::string(context) + " must not be empty");
  }

  std::string normalized;
  normalized.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    const unsigned char ch = value[i];
    if (!IsAsciiAlphaNumeric(ch) && ch != '_' && ch != '-') {
      throw HypervecException("invalid character at byte " + std::to_string(i) +
                              " in " + std::string(context));
    }
    if (ch >= 'A' && ch <= 'Z') {
      normalized.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else if (ch == '-') {
      normalized.push_back('_');
    } else {
      normalized.push_back(static_cast<char>(ch));
    }
  }
  return normalized;
}

}  // namespace hypervec::factory_detail
