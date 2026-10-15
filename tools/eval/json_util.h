/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <string>
#include <string_view>

namespace hypervec::eval_cli {

inline std::string JsonEscape(std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string escaped;
  for (unsigned char character : value) {
    switch (character) {
      case '\"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (character < 0x20U) {
          escaped += "\\u00";
          escaped.push_back(kHex[character >> 4U]);
          escaped.push_back(kHex[character & 0x0FU]);
        } else {
          escaped.push_back(static_cast<char>(character));
        }
    }
  }
  return escaped;
}

}  // namespace hypervec::eval_cli
