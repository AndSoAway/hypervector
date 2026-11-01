/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "index_format_identity.h"

#include <persistence/index_io_registry.h>

#include <algorithm>
#include <fstream>

namespace hypervec::eval_cli {
namespace {

std::string PrintableTag(uint32_t tag) {
  std::string text;
  for (unsigned int offset = 0; offset < 4U; ++offset) {
    const char byte = static_cast<char>((tag >> (8U * offset)) & 0xFFU);
    // A fourcc is built from printable ASCII; anything else is rendered as a
    // space so a corrupt tag still produces a stable, non-injecting string.
    text.push_back(byte >= 0x20 && byte < 0x7F ? byte : ' ');
  }
  return text;
}

bool ReadLeadingTag(const std::filesystem::path& path, uint32_t* tag) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input) {
    return false;
  }
  uint32_t stored = 0;
  input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(stored))) {
    return false;
  }
  *tag = stored;
  return true;
}

}  // namespace

IndexFormatIdentity DescribeIndexFormat(const std::filesystem::path& path) {
  IndexFormatIdentity identity;
  uint32_t tag = 0;
  if (!ReadLeadingTag(path, &tag)) {
    return identity;
  }
  identity.tag_printable = PrintableTag(tag);
  for (const hypervec::IndexIODescriptor& descriptor :
       hypervec::GetIndexIORegistry().List()) {
    if (descriptor.write_tag == tag ||
        std::find(descriptor.read_tags.begin(), descriptor.read_tags.end(),
                  tag) != descriptor.read_tags.end()) {
      identity.format_name = descriptor.name;
      identity.recognized = true;
      return identity;
    }
  }
  return identity;
}

}  // namespace hypervec::eval_cli
