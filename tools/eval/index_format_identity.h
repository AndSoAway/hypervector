/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace hypervec::eval_cli {

/** Identity of the concrete index format stored in one persisted file.
 *
 * WriteIndex emits a fourcc tag as the first uint32 of the file, and the
 * persistence registry maps every tag back to a registered format name. The
 * search-parameter family returned by DescribeSearchParameters() is much
 * coarser: it collapses the sixteen registered index types into eight labels,
 * so it cannot identify which algorithm produced a measurement. Reading the
 * tag recovers the specific format without adding any API to the persistence
 * layer.
 */
struct IndexFormatIdentity {
  /** Registered persistence name, e.g. "ivf_rabitq"; empty when unknown. */
  std::string format_name;
  std::string tag_printable;  // e.g. "IVRQ"
  bool recognized = false;    // false when no registered format matched
};

/** Read the leading fourcc tag of a persisted index and resolve its name. */
IndexFormatIdentity DescribeIndexFormat(const std::filesystem::path& path);

}  // namespace hypervec::eval_cli
