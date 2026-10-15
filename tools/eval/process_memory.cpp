/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "process_memory.h"

#include <limits>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace hypervec::eval_cli {

std::optional<uint64_t> PeakResidentSetBytes() {
#if defined(__linux__) || defined(__APPLE__)
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
    return std::nullopt;
  }
  const uint64_t resident_set = static_cast<uint64_t>(usage.ru_maxrss);
#if defined(__linux__)
  constexpr uint64_t kBytesPerUnit = 1024;
  if (resident_set > (std::numeric_limits<uint64_t>::max)() / kBytesPerUnit) {
    return std::nullopt;
  }
  return resident_set * kBytesPerUnit;
#else
  return resident_set;
#endif
#else
  return std::nullopt;
#endif
}

std::string_view PeakResidentSetSource() {
#if defined(__linux__) || defined(__APPLE__)
  return "getrusage.ru_maxrss";
#else
  return "unsupported";
#endif
}

}  // namespace hypervec::eval_cli
