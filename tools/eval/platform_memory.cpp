/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "eval/platform_memory.h"

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>

#include <fstream>
#endif

namespace hypervec::eval {

std::uint64_t GetCurrentRssBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) ==
      0) {
    return 0;
  }
  return static_cast<std::uint64_t>(counters.WorkingSetSize);
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::uint64_t>(info.resident_size);
#else
  std::ifstream statm("/proc/self/statm");
  std::uint64_t total_pages = 0;
  std::uint64_t resident_pages = 0;
  if (!(statm >> total_pages >> resident_pages)) {
    return 0;
  }
  static_cast<void>(total_pages);
  const std::int64_t page_size =
      static_cast<std::int64_t>(sysconf(_SC_PAGESIZE));
  if (page_size <= 0) {
    return 0;
  }
  return resident_pages * static_cast<std::uint64_t>(page_size);
#endif
}

}  // namespace hypervec::eval
