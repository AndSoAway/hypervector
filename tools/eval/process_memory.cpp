/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "process_memory.h"

#include <omp.h>

#include <limits>
#include <thread>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#if defined(__linux__)
#include <unistd.h>

#include <fstream>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace hypervec::eval_cli {

std::optional<uint64_t> PeakResidentSetBytes() {
#if defined(__linux__) || defined(__APPLE__)
  struct rusage usage{};
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

std::optional<uint64_t> CurrentResidentSetBytes() {
#if defined(__linux__)
  // /proc/self/statm reports sizes in pages: total, resident, shared, ...
  std::ifstream statm("/proc/self/statm");
  uint64_t total_pages = 0;
  uint64_t resident_pages = 0;
  if (!(statm >> total_pages >> resident_pages)) {
    return std::nullopt;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return std::nullopt;
  }
  const uint64_t pages_per_unit = static_cast<uint64_t>(page_size);
  if (resident_pages >
      (std::numeric_limits<uint64_t>::max)() / pages_per_unit) {
    return std::nullopt;
  }
  return resident_pages * pages_per_unit;
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(info.resident_size);
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

std::string_view CurrentResidentSetSource() {
#if defined(__linux__)
  return "/proc/self/statm";
#elif defined(__APPLE__)
  return "mach_task_basic_info.resident_size";
#else
  return "unsupported";
#endif
}

ExecutionEnvironment CurrentExecutionEnvironment() {
  ExecutionEnvironment environment;
  environment.omp_max_threads = omp_get_max_threads();
  environment.hardware_concurrency = std::thread::hardware_concurrency();
  return environment;
}

}  // namespace hypervec::eval_cli
