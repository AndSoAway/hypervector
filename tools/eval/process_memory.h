/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace hypervec::eval_cli {

/** Process-wide peak resident set size, in bytes, when supported.
 *
 * This is a high-water mark for the whole process, so it also counts the
 * query set, the ground truth and any exact rerank base that were loaded. It
 * is not an attribute of the index alone, and it never decreases.
 */
std::optional<uint64_t> PeakResidentSetBytes();

/** Current resident set size, in bytes, when supported.
 *
 * Paired with PeakResidentSetBytes, a before/after pair around one load step
 * attributes memory to that step. Current usage does fall when memory is
 * returned to the operating system, so a delta can understate a transient
 * high-water mark.
 */
std::optional<uint64_t> CurrentResidentSetBytes();

/** Operating-system source used by PeakResidentSetBytes. */
std::string_view PeakResidentSetSource();

/** Operating-system source used by CurrentResidentSetBytes. */
std::string_view CurrentResidentSetSource();

/** Parallelism the measured work was actually allowed to use.
 *
 * Build duration and query throughput are OpenMP-sensitive, and several index
 * types expose no build_threads option at all, so the effective thread count
 * comes from the environment. A report that omits it cannot be compared
 * across machines, so both numbers are recorded with every measurement.
 */
struct ExecutionEnvironment {
  /** omp_get_max_threads(): the team size an implicit region would use. */
  int omp_max_threads = 0;
  /** std::thread::hardware_concurrency(), zero when unknown. */
  unsigned int hardware_concurrency = 0;
};

/** Read the OpenMP and hardware parallelism limits of the current process. */
ExecutionEnvironment CurrentExecutionEnvironment();

}  // namespace hypervec::eval_cli
