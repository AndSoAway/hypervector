/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#include <utils/utils.h>

#include <index/index.h>
#include <utils/simd/simd_levels.h>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <sys/time.h>
#endif

#include <omp.h>

#include <vector>

namespace hypervec {

std::string get_compile_options() {
  std::string options;

#ifdef __OPTIMIZE__
  options += "OPTIMIZE ";
#endif

#ifdef HYPERVEC_ENABLE_DD
  options += "DD ";
  int supported = SIMDConfig::supported_simd_levels;
  for (int i = 0; i < static_cast<int>(SIMDLevel::COUNT); ++i) {
    auto level = static_cast<SIMDLevel>(i);
    if ((supported & (1 << i)) && level != SIMDLevel::NONE) {
      options += to_string(level) + " ";
    }
  }
#else
  SIMDLevel level = SIMDConfig::get_level();
  if (level != SIMDLevel::NONE) {
    options += to_string(level) + " ";
  }
#endif

#ifdef HYPERVEC_ENABLE_SVS
  options += "SVS ";
#endif

  return options;
}

std::string GetVersion() {
  return VERSION_STRING;
}

#ifdef _MSC_VER
double getmillisecs() {
  LARGE_INTEGER ts;
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&ts);
  return (ts.QuadPart * 1e3) / freq.QuadPart;
}
#else
double getmillisecs() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
#endif

bool check_openmp() {
  omp_set_num_threads(10);

  if (omp_get_max_threads() != 10) {
    return false;
  }

  std::vector<int> nt_per_thread(10);
  size_t sum = 0;
  bool in_parallel = true;
#pragma omp parallel reduction(+ : sum)
  {
    if (!omp_in_parallel()) {
      in_parallel = false;
    }

    int nt = omp_get_num_threads();
    int rank = omp_get_thread_num();

    nt_per_thread[rank] = nt;
#pragma omp for
    for (int i = 0; i < 1000 * 1000 * 10; i++) {
      sum += i;
    }
  }

  if (!in_parallel) {
    return false;
  }
  if (nt_per_thread[0] != 10) {
    return false;
  }
  if (sum == 0) {
    return false;
  }

  return true;
}

}  // namespace hypervec
