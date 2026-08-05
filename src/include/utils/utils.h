/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#ifndef HYPERVEC_utils_h
#define HYPERVEC_utils_h

#include <utils/common/platform_macros.h>

#include <stdint.h>

#include <string>

namespace hypervec {

/// get compile options
std::string get_compile_options();

/// Expose HyperVec version as a string
std::string GetVersion();

/// ms elapsed since some arbitrary epoch
double getmillisecs();

/// Whether OpenMP annotations were respected.
bool check_openmp();

}  // namespace hypervec

#endif /* HYPERVEC_utils_h */
