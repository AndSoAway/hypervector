/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#ifndef HYPERVEC_EXCEPTION_INCLUDED
#define HYPERVEC_EXCEPTION_INCLUDED

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace hypervec {

/// Base class for HyperVec exceptions
class HypervecException : public std::exception {
 public:
  explicit HypervecException(const std::string& msg);

  HypervecException(const std::string& msg, const char* funcName,
                    const char* file, int line);

  /// from std::exception
  const char* what() const noexcept override;

  std::string msg;
};

/// Handle multiple exceptions from worker threads, throwing an appropriate
/// exception that aggregates the information
/// The pair int is the thread that generated the exception
void handleExceptions(
  std::vector<std::pair<int, std::exception_ptr>>& exceptions);

/** RAII object for a set of possibly transformed vectors (deallocated only if
 * they are indeed transformed)
 */
struct TransformedVectors {
  const float* x;
  bool own_x;
  TransformedVectors(const float* x_orig, const float* x) : x(x) {
    own_x = x_orig != x;
  }

  ~TransformedVectors() {
    if (own_x) {
      delete[] x;
    }
  }
};

/// make typeids more readable
std::string demangle_cpp_symbol(const char* name);

}  // namespace hypervec

#endif
