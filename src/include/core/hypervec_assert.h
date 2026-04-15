/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#ifndef HYPERVEC_ASSERT_INCLUDED
#define HYPERVEC_ASSERT_INCLUDED

#include <core/hypervec_exception.h>
#include <core/platform_macros.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>

///
/// Assertions
///

#define HYPERVEC_ASSERT(X)                                  \
  do {                                                      \
    if (!(X)) {                                             \
      fprintf(stderr,                                       \
              "HyperVec assertion '%s' failed in %s "       \
              "at %s:%d\n",                                 \
              #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
      abort();                                              \
    }                                                       \
  } while (false)

#define HYPERVEC_ASSERT_MSG(X, MSG)                         \
  do {                                                      \
    if (!(X)) {                                             \
      fprintf(stderr,                                       \
              "HyperVec assertion '%s' failed in %s "       \
              "at %s:%d; details: " MSG "\n",               \
              #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
      abort();                                              \
    }                                                       \
  } while (false)

#define HYPERVEC_ASSERT_FMT(X, FMT, ...)                                 \
  do {                                                                   \
    if (!(X)) {                                                          \
      fprintf(stderr,                                                    \
              "HyperVec assertion '%s' failed in %s "                    \
              "at %s:%d; details: " FMT "\n",                            \
              #X, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); \
      abort();                                                           \
    }                                                                    \
  } while (false)

///
/// Exceptions for returning user errors
///

#define HYPERVEC_THROW_MSG(MSG)                                             \
  do {                                                                      \
    throw ::hypervec::HypervecException(MSG, __PRETTY_FUNCTION__, __FILE__, \
                                        __LINE__);                          \
  } while (false)

#define HYPERVEC_THROW_FMT(FMT, ...)                                        \
  do {                                                                      \
    std::string __s;                                                        \
    int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__);                    \
    __s.resize(__size + 1);                                                 \
    snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__);                        \
    throw ::hypervec::HypervecException(__s, __PRETTY_FUNCTION__, __FILE__, \
                                        __LINE__);                          \
  } while (false)

///
/// Exceptions thrown upon a conditional failure
///

#define HYPERVEC_THROW_IF_NOT(X)                    \
  do {                                              \
    if (!(X)) {                                     \
      HYPERVEC_THROW_FMT("Error: '%s' failed", #X); \
    }                                               \
  } while (false)

#define HYPERVEC_THROW_IF_MSG(X, MSG)                     \
  do {                                                    \
    if (X) {                                              \
      HYPERVEC_THROW_FMT("Error: '%s' failed: " MSG, #X); \
    }                                                     \
  } while (false)

#define HYPERVEC_THROW_IF_NOT_MSG(X, MSG) HYPERVEC_THROW_IF_MSG(!(X), MSG)

#define HYPERVEC_THROW_IF_NOT_FMT(X, FMT, ...)                         \
  do {                                                                 \
    if (!(X)) {                                                        \
      HYPERVEC_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
    }                                                                  \
  } while (false)

///
/// Safe arithmetic
///

#include <limits>

namespace hypervec {

/// Multiplication that throws on overflow instead of wrapping
inline size_t mul_no_overflow(size_t a, size_t b, const char* context) {
  if (a != 0 && b > (std::numeric_limits<size_t>::max)() / a) {
    HYPERVEC_THROW_FMT("integer overflow in %s: %zu * %zu", context, a, b);
  }
  return a * b;
}

/// Addition that throws on overflow instead of wrapping
inline size_t add_no_overflow(size_t a, size_t b, const char* context) {
  if (a > (std::numeric_limits<size_t>::max)() - b) {
    HYPERVEC_THROW_FMT("integer overflow in %s: %zu + %zu", context, a, b);
  }
  return a + b;
}

}  // namespace hypervec

///
/// Bounds checking
///

/// Check that val is in half-open range [lo, hi). Throws with the
/// stringified expression, its value, and the bounds on failure.
#define HYPERVEC_CHECK_RANGE(val, lo, hi)                               \
  HYPERVEC_THROW_IF_NOT_FMT(                                            \
    (int64_t)(val) >= (int64_t)(lo) && (int64_t)(val) < (int64_t)(hi),  \
    "%s (= %" PRId64 ") out of range [%" PRId64 ", %" PRId64 ")", #val, \
    (int64_t)(val), (int64_t)(lo), (int64_t)(hi))

/// Debug-only variant of HYPERVEC_CHECK_RANGE. Aborts in debug builds
/// (when NDEBUG is not defined); compiled out in release builds to
/// avoid overhead in hot paths.
#ifndef NDEBUG
#define HYPERVEC_CHECK_RANGE_DEBUG(val, lo, hi)                         \
  HYPERVEC_ASSERT_FMT(                                                  \
    (int64_t)(val) >= (int64_t)(lo) && (int64_t)(val) < (int64_t)(hi),  \
    "%s (= %" PRId64 ") out of range [%" PRId64 ", %" PRId64 ")", #val, \
    (int64_t)(val), (int64_t)(lo), (int64_t)(hi))
#else
#define HYPERVEC_CHECK_RANGE_DEBUG(val, lo, hi) ((void)0)
#endif

#endif
