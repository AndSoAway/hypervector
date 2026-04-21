/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/structures/maybe_owned_vector.h>

/*************************************************************
 * I/O macros
 *
 * we use macros so that we have a line number to report in abort
 * (). This makes debugging a lot easier. The IOReader or IOWriter is
 * always called f and thus is not passed in as a macro parameter.
 **************************************************************/

namespace hypervec {
size_t get_deserialization_vector_byte_limit();
}  // namespace hypervec

#define READANDCHECK(ptr, n)                                                   \
  {                                                                            \
    size_t ret = (*f)(ptr, sizeof(*(ptr)), n);                                 \
    HYPERVEC_THROW_IF_NOT_FMT(ret == (n), "read error in %s: %zd != %zd (%s)", \
                              f->name.c_str(), ret, size_t(n),                 \
                              strerror(errno));                                \
  }

#define READ1(x) READANDCHECK(&(x), 1)

#define READ1_DUMMY(x_type) \
  {                         \
    x_type x = {};          \
    READ1(x);               \
  }

// Rejects vectors whose total allocation would exceed the configurable
// byte limit (default 1 TB).
#define READVECTOR(vec)                                                        \
  {                                                                            \
    size_t size;                                                               \
    READANDCHECK(&size, 1);                                                    \
    HYPERVEC_THROW_IF_NOT(                                                     \
      size >= 0 && size < (hypervec::get_deserialization_vector_byte_limit() / \
                           sizeof(*(vec).data())));                            \
    (vec).resize(size);                                                        \
    READANDCHECK((vec).data(), size);                                          \
  }

#define WRITEANDCHECK(ptr, n)                                                 \
  {                                                                           \
    size_t ret = (*f)(ptr, sizeof(*(ptr)), n);                                \
    HYPERVEC_THROW_IF_NOT_FMT(                                                \
      ret == (n), "write error in %s: %zd != %zd (%s)", f->name.c_str(), ret, \
      size_t(n), strerror(errno));                                            \
  }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)

#define WRITEVECTOR(vec)               \
  {                                    \
    size_t size = (vec).size();        \
    WRITEANDCHECK(&size, 1);           \
    WRITEANDCHECK((vec).data(), size); \
  }

// read/write xb vector for backwards compatibility of IndexFlat

#define WRITEXBVECTOR(vec)                        \
  {                                               \
    HYPERVEC_THROW_IF_NOT((vec).size() % 4 == 0); \
    size_t size = (vec).size() / 4;               \
    WRITEANDCHECK(&size, 1);                      \
    WRITEANDCHECK((vec).data(), size * 4);        \
  }

#define READXBVECTOR(vec)                                                      \
  {                                                                            \
    size_t size;                                                               \
    READANDCHECK(&size, 1);                                                    \
    HYPERVEC_THROW_IF_NOT(                                                     \
      size >= 0 && size < (hypervec::get_deserialization_vector_byte_limit() / \
                           (4 * sizeof(*(vec).data()))));                      \
    size *= 4;                                                                 \
    (vec).resize(size);                                                        \
    READANDCHECK((vec).data(), size);                                          \
  }
