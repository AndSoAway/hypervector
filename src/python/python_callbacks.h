/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/selector/id_selector.h>
#include <invlists/inverted_lists.h>
#include <persistence/io.h>

#include <cstdint>

#include "Python.h"

namespace hypervec {

/** Sharding function interface for index sharding. */
struct ShardingFunction {
  virtual ~ShardingFunction() = default;
  virtual int64_t operator()(int64_t i, int64_t shard_count) = 0;
};

}  // namespace hypervec

//  all callbacks have to acquire the GIL on input

/***********************************************************
 * Callbacks for IO reader and writer
 ***********************************************************/

struct PyCallbackIOWriter : hypervec::IOWriter {
  PyObject* callback;
  size_t bs;  // maximum write size

  /** Callback: Python function that takes a bytes object and
   *  returns the number of bytes successfully written.
   */
  explicit PyCallbackIOWriter(PyObject* callback, size_t bs = 1024 * 1024);

  size_t operator()(const void* ptrv, size_t size, size_t nitems) override;

  ~PyCallbackIOWriter() override;
};

struct PyCallbackIOReader : hypervec::IOReader {
  PyObject* callback;
  size_t bs;  // maximum buffer size

  /** Callback: Python function that takes a size and returns a
   * bytes object with the resulting read */
  explicit PyCallbackIOReader(PyObject* callback, size_t bs = 1024 * 1024);

  size_t operator()(void* ptrv, size_t size, size_t nitems) override;

  ~PyCallbackIOReader() override;
};

/***********************************************************
 * Callbacks for IDSelector
 ***********************************************************/

struct PyCallbackIDSelector : hypervec::IDSelector {
  PyObject* callback;

  explicit PyCallbackIDSelector(PyObject* callback);

  bool IsMember(hypervec::idx_t id) const override;

  ~PyCallbackIDSelector() override;
};

/***********************************************************
 * Callbacks for index sharding
 ***********************************************************/

struct PyCallbackShardingFunction : hypervec::ShardingFunction {
  PyObject* callback;

  explicit PyCallbackShardingFunction(PyObject* callback);

  int64_t operator()(int64_t i, int64_t shard_count) override;

  ~PyCallbackShardingFunction() override;

  PyCallbackShardingFunction(const PyCallbackShardingFunction&) = delete;
  PyCallbackShardingFunction(PyCallbackShardingFunction&&) noexcept = default;
  PyCallbackShardingFunction& operator=(const PyCallbackShardingFunction&) =
    default;
  PyCallbackShardingFunction& operator=(PyCallbackShardingFunction&&) = default;
};
