/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "python_callbacks.h"

#include <cstring>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// PyCallbackIOWriter
// ---------------------------------------------------------------------------

PyCallbackIOWriter::PyCallbackIOWriter(PyObject* callback, size_t bs)
    : callback(callback), bs(bs) {
  Py_XINCREF(callback);
}

size_t PyCallbackIOWriter::operator()(const void* ptrv, size_t size,
                                      size_t nitems) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  size_t total = size * nitems;
  size_t written = 0;
  const char* ptr = reinterpret_cast<const char*>(ptrv);

  while (written < total) {
    size_t chunk = std::min(bs, total - written);
    PyObject* bytes = PyBytes_FromStringAndSize(ptr + written, (Py_ssize_t)chunk);
    if (!bytes) {
      PyGILState_Release(gstate);
      throw std::runtime_error("PyCallbackIOWriter: failed to create bytes object");
    }
    PyObject* result = PyObject_CallFunctionObjArgs(callback, bytes, nullptr);
    Py_DECREF(bytes);
    if (!result) {
      PyGILState_Release(gstate);
      throw std::runtime_error("PyCallbackIOWriter: callback raised an exception");
    }
    long n = PyLong_AsLong(result);
    Py_DECREF(result);
    if (n < 0) {
      PyGILState_Release(gstate);
      throw std::runtime_error("PyCallbackIOWriter: callback returned negative value");
    }
    written += (size_t)n;
    if ((size_t)n < chunk) break;  // short write
  }

  PyGILState_Release(gstate);
  return written / size;
}

PyCallbackIOWriter::~PyCallbackIOWriter() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gstate);
}

// ---------------------------------------------------------------------------
// PyCallbackIOReader
// ---------------------------------------------------------------------------

PyCallbackIOReader::PyCallbackIOReader(PyObject* callback, size_t bs)
    : callback(callback), bs(bs) {
  Py_XINCREF(callback);
}

size_t PyCallbackIOReader::operator()(void* ptrv, size_t size, size_t nitems) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  size_t total = size * nitems;
  size_t read_total = 0;
  char* ptr = reinterpret_cast<char*>(ptrv);

  while (read_total < total) {
    size_t chunk = std::min(bs, total - read_total);
    PyObject* size_obj = PyLong_FromSize_t(chunk);
    PyObject* result = PyObject_CallFunctionObjArgs(callback, size_obj, nullptr);
    Py_DECREF(size_obj);
    if (!result) {
      PyGILState_Release(gstate);
      throw std::runtime_error("PyCallbackIOReader: callback raised an exception");
    }
    if (!PyBytes_Check(result)) {
      Py_DECREF(result);
      PyGILState_Release(gstate);
      throw std::runtime_error("PyCallbackIOReader: callback must return bytes");
    }
    Py_ssize_t n = PyBytes_GET_SIZE(result);
    if (n > 0) {
      memcpy(ptr + read_total, PyBytes_AS_STRING(result), (size_t)n);
      read_total += (size_t)n;
    }
    Py_DECREF(result);
    if (n == 0 || (size_t)n < chunk) break;  // EOF or short read
  }

  PyGILState_Release(gstate);
  return read_total / size;
}

PyCallbackIOReader::~PyCallbackIOReader() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gstate);
}

// ---------------------------------------------------------------------------
// PyCallbackIDSelector
// ---------------------------------------------------------------------------

PyCallbackIDSelector::PyCallbackIDSelector(PyObject* callback)
    : callback(callback) {
  Py_XINCREF(callback);
}

bool PyCallbackIDSelector::IsMember(hypervec::idx_t id) const {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* id_obj = PyLong_FromLongLong((long long)id);
  PyObject* result = PyObject_CallFunctionObjArgs(callback, id_obj, nullptr);
  Py_DECREF(id_obj);
  bool member = false;
  if (result) {
    member = PyObject_IsTrue(result);
    Py_DECREF(result);
  }
  PyGILState_Release(gstate);
  return member;
}

PyCallbackIDSelector::~PyCallbackIDSelector() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gstate);
}

// ---------------------------------------------------------------------------
// PyCallbackShardingFunction
// ---------------------------------------------------------------------------

PyCallbackShardingFunction::PyCallbackShardingFunction(PyObject* callback)
    : callback(callback) {
  Py_XINCREF(callback);
}

int64_t PyCallbackShardingFunction::operator()(int64_t i,
                                                int64_t shard_count) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* i_obj = PyLong_FromLongLong(i);
  PyObject* sc_obj = PyLong_FromLongLong(shard_count);
  PyObject* result =
      PyObject_CallFunctionObjArgs(callback, i_obj, sc_obj, nullptr);
  Py_DECREF(i_obj);
  Py_DECREF(sc_obj);
  int64_t shard = 0;
  if (result) {
    shard = (int64_t)PyLong_AsLongLong(result);
    Py_DECREF(result);
  }
  PyGILState_Release(gstate);
  return shard;
}

PyCallbackShardingFunction::~PyCallbackShardingFunction() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gstate);
}
