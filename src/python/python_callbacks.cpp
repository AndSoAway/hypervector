/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "python_callbacks.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void throw_python_error(const char* context) {
  if (!PyErr_Occurred()) {
    throw std::runtime_error(context);
  }

  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  std::string message(context);
  if (value) {
    PyObject* text = PyObject_Str(value);
    if (text) {
      const char* utf8 = PyUnicode_AsUTF8(text);
      if (utf8) {
        message += ": ";
        message += utf8;
      }
      Py_DECREF(text);
    }
  }

  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  throw std::runtime_error(message);
}

}  // namespace

PyCallbackIOWriter::PyCallbackIOWriter(PyObject* callback, size_t bs)
  : callback(callback), bs(bs) {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XINCREF(callback);
  PyGILState_Release(gil);
}

size_t PyCallbackIOWriter::operator()(
    const void* ptrv,
    size_t size,
    size_t nitems) {
  const size_t total = size * nitems;
  const char* ptr = static_cast<const char*>(ptrv);
  size_t written = 0;

  PyGILState_STATE gil = PyGILState_Ensure();
  while (written < total) {
    const size_t chunk = std::min(bs, total - written);
    PyObject* bytes = PyBytes_FromStringAndSize(ptr + written, chunk);
    if (!bytes) {
      PyGILState_Release(gil);
      throw_python_error("failed to create Python bytes for write callback");
    }

    PyObject* result = PyObject_CallFunctionObjArgs(callback, bytes, nullptr);
    Py_DECREF(bytes);
    if (!result) {
      PyGILState_Release(gil);
      throw_python_error("Python write callback failed");
    }

    const long long accepted = PyLong_AsLongLong(result);
    Py_DECREF(result);
    if (PyErr_Occurred()) {
      PyGILState_Release(gil);
      throw_python_error("Python write callback returned a non-integer");
    }
    if (accepted <= 0) {
      break;
    }
    written += std::min(static_cast<size_t>(accepted), chunk);
  }
  PyGILState_Release(gil);

  return size == 0 ? 0 : written / size;
}

PyCallbackIOWriter::~PyCallbackIOWriter() {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gil);
}

PyCallbackIOReader::PyCallbackIOReader(PyObject* callback, size_t bs)
  : callback(callback), bs(bs) {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XINCREF(callback);
  PyGILState_Release(gil);
}

size_t PyCallbackIOReader::operator()(void* ptrv, size_t size, size_t nitems) {
  const size_t total = size * nitems;
  char* ptr = static_cast<char*>(ptrv);
  size_t read = 0;

  PyGILState_STATE gil = PyGILState_Ensure();
  while (read < total) {
    const size_t requested = std::min(bs, total - read);
    PyObject* result = PyObject_CallFunction(callback, "n", requested);
    if (!result) {
      PyGILState_Release(gil);
      throw_python_error("Python read callback failed");
    }

    char* buffer = nullptr;
    Py_ssize_t length = 0;
    if (PyBytes_AsStringAndSize(result, &buffer, &length) < 0) {
      Py_DECREF(result);
      PyGILState_Release(gil);
      throw_python_error("Python read callback returned non-bytes data");
    }
    if (length <= 0) {
      Py_DECREF(result);
      break;
    }

    const size_t chunk = std::min(static_cast<size_t>(length), requested);
    std::memcpy(ptr + read, buffer, chunk);
    read += chunk;
    Py_DECREF(result);
    if (chunk < requested) {
      break;
    }
  }
  PyGILState_Release(gil);

  return size == 0 ? 0 : read / size;
}

PyCallbackIOReader::~PyCallbackIOReader() {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gil);
}

PyCallbackIDSelector::PyCallbackIDSelector(PyObject* callback)
  : callback(callback) {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XINCREF(callback);
  PyGILState_Release(gil);
}

bool PyCallbackIDSelector::IsMember(hypervec::idx_t id) const {
  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* result = PyObject_CallFunction(callback, "L", static_cast<long long>(id));
  if (!result) {
    PyGILState_Release(gil);
    throw_python_error("Python ID selector callback failed");
  }
  const int selected = PyObject_IsTrue(result);
  Py_DECREF(result);
  PyGILState_Release(gil);
  if (selected < 0) {
    throw_python_error("Python ID selector callback returned invalid truth value");
  }
  return selected != 0;
}

PyCallbackIDSelector::~PyCallbackIDSelector() {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gil);
}

PyCallbackShardingFunction::PyCallbackShardingFunction(PyObject* callback)
  : callback(callback) {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XINCREF(callback);
  PyGILState_Release(gil);
}

int64_t PyCallbackShardingFunction::operator()(int64_t i, int64_t shard_count) {
  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* result = PyObject_CallFunction(callback, "LL",
                                          static_cast<long long>(i),
                                          static_cast<long long>(shard_count));
  if (!result) {
    PyGILState_Release(gil);
    throw_python_error("Python sharding callback failed");
  }
  const long long shard = PyLong_AsLongLong(result);
  Py_DECREF(result);
  PyGILState_Release(gil);
  if (PyErr_Occurred()) {
    throw_python_error("Python sharding callback returned a non-integer");
  }
  return static_cast<int64_t>(shard);
}

PyCallbackShardingFunction::~PyCallbackShardingFunction() {
  PyGILState_STATE gil = PyGILState_Ensure();
  Py_XDECREF(callback);
  PyGILState_Release(gil);
}
