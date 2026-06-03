/*
 * Minimal SWIG interface for HyperVec HNSW
 * 只包含公共接口，避免impl实现细节
 */

%module swighypervec

%include <stdint.i>
%include <typemaps.i>

/* Tell SWIG how to convert idx_t (int64_t on all platforms) */
%apply long long { hypervec::idx_t };
%apply long long * { hypervec::idx_t * };

/* Allow passing numpy array ctypes.data (Python int = memory address) as float* */
%typemap(in) float const * {
    if (PyLong_Check($input)) {
        $1 = reinterpret_cast<float*>(static_cast<uintptr_t>(PyLong_AsUnsignedLongLong($input)));
    } else {
        SWIG_exception_fail(SWIG_TypeError, "Expected int (ctypes.data address) for float*");
    }
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) float const * {
    $1 = PyLong_Check($input) ? 1 : 0;
}
%typemap(in) float * {
    if (PyLong_Check($input)) {
        $1 = reinterpret_cast<float*>(static_cast<uintptr_t>(PyLong_AsUnsignedLongLong($input)));
    } else {
        SWIG_exception_fail(SWIG_TypeError, "Expected int (ctypes.data address) for float*");
    }
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) float * {
    $1 = PyLong_Check($input) ? 1 : 0;
}

/* Allow passing numpy int64 ctypes.data as idx_t* */
%typemap(in) hypervec::idx_t const * {
    if (PyLong_Check($input)) {
        $1 = reinterpret_cast<hypervec::idx_t*>(static_cast<uintptr_t>(PyLong_AsUnsignedLongLong($input)));
    } else {
        SWIG_exception_fail(SWIG_TypeError, "Expected int (ctypes.data address) for idx_t*");
    }
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_INT64_ARRAY) hypervec::idx_t const * {
    $1 = PyLong_Check($input) ? 1 : 0;
}
%typemap(in) hypervec::idx_t * {
    if (PyLong_Check($input)) {
        $1 = reinterpret_cast<hypervec::idx_t*>(static_cast<uintptr_t>(PyLong_AsUnsignedLongLong($input)));
    } else {
        SWIG_exception_fail(SWIG_TypeError, "Expected int (ctypes.data address) for idx_t*");
    }
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_INT64_ARRAY) hypervec::idx_t * {
    $1 = PyLong_Check($input) ? 1 : 0;
}

%{
#include <Python.h>
#include <index/hnsw/index_hnsw.h>
#include <index/flat/index_flat.h>
#include <index/flat/index_flat_codes.h>
#include <invlists/inverted_lists.h>
#include <persistence/io.h>
#include <persistence/index_io.h>
#include <utils/selector/id_selector.h>
#include <python/python_callbacks.h>
using namespace hypervec;
%}

/* Base types needed by SWIG for inheritance - order matters */
%include <utils/distances/metric_type.h>
%include <utils/selector/id_selector.h>
%include <persistence/io.h>

/* Index base class MUST come before any derived index classes */
%include <index/index.h>

/* InvertedLists and code utilities */
%include <invlists/inverted_lists.h>
%include <index/flat/index_flat_codes.h>

/* Flat Index */
%include <index/flat/index_flat.h>

/* HNSW Index - 只通过公共接口 */
/* Ignore internal implementation methods not needed in Python */
%ignore hypervec::IndexHNSW::ShrinkLevel0Neighbors;
%ignore hypervec::IndexHNSW::SearchLevel0;
%ignore hypervec::IndexHNSW::InitLevel0FromKnngraph;
%ignore hypervec::IndexHNSW::InitLevel0FromEntryPoints;
%ignore hypervec::IndexHNSW::ReorderLinks;
%ignore hypervec::IndexHNSW::LinkSingletons;
%ignore hypervec::IndexHNSW::PermuteEntries;
%ignore hypervec::IndexHNSW::Search1;
%include <index/hnsw/index_hnsw.h>

/* Python callbacks */
%include <python/python_callbacks.h>

/* Index IO functions - manually declare only what we need */
%newobject hypervec::ReadIndex;
namespace hypervec {
  Index* ReadIndex(const char* fname, int io_flags = 0);
  void WriteIndex(const Index* idx, const char* fname, int io_flags = 0);
}

/* Handle Python type stubs */
%pythoncode %{
__all__ = [
    'IndexHNSW',
    'IndexFlat',
    'Index',
    'read_index',
    'write_index',
]
%}