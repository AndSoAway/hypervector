/*
 * Minimal SWIG interface for HyperVec HNSW
 * 只包含公共接口，避免impl实现细节
 */

%module swighypervec

%{
#include <Python.h>
#include <index/hnsw/index_hnsw.h>
#include <index/flat/index_flat.h>
%}

/* Index base class */
%include <core/Index.h>
%include <core/MetricType.h>

/* Flat Index */
%include <index/flat/index_flat.h>

/* HNSW Index - 只通过公共接口 */
%include <index/hnsw/index_hnsw.h>

/* Python callbacks */
%include <python/python_callbacks.h>

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