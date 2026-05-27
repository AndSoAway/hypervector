/*
 * Minimal SWIG interface for HyperVec HNSW
 * 只包含公共接口，避免impl实现细节
 */

%module swighypervec

%{
#include <Python.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/flat/index_flat.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/index_ivflvq.h>
#include <persistence/index_io.h>
%}

/* Index base class */
%include <core/Index.h>
%include <core/MetricType.h>

/* Flat Index */
%include <index/flat/index_flat.h>

/* HNSW Index - 只通过公共接口 */
%include <index/hnsw/index_hnsw.h>
%include <index/hnsw/index_hnsw_lvq.h>

/* LVQ Indexes */
%include <quantization/lvq/lvq.h>
%include <quantization/lvq/index_lvq.h>
%include <quantization/lvq/index_ivflvq.h>

/* Index IO */
%include <persistence/index_io.h>

/* Python callbacks */
%include <python/python_callbacks.h>

/* Handle Python type stubs */
%pythoncode %{
__all__ = [
    'IndexHNSW',
    'IndexHNSWLVQ',
    'IndexLVQ',
    'IndexIVFLVQ',
    'HyperVectorIndexBackend',
    'IndexFlat',
    'Index',
    'read_index',
    'write_index',
]
%}
