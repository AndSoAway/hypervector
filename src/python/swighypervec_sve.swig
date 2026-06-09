/*
 * Minimal SWIG interface for HyperVec HNSW
 * 只包含公共接口，避免impl实现细节
 */

%module swighypervec

%include <stdint.i>

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

/* LVQ indexes and filename persistence helpers used by the HTTP server. */
namespace hypervec {
struct IndexLVQ : Index {
  IndexLVQ(int64_t d, int64_t nlocal, int nbits, MetricType metric = kMetricL2);
};

struct IndexIVFLVQ : Index {
  IndexIVFLVQ(int64_t d, int64_t nlist, int64_t nlocal, int nbits,
              MetricType metric = kMetricL2);
};

struct IndexHNSWLVQ : IndexHNSW {
  IndexHNSWLVQ(int d, int nlocal, int nbits, int M_hnsw,
               MetricType metric = kMetricL2);
};

%rename(write_index) WriteIndex;
%rename(read_index) ReadIndex;
void WriteIndex(const Index* idx, const char* fname, int io_flags = 0);
Index* ReadIndex(const char* fname, int io_flags = 0);
}

/* Python callbacks */
%include <python/python_callbacks.h>
