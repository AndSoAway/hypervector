/*
 * Minimal SWIG interface used by the HyperVec HTTP server.
 */

%module swighypervec

%include <stdint.i>

%{
#include <Python.h>
#include <index/index.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/index_ivflvq.h>
#include <persistence/index_io.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void read_float32_matrix(PyObject* obj, hypervec::idx_t expected_d,
                         const float** data, hypervec::idx_t* n,
                         hypervec::idx_t* d, Py_buffer* view) {
  if (PyObject_GetBuffer(obj, view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
    throw std::runtime_error("expected a 2-D float32 buffer");
  }

  try {
    if (view->ndim != 2) {
      throw std::runtime_error("expected a 2-D float32 matrix");
    }
    if (view->itemsize != static_cast<Py_ssize_t>(sizeof(float)) ||
        view->format == nullptr || std::strcmp(view->format, "f") != 0) {
      throw std::runtime_error("expected dtype float32");
    }
    if (view->strides[1] != static_cast<Py_ssize_t>(sizeof(float)) ||
        view->strides[0] != view->shape[1] * static_cast<Py_ssize_t>(sizeof(float))) {
      throw std::runtime_error("expected C-contiguous matrix");
    }
    *n = static_cast<hypervec::idx_t>(view->shape[0]);
    *d = static_cast<hypervec::idx_t>(view->shape[1]);
    if (expected_d >= 0 && *d != expected_d) {
      throw std::runtime_error("matrix dimension does not match index dimension");
    }
    *data = static_cast<const float*>(view->buf);
  } catch (...) {
    PyBuffer_Release(view);
    throw;
  }
}

void index_train(hypervec::Index* index, PyObject* obj) {
  const float* data = nullptr;
  hypervec::idx_t n = 0;
  hypervec::idx_t d = 0;
  Py_buffer view;
  read_float32_matrix(obj, index->d, &data, &n, &d, &view);
  index->Train(n, data);
  PyBuffer_Release(&view);
}

void index_add(hypervec::Index* index, PyObject* obj) {
  const float* data = nullptr;
  hypervec::idx_t n = 0;
  hypervec::idx_t d = 0;
  Py_buffer view;
  read_float32_matrix(obj, index->d, &data, &n, &d, &view);
  index->Add(n, data);
  PyBuffer_Release(&view);
}

PyObject* index_search(const hypervec::Index* index, PyObject* obj,
                       hypervec::idx_t k) {
  if (k <= 0) {
    throw std::runtime_error("k must be positive");
  }

  const float* data = nullptr;
  hypervec::idx_t n = 0;
  hypervec::idx_t d = 0;
  Py_buffer view;
  read_float32_matrix(obj, index->d, &data, &n, &d, &view);

  std::vector<float> distances(static_cast<size_t>(n * k));
  std::vector<hypervec::idx_t> labels(static_cast<size_t>(n * k));
  index->Search(n, data, k, distances.data(), labels.data());
  PyBuffer_Release(&view);

  PyObject* py_distances = PyList_New(n);
  PyObject* py_labels = PyList_New(n);
  if (!py_distances || !py_labels) {
    Py_XDECREF(py_distances);
    Py_XDECREF(py_labels);
    throw std::runtime_error("failed to allocate Python search result");
  }

  for (hypervec::idx_t i = 0; i < n; ++i) {
    PyObject* row_d = PyList_New(k);
    PyObject* row_l = PyList_New(k);
    if (!row_d || !row_l) {
      Py_XDECREF(row_d);
      Py_XDECREF(row_l);
      Py_DECREF(py_distances);
      Py_DECREF(py_labels);
      throw std::runtime_error("failed to allocate Python search row");
    }
    for (hypervec::idx_t j = 0; j < k; ++j) {
      const size_t offset = static_cast<size_t>(i * k + j);
      PyList_SET_ITEM(row_d, j, PyFloat_FromDouble(distances[offset]));
      PyList_SET_ITEM(row_l, j, PyLong_FromLongLong(labels[offset]));
    }
    PyList_SET_ITEM(py_distances, i, row_d);
    PyList_SET_ITEM(py_labels, i, row_l);
  }

  PyObject* result = PyTuple_New(2);
  PyTuple_SET_ITEM(result, 0, py_distances);
  PyTuple_SET_ITEM(result, 1, py_labels);
  return result;
}

}  // namespace
%}

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception_fail(SWIG_RuntimeError, e.what());
  }
}

namespace hypervec {

typedef int64_t idx_t;

enum MetricType {
  kMetricInnerProduct,
  kMetricL2,
  kMetricL1,
  kMetricLinf,
  kMetricLp,
  kMetricCanberra = 20,
  kMetricBrayCurtis,
  kMetricJensenShannon,
  kMetricJaccard,
  kMetricNaNEuclidean,
  kMetricGower,
};

%nodefaultctor Index;
%nodefaultdtor Index;
struct Index {
  idx_t d;
  idx_t n_total;
  bool is_trained;
  MetricType metric_type;
  virtual ~Index();
};

struct IndexFlatIP : Index {
  explicit IndexFlatIP(idx_t d);
};

struct IndexFlatL2 : Index {
  explicit IndexFlatL2(idx_t d);
};

struct IndexHNSWFlat : Index {
  IndexHNSWFlat(int d, int M, MetricType metric = kMetricL2);
};

struct IndexLVQ : Index {
  IndexLVQ(idx_t d, idx_t nlocal, int nbits, MetricType metric = kMetricL2);
};

struct IndexIVFLVQ : Index {
  IndexIVFLVQ(idx_t d, idx_t nlist, idx_t nlocal, int nbits,
              MetricType metric = kMetricL2);
};

struct IndexHNSWLVQ : Index {
  IndexHNSWLVQ(int d, int nlocal, int nbits, int M_hnsw,
               MetricType metric = kMetricL2);
};

void WriteIndex(const Index* idx, const char* fname, int io_flags = 0);
Index* ReadIndex(const char* fname, int io_flags = 0);

%extend Index {
  void train(PyObject* x) {
    index_train($self, x);
  }

  void add(PyObject* x) {
    index_add($self, x);
  }

  PyObject* search(PyObject* x, hypervec::idx_t k) {
    return index_search($self, x, k);
  }
}

}  // namespace hypervec

%pythoncode %{
read_index = ReadIndex
write_index = WriteIndex
%}
