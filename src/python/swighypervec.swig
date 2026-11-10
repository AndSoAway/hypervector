/*
 * Minimal SWIG interface used by the HyperVec HTTP server.
 */

%module swighypervec

%include <stdint.i>
%include <std_string.i>
%include <std_vector.i>

%{
#include <Python.h>
#include <index/index.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/kdtree/index_kdtree.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/index_ivflvq.h>
#include <index/refine/index_refine.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <utils/structures/sparse_row.h>
#include <utils/structures/term_dictionary.h>
#include <utils/algo/sparse_vector/sparse_inverted_index.h>
#include <utils/algo/bm25/inverted_index.h>
#include <utils/algo/bm25/text_to_sparse.h>
#include <utils/algo/bm25/analyzer.h>

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

template <typename Callable>
void call_without_python_gil(Callable callable) {
  PyThreadState* thread_state = PyEval_SaveThread();
  try {
    callable();
  } catch (...) {
    PyEval_RestoreThread(thread_state);
    throw;
  }
  PyEval_RestoreThread(thread_state);
}

void index_train(hypervec::Index* index, PyObject* obj) {
  const float* data = nullptr;
  hypervec::idx_t n = 0;
  hypervec::idx_t d = 0;
  Py_buffer view;
  read_float32_matrix(obj, index->d, &data, &n, &d, &view);
  try {
    call_without_python_gil([&] { index->Train(n, data); });
  } catch (...) {
    PyBuffer_Release(&view);
    throw;
  }
  PyBuffer_Release(&view);
}

void index_add(hypervec::Index* index, PyObject* obj) {
  const float* data = nullptr;
  hypervec::idx_t n = 0;
  hypervec::idx_t d = 0;
  Py_buffer view;
  read_float32_matrix(obj, index->d, &data, &n, &d, &view);
  try {
    call_without_python_gil([&] { index->Add(n, data); });
  } catch (...) {
    PyBuffer_Release(&view);
    throw;
  }
  PyBuffer_Release(&view);
}

PyObject* index_search_with_params(const hypervec::Index* index, PyObject* obj,
                                   hypervec::idx_t k,
                                   const hypervec::SearchParameters* params) {
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
  try {
    call_without_python_gil([&] {
      index->Search(n, data, k, distances.data(), labels.data(), params);
    });
  } catch (...) {
    PyBuffer_Release(&view);
    throw;
  }
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

// Build a SparseRow from a Python dict {index: value}.  Indices must be
// non-negative integers that fit in uint32; values are coerced to float.
hypervec::SparseRow sparse_row_from_dict(PyObject* obj) {
  if (!PyDict_Check(obj)) {
    throw std::runtime_error("expected a dict {index: value}");
  }
  std::vector<uint32_t> indices;
  std::vector<float> values;
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    long long idx = PyLong_AsLongLong(key);
    if (idx == -1 && PyErr_Occurred()) {
      throw std::runtime_error("sparse index must be an integer");
    }
    if (idx < 0 || idx > 0xFFFFFFFFLL) {
      throw std::runtime_error("sparse index out of uint32 range");
    }
    double val = PyFloat_AsDouble(value);
    if (val == -1.0 && PyErr_Occurred()) {
      throw std::runtime_error("sparse value must be a number");
    }
    indices.push_back(static_cast<uint32_t>(idx));
    values.push_back(static_cast<float>(val));
  }
  return hypervec::SparseRow(std::move(indices), std::move(values));
}

// Serialize a SparseRow to a Python bytes object using the shared byte layout.
PyObject* sparse_row_serialize(const hypervec::SparseRow* row) {
  hypervec::VectorIOWriter wr;
  hypervec::write_sparse_row(*row, &wr);
  return PyBytes_FromStringAndSize(
    reinterpret_cast<const char*>(wr.data.data()),
    static_cast<Py_ssize_t>(wr.data.size()));
}

// Convert a SparseRow to a Python dict {index: value}.
PyObject* sparse_row_to_dict(const hypervec::SparseRow* row) {
  PyObject* d = PyDict_New();
  if (d == nullptr) {
    throw std::runtime_error("failed to allocate Python dict");
  }
  for (size_t i = 0; i < row->nnz(); ++i) {
    PyObject* key = PyLong_FromUnsignedLong(row->index_at(i));
    PyObject* val = PyFloat_FromDouble(row->value_at(i));
    if (key == nullptr || val == nullptr || PyDict_SetItem(d, key, val) != 0) {
      Py_XDECREF(key);
      Py_XDECREF(val);
      Py_DECREF(d);
      throw std::runtime_error("failed to build sparse dict");
    }
    Py_DECREF(key);
    Py_DECREF(val);
  }
  return d;
}

// Serialize a TermDictionary to a Python bytes object (cross-language
// layout shared with the Python mirror).
PyObject* term_dictionary_serialize(const hypervec::TermDictionary* dict) {
  hypervec::VectorIOWriter wr;
  hypervec::write_term_dictionary(*dict, &wr);
  return PyBytes_FromStringAndSize(
    reinterpret_cast<const char*>(wr.data.data()),
    static_cast<Py_ssize_t>(wr.data.size()));
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

// Token lists for the BM25 analyzer cross to Python as a list of str.
// Must be instantiated at global scope, before `namespace hypervec`.
%template(StringVector) std::vector<std::string>;

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

struct IndexKDTree : Index {
  int leaf_size;
  explicit IndexKDTree(idx_t d, MetricType metric = kMetricL2,
                       int leaf_size = 16);
  void BuildTree();
};

struct IndexLVQ : Index {
  IndexLVQ(idx_t d, idx_t nlocal, int nbits, MetricType metric = kMetricL2);
};

%nodefaultctor IndexIVF;
struct IndexIVF : Index {
  idx_t nlist;
  idx_t nprobe;
};
%clearnodefaultctor IndexIVF;

struct IndexIVFFlat : IndexIVF {
  IndexIVFFlat(idx_t d, idx_t nlist, MetricType metric = kMetricL2);
};

struct IndexIVFLVQ : IndexIVF {
  IndexIVFLVQ(idx_t d, idx_t nlist, idx_t nlocal, int nbits,
              MetricType metric = kMetricL2);
};

struct IndexIVFPQ : IndexIVF {
  IndexIVFPQ(idx_t d, idx_t nlist, idx_t M, int nbits,
             MetricType metric = kMetricL2);
};

struct IndexRefineFlat : Index {
  IndexRefineFlat(Index* base);
  float k_factor;
};

struct IndexHNSWLVQ : Index {
  IndexHNSWLVQ(int d, int nlocal, int nbits, int M_hnsw,
               MetricType metric = kMetricL2);
};

struct IndexHNSWPQ : Index {
  IndexHNSWPQ(int d, int M_pq, int nbits, int M_hnsw,
              MetricType metric = kMetricL2);
};

void WriteIndex(const Index* idx, const char* fname, int io_flags = 0);
Index* ReadIndex(const char* fname, int io_flags = 0);

%nodefaultctor SparseRow;
struct SparseRow {
};

%extend SparseRow {
  // Construct from a Python dict {index: value}.
  SparseRow(PyObject* mapping) {
    return new hypervec::SparseRow(sparse_row_from_dict(mapping));
  }

  size_t nnz() const {
    return $self->nnz();
  }

  uint32_t index_at(size_t i) const {
    if (i >= $self->nnz()) {
      throw std::runtime_error("SparseRow.index_at: index out of range");
    }
    return $self->index_at(i);
  }

  float value_at(size_t i) const {
    if (i >= $self->nnz()) {
      throw std::runtime_error("SparseRow.value_at: index out of range");
    }
    return $self->value_at(i);
  }

  void set(uint32_t index, float value) {
    $self->set(index, value);
  }

  // Serialize to bytes using the shared cross-language layout.
  PyObject* serialize() const {
    return sparse_row_serialize($self);
  }

  // Convert to a Python dict {index: value}.
  PyObject* to_dict() const {
    return sparse_row_to_dict($self);
  }

  // Highest index + 1 (0 when empty).
  uint32_t dim() const {
    return $self->dim();
  }

  // Plain sparse inner product with another row.
  float dot(const hypervec::SparseRow* other) const {
    return $self->dot(*other);
  }

  // BM25 score with $self as the document (values = term frequencies) and
  // query_idf as the query (values = IDF weights).
  float dot_bm25(const hypervec::SparseRow* query_idf, float k1, float b,
                 float avgdl, float doc_len) const {
    hypervec::BM25Params params{k1, b, avgdl};
    return $self->dot_bm25(*query_idf, params, doc_len);
  }
}

%nodefaultctor TermDictionary;
struct TermDictionary {
};

%extend TermDictionary {
  TermDictionary() {
    return new hypervec::TermDictionary();
  }

  // Return the id of `term`, allocating the next id (first-seen order)
  // if unseen.
  uint32_t get_or_add(const std::string& term) {
    return $self->get_or_add(term);
  }

  bool contains(const std::string& term) const {
    return $self->contains(term);
  }

  uint32_t id_of(const std::string& term) const {
    return $self->id_of(term);
  }

  std::string term_of(uint32_t id) const {
    return $self->term_of(id);
  }

  size_t size() const {
    return $self->size();
  }

  // Serialize to bytes using the shared cross-language layout.
  PyObject* serialize() const {
    return term_dictionary_serialize($self);
  }
}

// ---------------------------------------------------------------------------
// BM25 full-text stack (T1 C++) exposed for the Python server's
// search_by_full_text path. Token lists cross as a Python list of str.
// ---------------------------------------------------------------------------
// Three dict paths for the Chinese analyzer (jieba / hmm / user).
struct AnalyzerParams {
  std::string jieba_dict_path;
  std::string hmm_model_path;
  std::string user_dict_path;
};

#if defined(HYPERVEC_ENABLE_BM25_ANALYZER)
// Resolve default dict paths (HYPERVEC_DICT_DIR env, else baked path).
hypervec::AnalyzerParams DefaultAnalyzerParams();

// Chinese tokenizer. Move-only in C++, so wrapped as a pointer-owned object:
// only the ctor-from-params and analyze() are exposed, never copy/move.
%nodefaultctor CppjiebaAnalyzer;
struct CppjiebaAnalyzer {
};
%extend CppjiebaAnalyzer {
  CppjiebaAnalyzer(const hypervec::AnalyzerParams* params) {
    return new hypervec::CppjiebaAnalyzer(*params);
  }

  // Tokenize `text` into lowercased terms (input order, empties omitted).
  std::vector<std::string> analyze(const std::string& text) const {
    return $self->Analyze(text);
  }
}
#endif

// Tokens -> TF (doc side) / IDF (query side) sparse vectors + corpus stats.
%nodefaultctor TextToSparse;
struct TextToSparse {
};
%extend TextToSparse {
  TextToSparse() {
    return new hypervec::TextToSparse();
  }

  // Doc side: TF SparseRow; updates DF/avgdl/num_docs.
  hypervec::SparseRow add_document(const std::vector<std::string>& tokens) {
    return $self->AddDocument(tokens);
  }

  // Query side: IDF SparseRow; unseen terms dropped.
  hypervec::SparseRow query_to_sparse(const std::vector<std::string>& tokens) const {
    return $self->QueryToSparse(tokens);
  }

  float avgdl() const {
    return $self->Avgdl();
  }

  uint64_t num_docs() const {
    return $self->NumDocs();
  }
}

// BM25 inverted index over TF SparseRows; doc ids assigned 0..n-1 by add order.
%nodefaultctor BM25InvertedIndex;
struct BM25InvertedIndex {
};
%extend BM25InvertedIndex {
  BM25InvertedIndex() {
    return new hypervec::BM25InvertedIndex();
  }

  // Add a Python sequence of SparseRow (TF). Doc ids continue from the
  // current count, so call order defines doc id -> row mapping.
  void add(PyObject* rows) {
    if (!PySequence_Check(rows)) {
      throw std::runtime_error("BM25InvertedIndex.add: expected a sequence of SparseRow");
    }
    Py_ssize_t n = PySequence_Size(rows);
    std::vector<hypervec::SparseRow> vec;
    vec.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      PyObject* item = PySequence_GetItem(rows, i);
      void* ptr = nullptr;
      int res = SWIG_ConvertPtr(item, &ptr, SWIGTYPE_p_hypervec__SparseRow, 0);
      if (!SWIG_IsOK(res) || ptr == nullptr) {
        Py_XDECREF(item);
        throw std::runtime_error("BM25InvertedIndex.add: sequence element is not a SparseRow");
      }
      vec.push_back(*reinterpret_cast<hypervec::SparseRow*>(ptr));
      Py_DECREF(item);
    }
    $self->Add(static_cast<hypervec::idx_t>(vec.size()), vec.data(), 0);
  }

  // BM25 k-NN. Returns (distances, labels) as two Python lists. avgdl must be
  // the corpus average doc length (use avgdl()); defaulting to 1.0 disables
  // the b length-normalization and skews scores.
  PyObject* search(const hypervec::SparseRow* query, int k,
                   float k1 = 1.2f, float b = 0.75f, float avgdl = 1.0f) const {
    if (k <= 0) {
      throw std::runtime_error("BM25InvertedIndex.search: k must be positive");
    }
    hypervec::SparseSearchParameters params;
    params.bm25.k1 = k1;
    params.bm25.b = b;
    params.bm25.avgdl = avgdl;
    std::vector<float> dist(static_cast<size_t>(k));
    std::vector<hypervec::idx_t> lab(static_cast<size_t>(k));
    // Release the GIL for the TAAT search (pure C++, touches no Python state),
    // so concurrent Python callers actually run in parallel.
    Py_BEGIN_ALLOW_THREADS
    $self->Search(*query, k, dist.data(), lab.data(), &params);
    Py_END_ALLOW_THREADS
    PyObject* dlist = PyList_New(k);
    PyObject* llist = PyList_New(k);
    for (int i = 0; i < k; ++i) {
      PyList_SetItem(dlist, i, PyFloat_FromDouble(static_cast<double>(dist[i])));
      PyList_SetItem(llist, i, PyLong_FromLongLong(static_cast<long long>(lab[i])));
    }
    PyObject* tup = PyTuple_New(2);
    PyTuple_SetItem(tup, 0, dlist);
    PyTuple_SetItem(tup, 1, llist);
    return tup;
  }

  float avgdl() const {
    return $self->Avgdl();
  }

  hypervec::idx_t num_docs() const {
    return $self->NumDocs();
  }
}

%extend Index {
  void train(PyObject* x) {
    index_train($self, x);
  }

  void add(PyObject* x) {
    index_add($self, x);
  }

  PyObject* search(PyObject* x, hypervec::idx_t k) {
    return index_search_with_params($self, x, k, nullptr);
  }

  PyObject* search_with_params(PyObject* x, hypervec::idx_t k, PyObject* py_params) {
    if (!PyDict_Check(py_params)) {
      throw std::runtime_error("search_with_params: params must be a dict");
    }
    if (const auto* hnsw = dynamic_cast<const hypervec::IndexHNSW*>($self)) {
      (void)hnsw;
      hypervec::SearchParametersHNSW params;
      PyObject* val = PyDict_GetItemString(py_params, "ef_search");
      if (!val) val = PyDict_GetItemString(py_params, "ef");
      if (val) params.ef_search = static_cast<int>(PyLong_AsLong(val));
      return index_search_with_params($self, x, k, &params);
    }
    if (const auto* ivf = dynamic_cast<const hypervec::IndexIVF*>($self)) {
      (void)ivf;
      hypervec::IVFSearchParameters params;
      PyObject* val = PyDict_GetItemString(py_params, "nprobe");
      if (val) params.nprobe = static_cast<hypervec::idx_t>(PyLong_AsLong(val));
      return index_search_with_params($self, x, k, &params);
    }
    if (const auto* rf = dynamic_cast<const hypervec::IndexRefineFlat*>($self)) {
      (void)rf;
      hypervec::RefineSearchParameters params;
      PyObject* val = PyDict_GetItemString(py_params, "k_factor");
      if (val) params.k_factor = static_cast<float>(PyFloat_AsDouble(val));
      return index_search_with_params($self, x, k, &params);
    }
    return index_search_with_params($self, x, k, nullptr);
  }
}

}  // namespace hypervec

%pythoncode %{
read_index = ReadIndex
write_index = WriteIndex
%}
