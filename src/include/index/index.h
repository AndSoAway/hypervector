/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

// -*- c++ -*-

#ifndef HYPERVEC_INDEX_H
#define HYPERVEC_INDEX_H

#include <utils/log/assert.h>
#include <utils/distances/metric_type.h>

#include <cstdio>

#define HYPERVEC_VERSION_MAJOR 1
#define HYPERVEC_VERSION_MINOR 14
#define HYPERVEC_VERSION_PATCH 1

// Macro to combine the version components into a single string
#ifndef HYPERVEC_STRINGIFY
#define HYPERVEC_STRINGIFY(ARG) #ARG
#endif
#ifndef HYPERVEC_TOSTRING
#define HYPERVEC_TOSTRING(ARG) HYPERVEC_STRINGIFY(ARG)
#endif
#define VERSION_STRING                                                 \
  HYPERVEC_TOSTRING(HYPERVEC_VERSION_MAJOR)                            \
  "." HYPERVEC_TOSTRING(HYPERVEC_VERSION_MINOR) "." HYPERVEC_TOSTRING( \
    HYPERVEC_VERSION_PATCH)

/**
 * @namespace hypervec
 *
 * Throughout the library, vectors are provided as float * pointers.
 * Most algorithms can be optimized when several vectors are processed
 * (added/searched) together in a batch. In this case, they are passed
 * in as a matrix. When n vectors of size d are provided as float * x,
 * component j of vector i is
 *
 *   x[ i * d + j ]
 *
 * where 0 <= i < n and 0 <= j < d. In other words, matrices are
 * always compact. When specifying the size of the matrix, we call it
 * an n*d matrix, which implies a row-major storage.
 */

namespace hypervec {

/// Forward declarations see impl/AuxIndexStructures.h, impl/IDSelector.h
/// and impl/DistanceComputer.h
struct IDSelector;
struct RangeSearchResult;
struct DistanceComputer;
template <typename T, typename TI>
struct ResultHandlerUnordered;
using ResultHandler = ResultHandlerUnordered<float, idx_t>;

enum NumericType {
  kFloat32,
  kFloat16,
  kUInt8,
  kInt8,
};

inline size_t GetNumericTypeSize(NumericType numeric_type) {
  switch (numeric_type) {
    case NumericType::kFloat32:
      return 4;
    case NumericType::kFloat16:
      return 2;
    case NumericType::kUInt8:
    case NumericType::kInt8:
      return 1;
    default:
      HYPERVEC_THROW_MSG(
        "Unknown Numeric Type. Only supports kFloat32, kFloat16");
  }
}

/** Parent class for the optional Search parameters.
 *
 * Sub-classes with additional Search parameters should inherit this class.
 * Ownership of the object fields is always to the caller.
 */
struct SearchParameters {
  /// if non-null, only these IDs will be considered during Search.
  IDSelector* sel = nullptr;
  /// make sure we can dynamic_cast this
  virtual ~SearchParameters() {}
};

/** Abstract structure for an index, supports adding vectors and searching
 * them.
 *
 * All vectors provided at Add or Search time are 32-bit float arrays,
 * although the internal representation may vary.
 */
struct Index {
  using component_t = float;
  using distance_t = float;

  int d;          ///< vector dimension
  idx_t n_total;  ///< total nb of indexed vectors
  bool verbose;   ///< verbosity level

  /// set if the Index does not require training, or if training is
  /// done already
  bool is_trained;

  /// type of metric this index uses for Search
  MetricType metric_type;
  float metric_arg;  ///< argument of the metric type

  explicit Index(idx_t d = 0, MetricType metric = kMetricL2)
    : d(d)
    , n_total(0)
    , verbose(false)
    , is_trained(true)
    , metric_type(metric)
    , metric_arg(0) {}

  virtual ~Index();

  /** Perform training on a representative set of vectors
   *
   * @param n      nb of training vectors
   * @param x      training vectors, size n * d
   */
  virtual void Train(idx_t n, const float* x);

  /** Perfrom training on a representative set of vectors and a representative
   * set of queries
   *
   * @param n         nb of training vectors
   * @param x         training vectors, size n * d
   * @param n_train_q nb of training queries
   * @param xq_train  training queries, size n_train_q * d
   */
  virtual void Train(idx_t n, const float* x, idx_t n_train_q,
                     const float* xq_train);

  virtual void TrainEx(idx_t n, const void* x, NumericType numeric_type) {
    if (numeric_type == NumericType::kFloat32) {
      Train(n, static_cast<const float*>(x));
    } else {
      HYPERVEC_THROW_MSG("Index::Train: unsupported numeric type");
    }
  }

  /** Add n vectors of dimension d to the index.
   *
   * Vectors are implicitly assigned labels n_total .. n_total + n - 1
   * This function slices the input vectors in chunks smaller than
   * blocksize_add and calls add_core.
   * @param n      number of vectors
   * @param x      input matrix, size n * d
   */
  virtual void Add(idx_t n, const float* x) = 0;

  virtual void AddEx(idx_t n, const void* x, NumericType numeric_type) {
    if (numeric_type == NumericType::kFloat32) {
      Add(n, static_cast<const float*>(x));
    } else {
      HYPERVEC_THROW_MSG("Index::Add: unsupported numeric type");
    }
  }

  /** Same as Add, but stores xids instead of sequential ids.
   *
   * The default implementation fails with an assertion, as it is
   * not supported by all indexes.
   *
   * @param n         number of vectors
   * @param x         input vectors, size n * d
   * @param xids      if non-null, ids to store for the vectors (size n)
   */
  virtual void AddWithIds(idx_t n, const float* x, const idx_t* xids);
  virtual void AddWithIdsEx(idx_t n, const void* x, NumericType numeric_type,
                               const idx_t* xids) {
    if (numeric_type == NumericType::kFloat32) {
      AddWithIds(n, static_cast<const float*>(x), xids);
    } else {
      HYPERVEC_THROW_MSG("Index::AddWithIds: unsupported numeric type");
    }
  }

  /** query n vectors of dimension d to the index.
   *
   * return at most k vectors. If there are not enough results for a
   * query, the result array is padded with -1s.
   *
   * @param n           number of vectors
   * @param x           input vectors to Search, size n * d
   * @param k           number of extracted vectors
   * @param distances   output pairwise distances, size n*k
   * @param labels      output labels of the NNs, size n*k
   */
  virtual void Search(idx_t n, const float* x, idx_t k, float* distances,
                      idx_t* labels,
                      const SearchParameters* params = nullptr) const = 0;

  virtual void SearchEx(idx_t n, const void* x, NumericType numeric_type,
                         idx_t k, float* distances, idx_t* labels,
                         const SearchParameters* params = nullptr) const {
    if (numeric_type == NumericType::kFloat32) {
      Search(n, static_cast<const float*>(x), k, distances, labels, params);
    } else {
      HYPERVEC_THROW_MSG("Index::Search: unsupported numeric type");
    }
  }

  /** Search one vector with a custom result handler */
  virtual void Search1(const float* x, ResultHandler& handler,
                       SearchParameters* params = nullptr) const;

  /** query n vectors of dimension d to the index.
   *
   * return all vectors with distance < radius. Note that many
   * indexes do not implement the RangeSearch (only the k-NN Search
   * is mandatory).
   *
   * @param n           number of vectors
   * @param x           input vectors to Search, size n * d
   * @param radius      Search radius
   * @param result      result table
   */
  virtual void RangeSearch(idx_t n, const float* x, float radius,
                            RangeSearchResult* result,
                            const SearchParameters* params = nullptr) const;

  /** return the indexes of the k vectors closest to the query x.
   *
   * This function is identical as Search but only return labels of
   * neighbors.
   * @param n           number of vectors
   * @param x           input vectors to Search, size n * d
   * @param labels      output labels of the NNs, size n*k
   * @param k           number of nearest neighbours
   */
  virtual void Assign(idx_t n, const float* x, idx_t* labels,
                      idx_t k = 1) const;

  /// removes all elements from the database.
  virtual void Reset() = 0;

  /** removes IDs from the index. Not supported by all
   * indexes. Returns the number of elements removed.
   */
  virtual size_t RemoveIds(const IDSelector& sel);

  /** Reconstruct a stored vector (or an approximation if lossy coding)
   *
   * this function may not be defined for some indexes
   * @param key         id of the vector to Reconstruct
   * @param recons      reconstructed vector (size d)
   */
  virtual void Reconstruct(idx_t key, float* recons) const;

  /** Reconstruct several stored vectors (or an approximation if lossy
   * coding)
   *
   * this function may not be defined for some indexes
   * @param n           number of vectors to Reconstruct
   * @param keys        ids of the vectors to Reconstruct (size n)
   * @param recons      reconstructed vector (size n * d)
   */
  virtual void ReconstructBatch(idx_t n, const idx_t* keys,
                                 float* recons) const;

  /** Reconstruct vectors i0 to i0 + ni - 1
   *
   * this function may not be defined for some indexes
   * @param i0          index of the first vector in the sequence
   * @param ni          number of vectors in the sequence
   * @param recons      reconstructed vector (size ni * d)
   */
  virtual void ReconstructN(idx_t i0, idx_t ni, float* recons) const;

  /** Similar to Search, but also reconstructs the stored vectors (or an
   * approximation in the case of lossy coding) for the Search results.
   *
   * If there are not enough results for a query, the resulting arrays
   * is padded with -1s.
   *
   * @param n           number of vectors
   * @param x           input vectors to Search, size n * d
   * @param k           number of extracted vectors
   * @param distances   output pairwise distances, size n*k
   * @param labels      output labels of the NNs, size n*k
   * @param recons      reconstructed vectors size (n, k, d)
   **/
  virtual void SearchAndReconstruct(
    idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
    float* recons, const SearchParameters* params = nullptr) const;

  /** Similar to Search, but operates on a potentially different subset
   * of the dataset for each query.
   *
   * The default implementation fails with an assertion, as it is
   * not supported by all indexes.
   *
   * @param n           number of vectors
   * @param x           input vectors, size n * d
   * @param k_base      number of vectors to Search from
   * @param base_labels ids of the vectors to Search from
   * @param k           desired number of results per query
   * @param distances   output pairwise distances, size n*k
   * @param labels      output labels of the NNs, size n*k
   */
  virtual void SearchSubset(idx_t n, const float* x, idx_t k_base,
                             const idx_t* base_labels, idx_t k,
                             float* distances, idx_t* labels) const;

  /** Computes a residual vector after indexing encoding.
   *
   * The residual vector is the difference between a vector and the
   * reconstruction that can be decoded from its representation in
   * the index. The residual can be used for multiple-stage indexing
   * methods, like IndexIVF's methods.
   *
   * @param x           input vector, size d
   * @param residual    output residual vector, size d
   * @param key         encoded index, as returned by Search and Assign
   */
  virtual void ComputeResidual(const float* x, float* residual,
                                idx_t key) const;

  /** Computes a residual vector after indexing encoding (batch form).
   * Equivalent to calling ComputeResidual for each vector.
   *
   * The residual vector is the difference between a vector and the
   * reconstruction that can be decoded from its representation in
   * the index. The residual can be used for multiple-stage indexing
   * methods, like IndexIVF's methods.
   *
   * @param n           number of vectors
   * @param xs          input vectors, size (n x d)
   * @param residuals   output residual vectors, size (n x d)
   * @param keys        encoded index, as returned by Search and Assign
   */
  virtual void ComputeResidualN(idx_t n, const float* xs, float* residuals,
                                  const idx_t* keys) const;

  /** Get a DistanceComputer (defined in AuxIndexStructures) object
   * for this kind of index.
   *
   * DistanceComputer is implemented for indexes that support random
   * access of their vectors.
   */
  virtual DistanceComputer* GetDistanceComputer() const;

  /* The standalone codec interface */

  /** size of the produced codes in bytes */
  virtual size_t SaCodeSize() const;

  /** encode a set of vectors
   *
   * @param n       number of vectors
   * @param x       input vectors, size n * d
   * @param bytes   output encoded vectors, size n * SaCodeSize()
   */
  virtual void SaEncode(idx_t n, const float* x, uint8_t* bytes) const;

  /** decode a set of vectors
   *
   * @param n       number of vectors
   * @param bytes   input encoded vectors, size n * SaCodeSize()
   * @param x       output vectors, size n * d
   */
  virtual void SaDecode(idx_t n, const uint8_t* bytes, float* x) const;

  /** moves the entries from another dataset to self.
   * On output, other is empty.
   * add_id is added to all moved ids
   * (for sequential ids, this would be this->n_total) */
  virtual void MergeFrom(Index& otherIndex, idx_t add_id = 0);

  /** check that the two indexes are compatible (ie, they are
   * trained in the same way and have the same
   * parameters). Otherwise throw. */
  virtual void CheckCompatibleForMerge(const Index& otherIndex) const;

  /** Add vectors that are computed with the standalone codec
   *
   * @param codes  codes to Add size n * SaCodeSize()
   * @param xids   corresponding ids, size n
   */
  virtual void AddSaCodes(idx_t n, const uint8_t* codes, const idx_t* xids);
};

}  // namespace hypervec

#endif
