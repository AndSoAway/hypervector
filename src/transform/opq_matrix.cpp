/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <transform/opq_matrix.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifndef FINTEGER
#define FINTEGER int
#endif

extern "C" {

void sgesvd_(const char* jobu, const char* jobvt, const FINTEGER* m,
             const FINTEGER* n, float* a, const FINTEGER* lda, float* s,
             float* u, const FINTEGER* ldu, float* vt, const FINTEGER* ldvt,
             float* work, const FINTEGER* lwork, FINTEGER* info);
}

namespace hypervec {
namespace {

size_t CheckedElementCount(idx_t rows, idx_t columns, const char* context) {
  HYPERVEC_THROW_IF_NOT_FMT(rows >= 0, "%s: row count must be non-negative",
                            context);
  HYPERVEC_THROW_IF_NOT_FMT(columns > 0, "%s: dimension must be positive",
                            context);
  return mul_no_overflow(static_cast<size_t>(rows),
                         static_cast<size_t>(columns), context);
}

void ValidateTrainingInput(idx_t n, idx_t dimension, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(n > 0, "OPQMatrix::Train: n must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(x != nullptr,
                            "OPQMatrix::Train: input must not be null");
  const size_t count = CheckedElementCount(n, dimension, "OPQ training data");
  for (size_t i = 0; i < count; ++i) {
    HYPERVEC_THROW_IF_NOT_FMT(
        std::isfinite(x[i]),
        "OPQMatrix::Train: input contains a non-finite value at offset %zu", i);
  }
}

std::vector<float> IdentityMatrix(idx_t dimension) {
  const size_t count =
      CheckedElementCount(dimension, dimension, "OPQ identity matrix");
  std::vector<float> identity(count, 0.0F);
  for (idx_t i = 0; i < dimension; ++i) {
    identity[i * dimension + i] = 1.0F;
  }
  return identity;
}

/** Solve min_R ||R*x - target|| with R constrained to be orthogonal.
 *
 * The target-original cross-covariance C is decomposed as U*S*V^T and the
 * minimizer is R = U*V^T. LAPACK receives C in column-major layout.
 */
std::vector<float> SolveOrthogonalProcrustes(idx_t n, idx_t dimension,
                                             const float* x,
                                             const float* target) {
  HYPERVEC_THROW_IF_NOT_MSG(
      dimension <= static_cast<idx_t>((std::numeric_limits<FINTEGER>::max)()),
      "OPQMatrix::Train: dimension exceeds the LAPACK integer range");
  const size_t matrix_size =
      CheckedElementCount(dimension, dimension, "OPQ covariance matrix");
  std::vector<float> covariance(matrix_size);
  const double scale = 1.0 / static_cast<double>(n);
  for (idx_t row = 0; row < dimension; ++row) {
    for (idx_t column = 0; column < dimension; ++column) {
      double sum = 0.0;
      for (idx_t i = 0; i < n; ++i) {
        sum += static_cast<double>(target[i * dimension + row]) *
               x[i * dimension + column];
      }
      covariance[row + column * dimension] = static_cast<float>(sum * scale);
    }
  }

  const FINTEGER lapack_dimension = static_cast<FINTEGER>(dimension);
  const char all_vectors = 'A';
  std::vector<float> singular_values(static_cast<size_t>(dimension));
  std::vector<float> left(matrix_size);
  std::vector<float> right_transposed(matrix_size);
  float workspace_query = 0.0F;
  const FINTEGER query_size = -1;
  FINTEGER info = 0;
  std::vector<float> covariance_copy = covariance;
  sgesvd_(&all_vectors, &all_vectors, &lapack_dimension, &lapack_dimension,
          covariance_copy.data(), &lapack_dimension, singular_values.data(),
          left.data(), &lapack_dimension, right_transposed.data(),
          &lapack_dimension, &workspace_query, &query_size, &info);
  HYPERVEC_THROW_IF_NOT_FMT(
      info == 0,
      "OPQMatrix::Train: LAPACK workspace query failed with %" PRId64,
      static_cast<int64_t>(info));
  HYPERVEC_THROW_IF_NOT_MSG(
      std::isfinite(workspace_query) && workspace_query >= 1.0F &&
          static_cast<double>(workspace_query) <=
              static_cast<double>((std::numeric_limits<FINTEGER>::max)()),
      "OPQMatrix::Train: LAPACK returned an invalid workspace size");

  const FINTEGER workspace_size =
      static_cast<FINTEGER>(std::ceil(workspace_query));
  std::vector<float> workspace(static_cast<size_t>(workspace_size));
  covariance_copy = covariance;
  info = 0;
  sgesvd_(&all_vectors, &all_vectors, &lapack_dimension, &lapack_dimension,
          covariance_copy.data(), &lapack_dimension, singular_values.data(),
          left.data(), &lapack_dimension, right_transposed.data(),
          &lapack_dimension, workspace.data(), &workspace_size, &info);
  HYPERVEC_THROW_IF_NOT_FMT(
      info == 0,
      "OPQMatrix::Train: LAPACK SVD failed to converge with %" PRId64,
      static_cast<int64_t>(info));

  std::vector<float> rotation(matrix_size);
  for (idx_t row = 0; row < dimension; ++row) {
    for (idx_t column = 0; column < dimension; ++column) {
      double value = 0.0;
      for (idx_t k = 0; k < dimension; ++k) {
        value += static_cast<double>(left[row + k * dimension]) *
                 right_transposed[k + column * dimension];
      }
      rotation[row * dimension + column] = static_cast<float>(value);
    }
  }
  return rotation;
}

}  // namespace

OPQMatrix::OPQMatrix(idx_t dimension, idx_t subquantizer_count, int nbits)
    : LinearTransform(dimension, dimension),
      subquantizer_count(subquantizer_count),
      nbits(nbits) {
  HYPERVEC_THROW_IF_NOT_MSG(subquantizer_count > 0,
                            "OPQMatrix: subquantizer_count must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      dimension % subquantizer_count == 0,
      "OPQMatrix: dimension must be divisible by subquantizer_count");
  HYPERVEC_THROW_IF_NOT_FMT(nbits >= 1 && nbits <= HYPERVEC_PQ_MAX_NBITS,
                            "OPQMatrix: nbits must be in [1, %d]",
                            HYPERVEC_PQ_MAX_NBITS);
}

void OPQMatrix::Train(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT_MSG(parameters.iterations > 0,
                            "OPQMatrix::Train: iterations must be positive");
  ValidateTrainingInput(n, d_in, x);

  LinearTransform candidate(d_in, d_out);
  candidate.SetTransform(IdentityMatrix(d_in), {}, true);
  const size_t data_size = CheckedElementCount(n, d_in, "OPQ training data");
  std::vector<float> rotated(data_size);
  std::vector<float> reconstructed(data_size);

  for (int iteration = 0; iteration < parameters.iterations; ++iteration) {
    candidate.Apply(n, x, rotated.data());
    ProductQuantizer pq(d_in, subquantizer_count, nbits);
    pq.Train(n, rotated.data(), parameters.pq_parameters);

    const size_t code_count =
        mul_no_overflow(static_cast<size_t>(n), pq.code_size, "OPQ PQ codes");
    std::vector<uint8_t> codes(code_count);
    pq.ComputeCodes(n, rotated.data(), codes.data());
    pq.DecodeBatch(n, codes.data(), reconstructed.data());

    candidate.SetTransform(
        SolveOrthogonalProcrustes(n, d_in, x, reconstructed.data()), {}, true);
  }

  SetTransform(std::move(candidate.matrix), {}, true);
}

}  // namespace hypervec
