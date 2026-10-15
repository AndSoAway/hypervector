/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <omp.h>
#include <transform/vector_transform.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

constexpr double kOrthonormalTolerance = 1e-4;

size_t CheckedElementCount(idx_t rows, idx_t columns, const char* context) {
  HYPERVEC_THROW_IF_NOT_FMT(rows >= 0, "%s: row count must be non-negative",
                            context);
  HYPERVEC_THROW_IF_NOT_FMT(columns > 0, "%s: dimension must be positive",
                            context);
  return mul_no_overflow(static_cast<size_t>(rows),
                         static_cast<size_t>(columns), context);
}

void ValidateFinite(const std::vector<float>& values, const char* name) {
  for (size_t i = 0; i < values.size(); ++i) {
    HYPERVEC_THROW_IF_NOT_FMT(std::isfinite(values[i]),
                              "%s contains a non-finite value at offset %zu",
                              name, i);
  }
}

void ValidateOrthonormal(const std::vector<float>& matrix, idx_t dimension) {
  for (idx_t row = 0; row < dimension; ++row) {
    for (idx_t other = row; other < dimension; ++other) {
      double dot = 0.0;
      for (idx_t column = 0; column < dimension; ++column) {
        dot += static_cast<double>(matrix[row * dimension + column]) *
               matrix[other * dimension + column];
      }
      const double expected = row == other ? 1.0 : 0.0;
      HYPERVEC_THROW_IF_NOT_FMT(
          std::abs(dot - expected) <= kOrthonormalTolerance,
          "LinearTransform: matrix is not orthonormal at rows %" PRId64
          " and %" PRId64,
          row, other);
    }
  }
}

}  // namespace

VectorTransform::VectorTransform(idx_t d_in, idx_t d_out)
    : d_in(d_in), d_out(d_out), is_trained(false) {
  HYPERVEC_THROW_IF_NOT_FMT(
      d_in > 0, "VectorTransform: d_in must be positive, got %" PRId64, d_in);
  HYPERVEC_THROW_IF_NOT_FMT(
      d_out > 0, "VectorTransform: d_out must be positive, got %" PRId64,
      d_out);
}

bool VectorTransform::RequiresTraining() const { return false; }

bool VectorTransform::IsReversible() const { return false; }

void VectorTransform::Train(idx_t, const float*) {
  HYPERVEC_THROW_MSG("VectorTransform::Train: transform is not trainable");
}

std::vector<float> VectorTransform::Apply(idx_t n, const float* x) const {
  const size_t count = CheckedElementCount(n, d_out, "VectorTransform::Apply");
  std::vector<float> output(count);
  Apply(n, x, output.data());
  return output;
}

void VectorTransform::ReverseTransform(idx_t, const float*, float*) const {
  HYPERVEC_THROW_MSG("VectorTransform::ReverseTransform: inverse unavailable");
}

std::vector<float> VectorTransform::ReverseTransform(idx_t n,
                                                     const float* x) const {
  const size_t count =
      CheckedElementCount(n, d_in, "VectorTransform::ReverseTransform");
  std::vector<float> output(count);
  ReverseTransform(n, x, output.data());
  return output;
}

LinearTransform::LinearTransform(idx_t d_in, idx_t d_out)
    : VectorTransform(d_in, d_out) {}

void LinearTransform::SetTransform(std::vector<float> new_matrix,
                                   std::vector<float> new_bias,
                                   bool orthonormal) {
  const size_t matrix_size =
      CheckedElementCount(d_out, d_in, "LinearTransform matrix");
  HYPERVEC_THROW_IF_NOT_FMT(
      new_matrix.size() == matrix_size,
      "LinearTransform: matrix has %zu elements, expected %zu",
      new_matrix.size(), matrix_size);
  HYPERVEC_THROW_IF_NOT_FMT(
      new_bias.empty() || new_bias.size() == static_cast<size_t>(d_out),
      "LinearTransform: bias has %zu elements, expected 0 or %" PRId64,
      new_bias.size(), d_out);
  ValidateFinite(new_matrix, "LinearTransform matrix");
  ValidateFinite(new_bias, "LinearTransform bias");

  if (orthonormal) {
    HYPERVEC_THROW_IF_NOT_FMT(
        d_in == d_out,
        "LinearTransform: orthonormal transform must be square, got %" PRId64
        " x %" PRId64,
        d_out, d_in);
    ValidateOrthonormal(new_matrix, d_in);
  }

  matrix = std::move(new_matrix);
  bias = std::move(new_bias);
  is_orthonormal = orthonormal;
  is_trained = true;
}

void LinearTransform::SetIdentity() {
  HYPERVEC_THROW_IF_NOT_FMT(
      d_in == d_out,
      "LinearTransform::SetIdentity: transform must be square, got %" PRId64
      " x %" PRId64,
      d_out, d_in);
  const size_t matrix_size =
      CheckedElementCount(d_out, d_in, "LinearTransform identity");
  std::vector<float> identity(matrix_size, 0.0F);
  for (idx_t i = 0; i < d_in; ++i) {
    identity[i * d_in + i] = 1.0F;
  }
  SetTransform(std::move(identity), {}, true);
}

void LinearTransform::ValidateBatch(idx_t n, const float* x,
                                    const float* output,
                                    const char* operation) const {
  HYPERVEC_THROW_IF_NOT_FMT(is_trained, "%s: transform is not trained",
                            operation);
  HYPERVEC_THROW_IF_NOT_FMT(n >= 0, "%s: n must be non-negative", operation);
  if (n > 0) {
    HYPERVEC_THROW_IF_NOT_FMT(x != nullptr, "%s: input must not be null",
                              operation);
    HYPERVEC_THROW_IF_NOT_FMT(output != nullptr, "%s: output must not be null",
                              operation);
  }
}

void LinearTransform::Apply(idx_t n, const float* x, float* output) const {
  ValidateBatch(n, x, output, "LinearTransform::Apply");
  if (n == 0) {
    return;
  }

  std::vector<float> input_copy;
  if (x == output) {
    const size_t input_size =
        CheckedElementCount(n, d_in, "LinearTransform input");
    input_copy.assign(x, x + input_size);
    x = input_copy.data();
  }

  // Each output row is independent. Keep small searches serial to avoid
  // thread launch overhead; parallelize the large OPQ training/add batches.
#pragma omp parallel for if (n >= 256 && d_in >= 64 && d_out >= 64) \
    num_threads(std::min(32, omp_get_max_threads())) schedule(static)
  for (idx_t i = 0; i < n; ++i) {
    for (idx_t row = 0; row < d_out; ++row) {
      double value = bias.empty() ? 0.0 : bias[row];
      const float* matrix_row = matrix.data() + row * d_in;
      for (idx_t column = 0; column < d_in; ++column) {
        value += static_cast<double>(matrix_row[column]) * x[i * d_in + column];
      }
      output[i * d_out + row] = static_cast<float>(value);
    }
  }
}

void LinearTransform::ReverseTransform(idx_t n, const float* x,
                                       float* output) const {
  ValidateBatch(n, x, output, "LinearTransform::ReverseTransform");
  HYPERVEC_THROW_IF_NOT_MSG(
      is_orthonormal,
      "LinearTransform::ReverseTransform requires an orthonormal matrix");
  if (n == 0) {
    return;
  }

  std::vector<float> input_copy;
  if (x == output) {
    const size_t input_size =
        CheckedElementCount(n, d_out, "LinearTransform reverse input");
    input_copy.assign(x, x + input_size);
    x = input_copy.data();
  }

  for (idx_t i = 0; i < n; ++i) {
    for (idx_t column = 0; column < d_in; ++column) {
      double value = 0.0;
      for (idx_t row = 0; row < d_out; ++row) {
        const double centered =
            x[i * d_out + row] - (bias.empty() ? 0.0 : bias[row]);
        value += static_cast<double>(matrix[row * d_in + column]) * centered;
      }
      output[i * d_in + column] = static_cast<float>(value);
    }
  }
}

bool LinearTransform::IsReversible() const { return is_orthonormal; }

}  // namespace hypervec
