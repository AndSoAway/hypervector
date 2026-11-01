/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/distances/metric_type.h>

#include <vector>

namespace hypervec {

/** Common protocol for transforms applied before vector encoding or search.
 *
 * Input and output batches are compact row-major matrices. Implementations
 * own their learned state; callers own every input and output buffer.
 */
struct VectorTransform {
  idx_t d_in;
  idx_t d_out;
  bool is_trained;

  VectorTransform(idx_t d_in, idx_t d_out);
  virtual ~VectorTransform() = default;

  /** Whether this transform type learns state from representative vectors. */
  virtual bool RequiresTraining() const;

  /** Whether the current state supports an exact reverse transform. */
  virtual bool IsReversible() const;

  /** Train the transform from n input vectors.
   *
   * Fixed transforms keep the default implementation, which rejects the
   * operation. Learned transforms such as OPQ override it.
   */
  virtual void Train(idx_t n, const float* x);

  /** Apply the transform into caller-owned storage of size n * d_out. */
  virtual void Apply(idx_t n, const float* x, float* output) const = 0;

  /** Allocate and return the transformed batch. */
  std::vector<float> Apply(idx_t n, const float* x) const;

  /** Reverse the transform into caller-owned storage of size n * d_in.
   *
   * Transformations without an exact inverse reject this operation.
   */
  virtual void ReverseTransform(idx_t n, const float* x, float* output) const;

  /** Allocate and return the reverse-transformed batch. */
  std::vector<float> ReverseTransform(idx_t n, const float* x) const;
};

/** Dense affine transform y = A*x + b.
 *
 * A is row-major with shape (d_out, d_in). An empty bias disables the affine
 * offset. SetTransform validates new state before replacing the active state,
 * so a rejected update leaves the transform usable.
 */
struct LinearTransform : VectorTransform {
  std::vector<float> matrix;
  std::vector<float> bias;
  bool is_orthonormal = false;

  LinearTransform(idx_t d_in, idx_t d_out);

  using VectorTransform::Apply;
  using VectorTransform::ReverseTransform;

  /** Install a validated transform.
   *
   * If orthonormal is true, dimensions must be square and A*A^T must be the
   * identity within numerical tolerance. Only orthonormal state supports
   * ReverseTransform, implemented as A^T*(y-b).
   */
  void SetTransform(std::vector<float> new_matrix,
                    std::vector<float> new_bias = {}, bool orthonormal = false);

  /** Configure the square transform as the identity matrix. */
  void SetIdentity();

  void Apply(idx_t n, const float* x, float* output) const override;
  void ReverseTransform(idx_t n, const float* x, float* output) const override;
  bool IsReversible() const override;

 private:
  void ValidateBatch(idx_t n, const float* x, const float* output,
                     const char* operation) const;
};

}  // namespace hypervec
