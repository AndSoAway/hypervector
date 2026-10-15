/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <quantization/pq/pq.h>
#include <transform/vector_transform.h>

namespace hypervec {

/** Training controls for an optimized product-quantization rotation. */
struct OPQParameters {
  /// Number of alternating PQ and orthogonal-Procrustes updates.
  int iterations = 8;

  /// Maximum rows used to learn the rotation; 0 disables sampling. The
  /// index still encodes all input rows after training.
  idx_t max_training_rows = 16384;

  /// Controls the temporary ProductQuantizer trained at every update.
  PQParameters pq_parameters;
};

/** Learned orthogonal rotation that reduces product-quantization error.
 *
 * Training alternates between encoding rotated data with a temporary PQ and
 * solving the orthogonal Procrustes problem from the original vectors to the
 * PQ reconstructions. The final component only stores the rotation; an index
 * remains responsible for training and storing its own PQ codec afterward.
 */
struct OPQMatrix : LinearTransform {
  idx_t subquantizer_count;
  int nbits;
  OPQParameters parameters;

  OPQMatrix(idx_t dimension, idx_t subquantizer_count, int nbits = 8);

  bool RequiresTraining() const override;

  /** Learn a square orthonormal rotation from n finite vectors.
   *
   * New state is committed only after every iteration succeeds. A failed
   * retraining therefore leaves an already-trained transform unchanged.
   */
  void Train(idx_t n, const float* x) override;
};

}  // namespace hypervec
