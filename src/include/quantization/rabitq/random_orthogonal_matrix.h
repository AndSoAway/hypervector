/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>
#include <persistence/io.h>
#include <utils/structures/random.h>

#include <cstdint>
#include <vector>

namespace hypervec {

/** Random orthogonal matrix (Haar-distributed) for RaBitQ's Johnson-Lindenstrauss
 *  transformation.
 *
 *  Constructs a d×d orthogonal matrix P by QR-decomposing a standard-normal
 *  random matrix and adjusting column signs so that det(P) = +1.  The matrix is
 *  stored row-major in `matrix_data` and supports forward (P·x) and inverse
 *  (P^T·x) transforms.
 *
 *  Storage cost: d×d floats (e.g. 64 KiB for d=128, ~3.5 MiB for d=960).
 */
struct RandomOrthogonalMatrix {
    int d = 0;                         ///< matrix dimension
    std::vector<float> matrix_data;    ///< d×d row-major storage

    RandomOrthogonalMatrix() = default;

    /** Sample a d×d Haar-distributed orthogonal matrix.
     *
     *  @param d    matrix dimension
     *  @param rng  random generator (seed is managed by the caller)
     */
    RandomOrthogonalMatrix(int d, RandomGenerator& rng);

    // -----------------------------------------------------------------------
    //  Transforms
    // -----------------------------------------------------------------------

    /** Forward transform: y = P · x  (for n vectors, stored flat). */
    void Transform(idx_t n, const float* x, float* y) const;

    /** Inverse transform: y = P^{-1} · x = P^T · x. */
    void InverseTransform(idx_t n, const float* x, float* y) const;

    /** Compute sign bits of P^{-1}·x for n vectors.
     *
     *  For each vector, writes ceil(d/8) bytes where byte k's bit j is 1 iff
     *  the (k*8+j)-th coordinate of P^{-1}·x is positive.
     *
     *  @param n      number of vectors
     *  @param x       input vectors, size n * d
     *  @param bits    output bitstrings, size n * ((d + 7) / 8)
     */
    void ComputeSignBits(idx_t n, const float* x, uint8_t* bits) const;

    /** Sign bits for a single vector (convenience wrapper). */
    void ComputeSignBitsOne(const float* x, uint8_t* bits) const;

    // -----------------------------------------------------------------------
    //  Serialization helpers
    // -----------------------------------------------------------------------

    size_t storage_size() const;
    void write(IOWriter* f) const;
    void read(IOReader* f);
};

}  // namespace hypervec