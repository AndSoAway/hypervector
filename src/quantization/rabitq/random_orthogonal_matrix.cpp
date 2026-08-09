/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/rabitq/random_orthogonal_matrix.h>

#include <utils/log/exception.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hypervec {

namespace {
    // Portable PI constant (MSVC doesn't define M_PI)
    constexpr float kPi = 3.14159265358979323846f;
}

// ===========================================================================
//  Helper: Box-Muller transform for standard normal variates
// ===========================================================================

static void box_muller_fill(float* data, size_t n, RandomGenerator& rng) {
    // Box-Muller generates 2 outputs per 2 uniforms
    for (size_t i = 0; i + 1 < n; i += 2) {
        float u1 = rng.rand_float();  // (0, 1]
        if (u1 == 0.0f) u1 = 1e-10f; // avoid log(0)
        float u2 = rng.rand_float();
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * kPi * u2;
        data[i] = r * std::cos(theta);
        data[i + 1] = r * std::sin(theta);
    }
    if (n % 2 != 0) {
        // Handle odd size: generate one extra and keep only the first
        float u1 = rng.rand_float();
        if (u1 == 0.0f) u1 = 1e-10f;
        float u2 = rng.rand_float();
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * kPi * u2;
        data[n - 1] = r * std::cos(theta);
    }
}

// ===========================================================================
//  RandomOrthogonalMatrix
// ===========================================================================

RandomOrthogonalMatrix::RandomOrthogonalMatrix(int d, RandomGenerator& rng)
    : d(d) {
    HYPERVEC_THROW_IF_NOT_MSG(d > 0,
        "RandomOrthogonalMatrix: dimension must be positive");

    const size_t n = static_cast<size_t>(d) * d;
    matrix_data.resize(n);

    // Step 1: fill with standard normal variates (Box-Muller)
    box_muller_fill(matrix_data.data(), n, rng);

    // Step 2: in-place modified Gram-Schmidt QR, giving a Haar-distributed Q.
    std::vector<float> col_norm(static_cast<size_t>(d), 0.0f);

    for (int k = 0; k < d; k++) {
        // Load column k (row-major: column k means elements at [k, k+d, k+2d, ...])
        float* qk = matrix_data.data() + k;
        size_t k_stride = static_cast<size_t>(d);

        // Subtract projections onto previous orthogonalized columns
        for (int j = 0; j < k; j++) {
            float* qj = matrix_data.data() + j;
            // Compute dot(qj, qk)
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += qj[static_cast<size_t>(i) * k_stride] *
                       qk[static_cast<size_t>(i) * k_stride];
            }
            // qk -= dot * qj
            for (int i = 0; i < d; i++) {
                qk[static_cast<size_t>(i) * k_stride] -=
                    dot * qj[static_cast<size_t>(i) * k_stride];
            }
        }

        // Normalize qk
        float sq_norm = 0.0f;
        for (int i = 0; i < d; i++) {
            float v = qk[static_cast<size_t>(i) * k_stride];
            sq_norm += v * v;
        }
        float norm = std::sqrt(sq_norm);
        HYPERVEC_THROW_IF_NOT_FMT(
            norm > 1e-10f,
            "RandomOrthogonalMatrix: Gram-Schmidt failed at column %d; "
            "the random matrix was nearly singular. Retry with a different seed.",
            k);
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < d; i++) {
            qk[static_cast<size_t>(i) * k_stride] *= inv_norm;
        }
        col_norm[static_cast<size_t>(k)] = norm;
    }

    // Step 3: no sign adjustment needed — diag(R) = ||q_k_unnormalized|| > 0
    // by construction, so det(Q) = +1.  See paper §3.1.2.
}

void RandomOrthogonalMatrix::Transform(
        idx_t n, const float* x, float* y) const {
    // y = P · x  =>  y[i] = Σ_j P[i,j] * x[j]
    // For each vector k in [0, n):
    //   y[k*d + i] = Σ_j matrix_data[i*d + j] * x[k*d + j]
    const size_t d_sz = static_cast<size_t>(d);
    for (idx_t k = 0; k < n; k++) {
        const float* xv = x + k * d;
        float* yv = y + k * d;
        for (size_t i = 0; i < d_sz; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < d_sz; j++) {
                sum += matrix_data[i * d_sz + j] * xv[j];
            }
            yv[i] = sum;
        }
    }
}

void RandomOrthogonalMatrix::InverseTransform(
        idx_t n, const float* x, float* y) const {
    // y = P^{-1} · x = P^T · x  =>  y[j] = Σ_i P[i,j] * x[i]
    // For each vector k in [0, n):
    //   y[k*d + j] = Σ_i matrix_data[i*d + j] * x[k*d + i]
    const size_t d_sz = static_cast<size_t>(d);
    for (idx_t k = 0; k < n; k++) {
        const float* xv = x + k * d;
        float* yv = y + k * d;
        for (size_t j = 0; j < d_sz; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < d_sz; i++) {
                sum += matrix_data[i * d_sz + j] * xv[i];
            }
            yv[j] = sum;
        }
    }
}

void RandomOrthogonalMatrix::ComputeSignBits(
        idx_t n, const float* x, uint8_t* bits) const {
    // Temporary buffer for one transformed vector
    std::vector<float> transformed(static_cast<size_t>(d));

    const size_t d_sz = static_cast<size_t>(d);
    const size_t nbytes = (d_sz + 7) / 8;

    for (idx_t k = 0; k < n; k++) {
        // Inverse transform: x' = P^{-1} · x
        const float* xv = x + k * d;
        float* tv = transformed.data();
        for (size_t j = 0; j < d_sz; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < d_sz; i++) {
                sum += matrix_data[i * d_sz + j] * xv[i];
            }
            tv[j] = sum;
        }

        // Compute sign bits: bit j = 1 if tv[j] > 0 else 0
        uint8_t* bit_out = bits + k * nbytes;
        std::memset(bit_out, 0, nbytes);
        for (size_t j = 0; j < d_sz; j++) {
            if (tv[j] > 0.0f) {
                bit_out[j >> 3] |= (1 << (j & 7));
            }
        }
    }
}

void RandomOrthogonalMatrix::ComputeSignBitsOne(
        const float* x, uint8_t* bits) const {
    ComputeSignBits(1, x, bits);
}

size_t RandomOrthogonalMatrix::storage_size() const {
    const size_t d_sz = static_cast<size_t>(d);
    return sizeof(int32_t) + d_sz * d_sz * sizeof(float);
}

void RandomOrthogonalMatrix::write(IOWriter* f) const {
    // Write FourCC "RbtM"
    uint32_t fourcc_val = fourcc("RbtM");
    (*f)(&fourcc_val, sizeof(fourcc_val), 1);

    // Write dimension
    int32_t d32 = static_cast<int32_t>(d);
    (*f)(&d32, sizeof(d32), 1);

    // Write matrix data
    size_t n = static_cast<size_t>(d) * d;
    (*f)(matrix_data.data(), sizeof(float), n);
}

void RandomOrthogonalMatrix::read(IOReader* f) {
    // Read and verify FourCC
    uint32_t fourcc_val;
    (*f)(&fourcc_val, sizeof(fourcc_val), 1);
    if (fourcc_val != fourcc("RbtM")) {
        HYPERVEC_THROW_FMT(
            "RandomOrthogonalMatrix::read: expected FourCC 'RbtM' got '%s'",
            fourcc_inv_printable(fourcc_val).c_str());
    }

    // Read dimension
    int32_t d32;
    (*f)(&d32, sizeof(d32), 1);
    d = static_cast<int>(d32);

    // Read matrix data
    size_t n = static_cast<size_t>(d) * d;
    matrix_data.resize(n);
    (*f)(matrix_data.data(), sizeof(float), n);
}

}  // namespace hypervec