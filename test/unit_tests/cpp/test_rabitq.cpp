/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <index/flat/index_flat.h>
#include <index/ivf/index_ivf.h>
#include <quantization/rabitq/rabitq.h>
#include <quantization/rabitq/index_ivfrabitq.h>
#include <utils/distances/distances.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
//  Synthetic data helpers
// ---------------------------------------------------------------------------

// Generate k_true clusters on a hypercube vertex pattern with Gaussian jitter.
std::vector<float> MakeClusteredData(hypervec::idx_t d, hypervec::idx_t k_true,
                                     hypervec::idx_t pts_per_cluster,
                                     int64_t seed) {
    hypervec::RandomGenerator rng(seed);
    const hypervec::idx_t n = k_true * pts_per_cluster;
    std::vector<float> x(static_cast<size_t>(n) * d);
    std::vector<float> centres(static_cast<size_t>(k_true) * d);
    for (hypervec::idx_t c = 0; c < k_true; c++) {
        for (hypervec::idx_t j = 0; j < d; j++) {
            centres[c * d + j] = (((c >> (j % 6)) & 1) ? 10.0f : -10.0f);
        }
    }
    for (hypervec::idx_t c = 0; c < k_true; c++) {
        for (hypervec::idx_t i = 0; i < pts_per_cluster; i++) {
            const hypervec::idx_t row = c * pts_per_cluster + i;
            for (hypervec::idx_t j = 0; j < d; j++) {
                const float jitter = 2.0f * rng.rand_float() - 1.0f;
                x[row * d + j] = centres[c * d + j] + jitter;
            }
        }
    }
    return x;
}

// Compute recall@k: fraction of ground-truth neighbours present in results.
float Recall(const std::vector<hypervec::idx_t>& result,
             const std::vector<hypervec::idx_t>& gt, int nq, int k) {
    int hits = 0;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            const hypervec::idx_t want = gt[i * k + j];
            for (int l = 0; l < k; l++) {
                if (result[i * k + l] == want) {
                    hits++;
                    break;
                }
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(nq * k);
}

}  // namespace

// ===========================================================================
//  Module 1: RandomOrthogonalMatrix tests
// ===========================================================================

TEST(RandomOrthogonalMatrix, IsOrthogonal) {
    const int d = 32;
    hypervec::RandomGenerator rng(42);
    hypervec::RandomOrthogonalMatrix rot(d, rng);

    // Verify P^T · P ≈ I (orthogonality)
    for (int i = 0; i < d; i++) {
        for (int j = i; j < d; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += rot.matrix_data[k * d + i] *
                       rot.matrix_data[k * d + j];
            }
            if (i == j) {
                EXPECT_NEAR(dot, 1.0f, 1e-5f)
                    << "diagonal entry (" << i << "," << i << ") failed";
            } else {
                EXPECT_NEAR(dot, 0.0f, 1e-5f)
                    << "off-diagonal entry (" << i << "," << j << ") failed";
            }
        }
    }
}

TEST(RandomOrthogonalMatrix, TransformPreservesNorm) {
    const int d = 64;
    hypervec::RandomGenerator rng(123);
    hypervec::RandomOrthogonalMatrix rot(d, rng);

    // For a random vector, ||P·x|| must equal ||x||
    std::vector<float> x(d);
    for (int j = 0; j < d; j++) {
        x[j] = 2.0f * rng.rand_float() - 1.0f;
    }
    float norm_x = 0.0f;
    for (int j = 0; j < d; j++) {
        norm_x += x[j] * x[j];
    }
    norm_x = std::sqrt(norm_x);

    std::vector<float> y(d);
    rot.Transform(1, x.data(), y.data());

    float norm_y = 0.0f;
    for (int j = 0; j < d; j++) {
        norm_y += y[j] * y[j];
    }
    norm_y = std::sqrt(norm_y);

    EXPECT_NEAR(norm_y, norm_x, 1e-5f);
}

TEST(RandomOrthogonalMatrix, InverseIsTranspose) {
    const int d = 32;
    hypervec::RandomGenerator rng(77);
    hypervec::RandomOrthogonalMatrix rot(d, rng);

    std::vector<float> x(d);
    for (int j = 0; j < d; j++) {
        x[j] = 2.0f * rng.rand_float() - 1.0f;
    }

    std::vector<float> y(d);
    rot.Transform(1, x.data(), y.data());

    std::vector<float> x_back(d);
    rot.InverseTransform(1, y.data(), x_back.data());

    // Tolerance 1e-4 allows for Gram-Schmidt numerical errors on 32×32
    // matrices; the error grows with dimension and is seed-dependent.
    for (int j = 0; j < d; j++) {
        EXPECT_NEAR(x_back[j], x[j], 1e-4f)
            << "dim " << j << ": P^{-1}(P(x)) != x";
    }
}

TEST(RandomOrthogonalMatrix, SignBitsRoundtrip) {
    const int d = 128;
    hypervec::RandomGenerator rng(99);
    hypervec::RandomOrthogonalMatrix rot(d, rng);

    std::vector<float> x(d);
    for (int j = 0; j < d; j++) {
        x[j] = 2.0f * rng.rand_float() - 1.0f;
    }

    const size_t nbytes = (static_cast<size_t>(d) + 7) / 8;
    std::vector<uint8_t> bits(nbytes);
    rot.ComputeSignBitsOne(x.data(), bits.data());

    // Verify: each bit corresponds to sign(P^{-1}·x)
    // We compute P^{-1}·x and check sign consistency
    std::vector<float> transformed(d);
    rot.InverseTransform(1, x.data(), transformed.data());

    for (int j = 0; j < d; j++) {
        int byte_idx = j >> 3;
        int bit_idx = j & 7;
        bool bit_set = (bits[byte_idx] >> bit_idx) & 1;
        bool expected = (transformed[j] > 0.0f);
        EXPECT_EQ(bit_set, expected)
            << "sign bit mismatch at dim " << j;
    }
}

// ===========================================================================
//  Module 2: RaBitQQuantizer tests (B=1, original RaBitQ)
// ===========================================================================

TEST(RaBitQQuantizer, ConstructAndTrain) {
    const int d = 32;
    hypervec::RaBitQQuantizer rabitq(d, 1);

    EXPECT_EQ(rabitq.d, d);
    EXPECT_EQ(rabitq.B, 1);
    EXPECT_GT(rabitq.code_size, 0u);
    EXPECT_EQ(rabitq.code_size, (static_cast<size_t>(d) + 7) / 8);
    EXPECT_FALSE(rabitq.is_trained);

    rabitq.Train(0, nullptr);
    EXPECT_TRUE(rabitq.is_trained);
}

TEST(RaBitQQuantizer, EncodeDecodeRoundtripIsUnitVector) {
    const int d = 64;
    const int n = 50;

    std::vector<float> x = MakeClusteredData(d, 5, n / 5, 42);

    hypervec::RaBitQQuantizer rabitq(d, 1);
    rabitq.Train(n, x.data());
    ASSERT_TRUE(rabitq.is_trained);

    std::vector<uint8_t> codes(static_cast<size_t>(n) * rabitq.code_size);
    rabitq.ComputeCodes(n, x.data(), codes.data());

    std::vector<float> decoded(static_cast<size_t>(n) * d);
    rabitq.DecodeBatch(n, codes.data(), decoded.data());

    // Decoded vectors must be unit vectors (since RaBitQ encodes to unit
    // vectors on the sphere).
    for (int i = 0; i < n; i++) {
        float sq_norm = 0.0f;
        for (int j = 0; j < d; j++) {
            sq_norm += decoded[i * d + j] * decoded[i * d + j];
        }
        EXPECT_NEAR(sq_norm, 1.0f, 1e-5f)
            << "vector " << i << " is not a unit vector";
    }
}

TEST(RaBitQQuantizer, PreprocessQueryOutputsValidConstants) {
    const int d = 32;
    hypervec::RaBitQQuantizer rabitq(d, 1);
    rabitq.Train(0, nullptr);

    std::vector<float> q(d);
    for (int j = 0; j < d; j++) q[j] = static_cast<float>(j + 1);

    std::vector<float> q_transformed(d);
    float norm, offset;
    rabitq.PreprocessQuery(q.data(), q_transformed.data(), norm, offset);

    EXPECT_GT(norm, 0.0f);
    EXPECT_NEAR(offset, norm * norm, 1e-5f);

    // The transformed vector should have norm ~1 (since input was normalized
    // then multiplied by an orthogonal matrix which preserves norm).
    float t_norm = 0.0f;
    for (int j = 0; j < d; j++) {
        t_norm += q_transformed[j] * q_transformed[j];
    }
    EXPECT_NEAR(std::sqrt(t_norm), 1.0f, 1e-5f);
}

TEST(RaBitQQuantizer, QuantizeQueryOutputsValidRange) {
    const int d = 32;
    hypervec::RaBitQQuantizer rabitq(d, 1);
    rabitq.Train(0, nullptr);

    std::vector<float> q(d);
    for (int j = 0; j < d; j++) q[j] = static_cast<float>(j % 5) * 0.5f;

    std::vector<uint8_t> q_quantized(d);
    float v_l, delta;
    rabitq.QuantizeQuery(q.data(), q_quantized.data(), v_l, delta, 4);

    // All quantized values should be in [0, 15]
    for (int j = 0; j < d; j++) {
        EXPECT_LE(q_quantized[j], 15);
        EXPECT_GE(q_quantized[j], 0);
    }
    EXPECT_GT(delta, 0.0f);
}

TEST(RaBitQQuantizer, ComputeSingleCodeProducesFiniteResult) {
    const int d = 64;
    const int n = 10;
    hypervec::RaBitQQuantizer rabitq(d, 1);
    rabitq.Train(0, nullptr);

    // Encode a data vector
    std::vector<float> x(d);
    for (int j = 0; j < d; j++) x[j] = static_cast<float>(j % 3);

    std::vector<uint8_t> code(rabitq.code_size);
    rabitq.ComputeCode(x.data(), code.data());

    // Set up query
    std::vector<float> q(d);
    for (int j = 0; j < d; j++) q[j] = static_cast<float>(j % 3 - 1);

    std::vector<float> q_transformed(d);
    float q_norm, dot_offset;
    rabitq.PreprocessQuery(q.data(), q_transformed.data(), q_norm, dot_offset);

    float v_l, delta;
    std::vector<uint8_t> q_quantized(d);
    rabitq.QuantizeQuery(q_transformed.data(), q_quantized.data(), v_l, delta);

    uint8_t* bit_planes[4];
    uint8_t bp_storage[4 * 16];  // 64/8 = 8 bytes per plane, 4 planes = 32 < 64
    const size_t nbytes = (static_cast<size_t>(d) + 7) / 8;
    for (int bp = 0; bp < 4; bp++) {
        bit_planes[bp] = bp_storage + bp * nbytes;
    }
    rabitq.ComputeBitPlanes(q_quantized.data(), bit_planes, d);

    int sum_q = 0;
    for (int j = 0; j < d; j++) sum_q += q_quantized[j];

    int popcnt = 0;
    for (size_t b = 0; b < nbytes; b++) {
        popcnt += __builtin_popcount(code[b]);
    }

    float ip_est = rabitq.ComputeSingleCode(
        code.data(), const_cast<const uint8_t**>(bit_planes),
        sum_q, v_l, delta, popcnt);

    EXPECT_TRUE(std::isfinite(ip_est));
    // For unit vectors, inner product must be in [-1, 1]
    EXPECT_LE(std::abs(ip_est), 1.0f + 1e-3f);
}

TEST(RaBitQQuantizer, EstimateDistanceSymmetricNonNegative) {
    const int d = 32;
    hypervec::RaBitQQuantizer rabitq(d, 1);
    rabitq.Train(0, nullptr);

    float dist = rabitq.EstimateDistance(0.5f, 10.0f, 8.0f, 0.0f);
    // With ip_est=0.5, norm_o=10, norm_q=8:
    // dist = 100 + 64 - 2*10*8*0.5 = 164 - 80 = 84
    EXPECT_NEAR(dist, 84.0f, 1e-5f);
}

TEST(RaBitQQuantizer, ErrorBoundFinitePositive) {
    const int d = 128;
    hypervec::RaBitQQuantizer rabitq(d, 1);

    float bound = rabitq.ComputeErrorBound(0.8f);
    EXPECT_TRUE(std::isfinite(bound));
    EXPECT_GT(bound, 0.0f);

    // With default epsilon0=1.9, ip=0.8, D=128:
    // factor = sqrt((1-0.64)/0.64) = sqrt(0.56) = 0.748
    // bound = 0.748 * 1.9 / sqrt(127) ≈ 0.748*1.9/11.26 ≈ 0.126
    EXPECT_LT(bound, 0.5f);
}

// ===========================================================================
//  Module 3: IndexIVFRaBitQ integration tests
// ===========================================================================

TEST(IndexIVFRaBitQ, TrainAndAdd) {
    const int d = 16, nlist = 4, B = 1;
    const int n = 100;

    std::vector<float> train = MakeClusteredData(d, nlist, n / nlist, 42);

    // Train coarse quantizer (k-means via IndexFlatL2)
    hypervec::IndexFlatL2 coarse_quantizer(d);
    // We need to prepare centroids. Simplest: use an IndexFlatL2 as quantizer
    // by adding vectors and training. But the IndexIVF constructor expects
    // a quantizer with properly set centroids.
    // Use a different approach: train IndexIVFRaBitQ directly which runs k-means.
    hypervec::IndexIVFRaBitQ index(
        &coarse_quantizer, d, nlist, B);

    index.Train(n, train.data());
    EXPECT_TRUE(index.is_trained);
    EXPECT_TRUE(index.rabitq.is_trained);

    index.Add(n, train.data());
    EXPECT_EQ(index.n_total, n);
}

TEST(IndexIVFRaBitQ, SearchReturnsResults) {
    const int d = 16, nlist = 4, B = 1;
    const int n = 200, nq = 8, k = 5;

    std::vector<float> train = MakeClusteredData(d, nlist, n / nlist, 42);
    std::vector<float> queries = MakeClusteredData(d, nlist, nq / nlist, 99);

    // Build IVF+RaBitQ index
    hypervec::IndexFlatL2 coarse_quantizer(d);
    hypervec::IndexIVFRaBitQ index(&coarse_quantizer, d, nlist, B);
    index.Train(n, train.data());
    index.Add(n, train.data());

    // Get ground truth via flat L2 search
    hypervec::IndexFlatL2 flat(d);
    flat.Add(n, train.data());
    flat.is_trained = true;

    std::vector<float> gt_distances(static_cast<size_t>(nq) * k);
    std::vector<hypervec::idx_t> gt_labels(static_cast<size_t>(nq) * k);
    flat.Search(nq, queries.data(), k,
                gt_distances.data(), gt_labels.data());

    // Run RaBitQ search
    std::vector<float> distances(static_cast<size_t>(nq) * k);
    std::vector<hypervec::idx_t> labels(static_cast<size_t>(nq) * k);

    hypervec::IVFSearchParameters params;
    params.nprobe = 2;
    index.Search(nq, queries.data(), k,
                 distances.data(), labels.data(), &params);

    // Verify recall is reasonable (better than random)
    float recall = Recall(labels, gt_labels, nq, k);
    EXPECT_GT(recall, 0.2f)
        << "Recall@k too low: " << recall;

    // All labels should be non-negative (valid)
    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0) << "invalid label at " << i;
    }
}

TEST(IndexIVFRaBitQ, SearchWithMultipleNprobe) {
    const int d = 16, nlist = 4, B = 1;
    const int n = 200, nq = 8, k = 5;

    std::vector<float> train = MakeClusteredData(d, nlist, n / nlist, 42);
    std::vector<float> queries = MakeClusteredData(d, nlist, nq / nlist, 99);

    hypervec::IndexFlatL2 coarse_quantizer(d);
    hypervec::IndexIVFRaBitQ index(&coarse_quantizer, d, nlist, B);
    index.Train(n, train.data());
    index.Add(n, train.data());

    // Ground truth
    hypervec::IndexFlatL2 flat(d);
    flat.Add(n, train.data());
    flat.is_trained = true;

    std::vector<float> gt_distances(static_cast<size_t>(nq) * k);
    std::vector<hypervec::idx_t> gt_labels(static_cast<size_t>(nq) * k);
    flat.Search(nq, queries.data(), k,
                gt_distances.data(), gt_labels.data());

    // Test with nprobe=1 (less accurate) and nprobe=4 (more accurate)
    float recall_1, recall_4;

    {
        hypervec::IVFSearchParameters params;
        params.nprobe = 1;
        std::vector<float> d1(static_cast<size_t>(nq) * k);
        std::vector<hypervec::idx_t> l1(static_cast<size_t>(nq) * k);
        index.Search(nq, queries.data(), k,
                     d1.data(), l1.data(), &params);
        recall_1 = Recall(l1, gt_labels, nq, k);
    }

    {
        hypervec::IVFSearchParameters params;
        params.nprobe = 4;
        std::vector<float> d4(static_cast<size_t>(nq) * k);
        std::vector<hypervec::idx_t> l4(static_cast<size_t>(nq) * k);
        index.Search(nq, queries.data(), k,
                     d4.data(), l4.data(), &params);
        recall_4 = Recall(l4, gt_labels, nq, k);
    }

    // More probes should give equal or better recall
    EXPECT_GE(recall_4, recall_1);
}

// ===========================================================================
//  Module 4: Extended RaBitQ (B>1) skeleton tests
// ===========================================================================

TEST(RaBitQQuantizer, ExtendedConstructB4) {
    const int d = 32;
    hypervec::RaBitQQuantizer rabitq(d, 4);

    EXPECT_EQ(rabitq.d, d);
    EXPECT_EQ(rabitq.B, 4);
    EXPECT_GT(rabitq.code_size, 0u);
    EXPECT_FALSE(rabitq.is_trained);

    // For B=4, code_size = (4 * d + 7) / 8 but with extended RaBitQ
    // the storage is also (B*d + 7) / 8 = (4*d + 7) / 8
    // For d=32: (128+7)/8 = 16
    EXPECT_EQ(rabitq.code_size, (4 * d + 7) / 8);
}

TEST(IndexIVFRaBitQ, ConstructWithExtendedB) {
    const int d = 32, nlist = 4;
    hypervec::IndexFlatL2 coarse_quantizer(d);
    hypervec::IndexIVFRaBitQ index(&coarse_quantizer, d, nlist, 4);
    EXPECT_EQ(index.rabitq.B, 4);
}

// ===========================================================================
//  Module 5: Edge cases
// ===========================================================================

TEST(RaBitQQuantizer, TrainIdempotent) {
    const int d = 32;
    hypervec::RaBitQQuantizer rabitq(d, 1);

    rabitq.Train(0, nullptr);
    EXPECT_TRUE(rabitq.is_trained);

    // Capture the matrix pointer (data() address) to check idempotency
    const float* mat_ptr_before = rabitq.rot.matrix_data.data();

    // Train again — should be a no-op
    rabitq.Train(0, nullptr);

    // After second train, matrix should be the same (untouched)
    EXPECT_EQ(rabitq.rot.matrix_data.data(), mat_ptr_before);
}

TEST(RaBitQQuantizer, ZeroQueryPreprocess) {
    const int d = 32;
    hypervec::RaBitQQuantizer rabitq(d, 1);
    rabitq.Train(0, nullptr);

    std::vector<float> q(d, 0.0f);
    std::vector<float> q_transformed(d);
    float norm, offset;
    rabitq.PreprocessQuery(q.data(), q_transformed.data(), norm, offset);

    // Zero query should have norm=0
    EXPECT_NEAR(norm, 0.0f, 1e-10f);
    EXPECT_NEAR(offset, 0.0f, 1e-10f);
}

TEST(IndexIVFRaBitQ, EmptyIndexSearch) {
    const int d = 16, nlist = 4, B = 1;
    hypervec::IndexFlatL2 coarse_quantizer(d);
    hypervec::IndexIVFRaBitQ index(&coarse_quantizer, d, nlist, B);

    // Even without training, Search should throw
    // We need to at least initialize properly
    std::vector<float> q(d);
    std::vector<float> dist(5);
    std::vector<hypervec::idx_t> labels(5);
    // Search on an untrained index should throw
    EXPECT_THROW({
        index.Search(1, q.data(), 5, dist.data(), labels.data(), nullptr);
    }, hypervec::HypervecException);
}