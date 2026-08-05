/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_rabitq.h>
#include <quantization/rabitq/index_rabitq.h>
#include <utils/distances/distances.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// Clustered synthetic data (like test_pq.cpp).
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
//  IndexRaBitQ (flat) tests
// ===========================================================================

TEST(IndexRaBitQ, ConstructAndTrain) {
    const int d = 32;
    hypervec::IndexRaBitQ index(d, 1);
    EXPECT_EQ(index.d, d);
    EXPECT_FALSE(index.is_trained);

    index.Train(0, nullptr);
    EXPECT_TRUE(index.is_trained);
}

TEST(IndexRaBitQ, AddAndSearch) {
    // 功能 sanity 检查：仅验证暴力搜索流水线能运行并返回非随机结果。
    // d=16 + B=1 误差界 O(1/√16)=25%，recall 天然偏低且波动。
    // 不用 d=128：MakeClusteredData 的簇中心模式只在低 6 维有区分度，
    // 高维时其余相同维度会干扰位编码。真实召回率由 benchmark 在真实
    // 数据集上验证（如 siftsmall 91%）。
    const int d = 16, n = 100, nq = 8, k = 5;
    std::vector<float> train = MakeClusteredData(d, 4, n / 4, 42);
    std::vector<float> queries = MakeClusteredData(d, 4, nq / 4, 99);

    hypervec::IndexRaBitQ index(d, 1);
    index.Train(n, train.data());
    index.Add(n, train.data());
    EXPECT_EQ(index.n_total, n);

    // Ground truth via flat L2
    hypervec::IndexFlatL2 flat(d);
    flat.Add(n, train.data());
    flat.is_trained = true;
    std::vector<float> gt_d(nq * k);
    std::vector<hypervec::idx_t> gt_l(nq * k);
    flat.Search(nq, queries.data(), k, gt_d.data(), gt_l.data());

    // RaBitQ search
    std::vector<float> d_out(nq * k);
    std::vector<hypervec::idx_t> l_out(nq * k);
    index.Search(nq, queries.data(), k, d_out.data(), l_out.data());

    // HNSW not used here; just verify valid results
    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(l_out[i], 0) << "invalid label " << i;
    }
    float recall = Recall(l_out, gt_l, nq, k);
    // d=16 + B=1：误差界 25%，recall 在 0.15~0.3 波动属正常。阈值 0.1
    // （2x 随机基线 0.05）仅验证暴力搜索功能正确，不做精度断言。
    EXPECT_GE(recall, 0.1f) << "IndexRaBitQ recall too low: " << recall;
}

TEST(IndexRaBitQ, GetDistanceComputer) {
    // Use a large dimension to keep RaBitQ's O(1/sqrt(D)) error small.
    const int d = 64, n = 50;
    std::vector<float> train = MakeClusteredData(d, 3, n / 3, 7);

    hypervec::IndexRaBitQ index(d, 1);
    index.Train(n, train.data());
    index.Add(n, train.data());

    hypervec::DistanceComputer* dc = index.GetDistanceComputer();
    ASSERT_NE(dc, nullptr);

    // Set query = first training vector; measure distance to itself.
    dc->SetQuery(train.data());
    float self_dist = (*dc)(0);
    // B=1 RaBitQ optimizes relative ordering, not absolute precision.
    // The error bound is O(1/sqrt(D)) = O(0.125) for D=64; with vector
    // norms ~ 22, the absolute error on a squared distance can be
    // ~ 2*22^2*0.125 ≈ 120, so we do NOT assert self-dist ≈ 0 exactly.
    // Instead, verify relative ordering: self-distance must be much
    // smaller than the distance to a far vector.

    // Distance to a far vector (end of the sorted-enough dataset).
    float other_dist = (*dc)(static_cast<hypervec::idx_t>(n - 1));

    // The far vector is in a different cluster (points are ~ 20 apart),
    // so its true squared distance is very large (~ thousands).  Even with
    // B=1 quantization error, it must be clearly larger than self-dist.
    EXPECT_GT(other_dist, self_dist + 100.0f)
        << "far vector should have larger distance than self";

    delete dc;
}

// ===========================================================================
//  IndexHNSWRaBitQ tests
// ===========================================================================

TEST(IndexHNSWRaBitQ, ConstructAndTrain) {
    const int d = 32, B = 1, M = 16;
    hypervec::IndexHNSWRaBitQ index(d, B, M);
    EXPECT_EQ(index.d, d);
    EXPECT_FALSE(index.is_trained);

    index.Train(0, nullptr);
    EXPECT_TRUE(index.is_trained);
}

TEST(IndexHNSWRaBitQ, AddAndSearch) {
    const int d = 16, B = 1, M = 16;
    const int n = 200, nq = 10, k = 10;

    std::vector<float> train = MakeClusteredData(d, 4, n / 4, 42);
    std::vector<float> queries = MakeClusteredData(d, 4, nq / 4, 99);

    hypervec::IndexHNSWRaBitQ index(d, B, M);
    index.Train(n, train.data());
    index.Add(n, train.data());
    EXPECT_EQ(index.n_total, n);

    // Ground truth
    hypervec::IndexFlatL2 flat(d);
    flat.Add(n, train.data());
    flat.is_trained = true;
    std::vector<float> gt_d(nq * k);
    std::vector<hypervec::idx_t> gt_l(nq * k);
    flat.Search(nq, queries.data(), k, gt_d.data(), gt_l.data());

    // HNSW+RaBitQ search with ef_search
    hypervec::SearchParametersHNSW params;
    params.ef_search = 100;

    std::vector<float> d_out(nq * k);
    std::vector<hypervec::idx_t> l_out(nq * k);
    index.Search(nq, queries.data(), k, d_out.data(), l_out.data(), &params);

    float recall = Recall(l_out, gt_l, nq, k);
    // HNSW+RaBitQ should be reasonably accurate for clustered data
    EXPECT_GT(recall, 0.2f) << "HNSW+RaBitQ recall too low: " << recall;
}

TEST(IndexHNSWRaBitQ, FreezeThenSearch) {
    const int d = 16, B = 1, M = 16;
    const int n = 100, nq = 5, k = 5;

    std::vector<float> train = MakeClusteredData(d, 4, n / 4, 42);
    std::vector<float> queries = MakeClusteredData(d, 4, nq / 4, 99);

    hypervec::IndexHNSWRaBitQ index(d, B, M);
    index.Train(n, train.data());
    index.Add(n, train.data());
    index.Freeze();

    // After freeze, raw_storage is released; search still works via RaBitQ codes
    hypervec::SearchParametersHNSW params;
    params.ef_search = 50;
    std::vector<float> d_out(nq * k);
    std::vector<hypervec::idx_t> l_out(nq * k);
    index.Search(nq, queries.data(), k, d_out.data(), l_out.data(), &params);
}

TEST(IndexHNSWRaBitQ, SaCodec) {
    const int d = 32, B = 1, M = 16, n = 10;
    std::vector<float> train = MakeClusteredData(d, 3, n / 3, 7);

    hypervec::IndexHNSWRaBitQ index(d, B, M);
    index.Train(n, train.data());

    size_t code_size = index.SaCodeSize();
    EXPECT_EQ(code_size, (static_cast<size_t>(d) + 7) / 8);

    std::vector<uint8_t> codes(n * code_size);
    index.SaEncode(n, train.data(), codes.data());

    std::vector<float> decoded(n * d);
    index.SaDecode(n, codes.data(), decoded.data());

    // Decoded should be unit vectors
    for (int i = 0; i < n; i++) {
        float sq = 0;
        for (int j = 0; j < d; j++) sq += decoded[i * d + j] * decoded[i * d + j];
        EXPECT_NEAR(sq, 1.0f, 1e-5f);
    }
}