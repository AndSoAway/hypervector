/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <test/benchmarks/cpp/vector_dataset_utils.h>

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_rabitq.h>
#include <quantization/rabitq/rabitq.h>
#include <quantization/rabitq/index_rabitq.h>
#include <quantization/rabitq/index_ivfrabitq.h>
#include <utils/distances/distances.h>
#include <utils/structures/heap.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace hypervec;

// ===========================================================================
//  Evaluation metrics
// ===========================================================================

static double ComputeAvgRelativeError(
        const float* est_dist, const float* true_dist,
        size_t n_vals) {
    double sum = 0.0;
    size_t cnt = 0;
    for (size_t i = 0; i < n_vals; i++) {
        if (true_dist[i] > 1e-10f) {
            sum += std::abs(static_cast<double>(est_dist[i] - true_dist[i]) /
                            static_cast<double>(true_dist[i]));
            cnt++;
        }
    }
    return (cnt > 0) ? (sum / static_cast<double>(cnt)) : 0.0;
}

static double ComputeMaxRelativeError(
        const float* est_dist, const float* true_dist,
        size_t n_vals) {
    double max_err = 0.0;
    for (size_t i = 0; i < n_vals; i++) {
        if (true_dist[i] > 1e-10f) {
            double err = std::abs(
                static_cast<double>(est_dist[i] - true_dist[i]) /
                static_cast<double>(true_dist[i]));
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

static double ComputeRecall(
        const idx_t* result, const int* gt,
        size_t nq, size_t k) {
    size_t hits = 0;
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < k; j++) {
            idx_t want = gt[i * k + j];
            for (size_t l = 0; l < k; l++) {
                if (result[i * k + l] == want) {
                    hits++;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq * k);
}

static double ComputeAvgDistanceRatio(
        const float* est_dist, const float* true_dist,
        size_t n_vals) {
    double sum = 0.0;
    for (size_t i = 0; i < n_vals; i++) {
        if (true_dist[i] > 1e-10f) {
            sum += static_cast<double>(est_dist[i]) /
                   static_cast<double>(true_dist[i]);
        }
    }
    return sum / static_cast<double>(n_vals);
}

// ===========================================================================
//  Main
// ===========================================================================

int main(int argc, char** argv) {
    // ---- Default paths ----
    std::string base_path = "test/datasets/siftsmall/siftsmall_base.fvecs";
    std::string query_path = "test/datasets/siftsmall/siftsmall_query.fvecs";
    std::string gt_path = "test/datasets/siftsmall/siftsmall_groundtruth.ivecs";

    // ---- Parameters ----
    int nlist = 16;
    int nprobe = 4;
    int B = 1;
    int k = 100;
    int pool_mult = 3;
    int M_hnsw = 16;
    int ef_search = 100;
    int ef_construction = 200;
    bool use_rabitq = true;
    bool use_flat = false;
    bool use_rerank = false;
    bool use_hnsw = false;
    bool use_freeze = false;
    bool print_help = false;

    // ---- Parse command line ----
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--base" && i + 1 < argc) base_path = argv[++i];
        else if (arg == "--query" && i + 1 < argc) query_path = argv[++i];
        else if (arg == "--gt" && i + 1 < argc) gt_path = argv[++i];
        else if (arg == "--nlist" && i + 1 < argc) nlist = std::atoi(argv[++i]);
        else if (arg == "--nprobe" && i + 1 < argc) nprobe = std::atoi(argv[++i]);
        else if (arg == "--B" && i + 1 < argc) B = std::atoi(argv[++i]);
        else if (arg == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (arg == "--pool-mult" && i + 1 < argc) pool_mult = std::atoi(argv[++i]);
        else if (arg == "--M-hnsw" && i + 1 < argc) M_hnsw = std::atoi(argv[++i]);
        else if (arg == "--ef-search" && i + 1 < argc) ef_search = std::atoi(argv[++i]);
        else if (arg == "--ef-construction" && i + 1 < argc) ef_construction = std::atoi(argv[++i]);
        else if (arg == "--flat") use_rabitq = false, use_flat = true;
        else if (arg == "--rerank") use_rerank = true;
        else if (arg == "--hnsw") use_hnsw = true;
        else if (arg == "--freeze") use_freeze = true;
        else if (arg == "--help") print_help = true;
        else {
            fprintf(stderr, "Unknown arg: %s\n", arg.c_str());
            print_help = true;
        }
    }

    if (print_help) {
        fprintf(stderr,
            "Usage: %s [options]\n"
            "  --base <path>        base vectors (.fvecs or .bvecs)\n"
            "  --query <path>       query vectors (.fvecs)\n"
            "  --gt <path>          ground truth (.ivecs)\n"
            "  --nlist <int>        number of IVF clusters (default 16)\n"
            "  --nprobe <int>       IVF probes (default 4)\n"
            "  --B <int>            RaBitQ bits per dim (default 1)\n"
            "  --k <int>            neighbours per query (default 100)\n"
            "  --pool-mult <int>    pool multiplier for re-ranking (default 3)\n"
            "  --M-hnsw <int>       HNSW out-degree (default 16)\n"
            "  --ef-search <int>    HNSW search expansion (default 100)\n"
            "  --ef-construction <int> HNSW build expansion (default 200)\n"
            "  --flat               use flat L2 instead of RaBitQ\n"
            "  --rerank             enable re-ranking with exact distances\n"
            "  --hnsw               use HNSW+RaBitQ instead of IVF+RaBitQ\n"
            "  --freeze             drop raw scaffold after HNSW build\n"
            "  --help               this message\n",
            argv[0]);
        return (print_help && argc > 1) ? 0 : 1;
    }

    // ---- Load data ----
    printf("Loading base vectors from %s ...\n", base_path.c_str());
    size_t n = 0, d = 0;
    std::vector<float> base = ReadFvecs(base_path, n, d);
    printf("  n=%zu, d=%zu\n", n, d);

    printf("Loading query vectors from %s ...\n", query_path.c_str());
    size_t nq = 0, dq = 0;
    std::vector<float> queries = ReadFvecs(query_path, nq, dq);
    printf("  nq=%zu, dq=%zu\n", nq, dq);

    if (d != dq) {
        fprintf(stderr, "Dimension mismatch: base=%zu query=%zu\n", d, dq);
        return 1;
    }

    // ---- Load or compute ground truth ----
    std::vector<int> gt;
    size_t gt_nq = 0, gt_k = 0;
    try {
        gt = ReadIvecs(gt_path, gt_nq, gt_k);
        printf("Loaded ground truth: %zu queries x %zu neighbours\n", gt_nq, gt_k);
        if (gt_k < static_cast<size_t>(k)) {
            fprintf(stderr, "gt has k=%zu but requested k=%d\n", gt_k, k);
            return 1;
        }
    } catch (...) {
        printf("Ground truth file not found; computing via brute-force...\n");
        auto result = ComputeGroundTruth(
            base.data(), n, d, queries.data(), nq, static_cast<size_t>(k));
        gt_nq = nq;
        gt_k = static_cast<size_t>(k);
        gt.resize(gt_nq * gt_k);
        for (size_t i = 0; i < gt_nq * gt_k; i++) {
            gt[i] = static_cast<int>(result[i]);
        }
    }

    // ---- Output buffers ----
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    if (use_flat) {
        // ---- Baseline: flat L2 (brute-force) ----
        printf("\n=== Flat L2 (baseline) ===\n");
        IndexFlatL2 flat(static_cast<idx_t>(d));
        flat.Add(static_cast<idx_t>(n), base.data());
        flat.is_trained = true;

        auto t0 = std::chrono::high_resolution_clock::now();
        flat.Search(static_cast<idx_t>(nq), queries.data(),
                    static_cast<idx_t>(k),
                    distances.data(), labels.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double recall = ComputeRecall(labels.data(), gt.data(), nq, k);
        printf("  QPS:        %.1f\n", nq / (ms / 1000.0));
        printf("  Latency:    %.4f ms/query\n", ms / nq);
        printf("  Recall@%d:  %.4f\n", k, recall);

    } else if (use_hnsw) {
        const idx_t out_k = use_rerank ? k * pool_mult : k;
        printf("\n=== HNSW + RaBitQ ===\n");
        printf("  M=%d ef_search=%d ef_construction=%d B=%d k=%d",
               M_hnsw, ef_search, ef_construction, B, k);
        if (use_rerank) printf(" pool_mult=%d", pool_mult);
        printf("\n");

        // Build
        IndexHNSWRaBitQ index(static_cast<int>(d), B, M_hnsw);
        index.Train(static_cast<idx_t>(n), base.data());
        index.Add(static_cast<idx_t>(n), base.data());

        if (use_freeze) {
            printf("  Freezing index (raw scaffold released)...\n");
            index.Freeze();
        }

        // Buffers for search output (pool size when re-ranking)
        std::vector<float> search_dist(static_cast<size_t>(nq) * out_k);
        std::vector<idx_t> search_labels(static_cast<size_t>(nq) * out_k);

        SearchParametersHNSW params;
        params.ef_search = ef_search;
        index.Search(1, queries.data(), out_k,
                     search_dist.data(), search_labels.data(), &params);

        auto t0 = std::chrono::high_resolution_clock::now();
        index.Search(static_cast<idx_t>(nq), queries.data(), out_k,
                     search_dist.data(), search_labels.data(), &params);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- Re-rank with exact L2 distances if requested ----
        if (use_rerank) {
            #pragma omp parallel for
            for (idx_t qi = 0; qi < nq; qi++) {
                float* heap_dis = distances.data() + qi * k;
                idx_t* heap_ids = labels.data() + qi * k;
                heap_heapify<CMax<float, idx_t>>(static_cast<size_t>(k),
                                                 heap_dis, heap_ids);
                const float* xq = queries.data() + qi * d;
                for (idx_t pi = 0; pi < out_k; pi++) {
                    idx_t id = search_labels[qi * out_k + pi];
                    if (id < 0) continue;
                    float true_dist = fvec_L2sqr(
                        xq, base.data() + static_cast<size_t>(id) * d, d);
                    if (CMax<float, idx_t>::cmp(heap_dis[0], true_dist)) {
                        heap_replace_top<CMax<float, idx_t>>(
                            static_cast<size_t>(k), heap_dis, heap_ids,
                            true_dist, id);
                    }
                }
                heap_reorder<CMax<float, idx_t>>(static_cast<size_t>(k),
                                                 heap_dis, heap_ids);
            }
            double recall = ComputeRecall(labels.data(), gt.data(), nq, k);
            printf("  QPS:         %.1f\n", nq / (ms / 1000.0));
            printf("  Latency:     %.4f ms/query\n", ms / nq);
            printf("  Recall@%d:   %.4f\n", k, recall);
        } else {
            // Copy search results into output (they are the final labels)
            for (size_t i = 0; i < static_cast<size_t>(nq) * k; i++) {
                distances[i] = search_dist[i];
                labels[i] = search_labels[i];
            }
            double recall = ComputeRecall(labels.data(), gt.data(), nq, k);

            std::vector<float> true_dist_for_results(nq * k);
            for (size_t qi = 0; qi < nq; qi++) {
                for (size_t j = 0; j < static_cast<size_t>(k); j++) {
                    idx_t id = labels[qi * k + j];
                    if (id >= 0) {
                        true_dist_for_results[qi * k + j] = fvec_L2sqr(
                            queries.data() + qi * d,
                            base.data() + static_cast<size_t>(id) * d,
                            d);
                    } else {
                        true_dist_for_results[qi * k + j] = 0.0f;
                    }
                }
            }

            double avg_err = ComputeAvgRelativeError(
                distances.data(), true_dist_for_results.data(), nq * k);
            double max_err = ComputeMaxRelativeError(
                distances.data(), true_dist_for_results.data(), nq * k);
            double avg_ratio = ComputeAvgDistanceRatio(
                distances.data(), true_dist_for_results.data(), nq * k);

            printf("  QPS:         %.1f\n", nq / (ms / 1000.0));
            printf("  Latency:     %.4f ms/query\n", ms / nq);
            printf("  Recall@%d:   %.4f\n", k, recall);
            printf("  AvgRelErr:   %.4f\n", avg_err);
            printf("  MaxRelErr:   %.4f\n", max_err);
            printf("  AvgDistRat:  %.4f\n", avg_ratio);
        }

    } else if (use_rabitq) {
        // ---- Build index ----
        IndexFlatL2 coarse_quantizer(static_cast<idx_t>(d));
        IndexIVFRaBitQ index(&coarse_quantizer,
                             static_cast<size_t>(d),
                             static_cast<size_t>(nlist), B);

        // ---- Enable re-ranking if requested ----
        if (use_rerank) {
            printf("\n=== IVF + RaBitQ (with re-ranking) ===\n");
            index.EnableReranking();
        } else {
            printf("\n=== IVF + RaBitQ ===\n");
        }
        printf("  nlist=%d nprobe=%d B=%d k=%d", nlist, nprobe, B, k);
        if (use_rerank) printf(" pool_mult=%d", pool_mult);
        printf("\n");

        // ---- Train and add ----
        index.Train(static_cast<idx_t>(n), base.data());
        index.Add(static_cast<idx_t>(n), base.data());

        // ---- Warm-up ----
        if (use_rerank) {
            index.SearchWithRerank(1, queries.data(), k, pool_mult,
                                   distances.data(), labels.data(), nprobe);
        } else {
            IVFSearchParameters params;
            params.nprobe = nprobe;
            index.Search(1, queries.data(), k,
                         distances.data(), labels.data(), &params);
        }

        // ---- Timed search ----
        auto t0 = std::chrono::high_resolution_clock::now();
        if (use_rerank) {
            index.SearchWithRerank(static_cast<idx_t>(nq), queries.data(),
                                   static_cast<idx_t>(k), pool_mult,
                                   distances.data(), labels.data(), nprobe);
        } else {
            IVFSearchParameters params;
            params.nprobe = nprobe;
            index.Search(static_cast<idx_t>(nq), queries.data(),
                         static_cast<idx_t>(k),
                         distances.data(), labels.data(), &params);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- Evaluate ----
        double recall = ComputeRecall(labels.data(), gt.data(), nq, k);

        if (use_rerank) {
            // With re-ranking, distances returned are exact; metrics are trivial
            printf("  QPS:         %.1f\n", nq / (ms / 1000.0));
            printf("  Latency:     %.4f ms/query\n", ms / nq);
            printf("  Recall@%d:   %.4f\n", k, recall);
        } else {
            // Compute exact distances for the returned neighbours
            std::vector<float> true_dist_for_results(nq * k);
            for (size_t qi = 0; qi < nq; qi++) {
                for (size_t j = 0; j < static_cast<size_t>(k); j++) {
                    idx_t id = labels[qi * k + j];
                    if (id >= 0) {
                        true_dist_for_results[qi * k + j] = fvec_L2sqr(
                            queries.data() + qi * d,
                            base.data() + static_cast<size_t>(id) * d,
                            d);
                    } else {
                        true_dist_for_results[qi * k + j] = 0.0f;
                    }
                }
            }

            double avg_err = ComputeAvgRelativeError(
                distances.data(), true_dist_for_results.data(), nq * k);
            double max_err = ComputeMaxRelativeError(
                distances.data(), true_dist_for_results.data(), nq * k);
            double avg_ratio = ComputeAvgDistanceRatio(
                distances.data(), true_dist_for_results.data(), nq * k);

            printf("  QPS:         %.1f\n", nq / (ms / 1000.0));
            printf("  Latency:     %.4f ms/query\n", ms / nq);
            printf("  Recall@%d:   %.4f\n", k, recall);
            printf("  AvgRelErr:   %.4f\n", avg_err);
            printf("  MaxRelErr:   %.4f\n", max_err);
            printf("  AvgDistRat:  %.4f\n", avg_ratio);
        }
    }

    return 0;
}