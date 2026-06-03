/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.

 * HNSW index demo - demonstration of HNSW index functionality
 * This demo shows vector Search with HNSW index and calculates recall rate
 */

#include <index/hnsw/index_hnsw.h>
#include <index/flat/index_flat.h>
#include <utils/utils.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>

using namespace hypervec;

int main() {
    std::cout << "HNSW Index Demo" << std::endl;
    std::cout << "===============" << std::endl;

    // Parameters (unified: d=128, 100K vectors, 1K queries per teacher spec)
    int d = 128;            // dimension
    int n = 100000;         // number of vectors in database
    int nb = 1000;          // number of query vectors
    int M = 32;             // HNSW parameter: number of connections
    int ef_search = 64;     // HNSW parameter: Search width
    int ef_construction = 40; // build-time search depth (higher = better graph, slower build)
    int k = 10;             // number of nearest neighbors to Search

    // Generate random vectors
    std::cout << "Generating " << n << " random vectors of dimension " << d
              << std::endl;
    std::vector<float> database(n * d);
    std::vector<float> query(nb * d);

    // Use HyperVec random generator (uniform distribution)
    FloatRand(database.data(), n * d, 1234);
    FloatRand(query.data(), nb * d, 5678);

    // Create ground truth using brute-force Search (IndexFlatL2)
    std::cout << "Computing ground truth with brute-force Search..." << std::endl;
    IndexFlatL2 ground_truth(d);
    ground_truth.Add(n, database.data());

    std::vector<float> gt_distances(nb * k);
    std::vector<idx_t> gt_labels(nb * k);
    ground_truth.Search(nb, query.data(), k, gt_distances.data(), gt_labels.data());

    // Create HNSW index
    std::cout << "Creating HNSW index with M=" << M
              << ", ef_construction=" << ef_construction << std::endl;
    IndexHNSWFlat index(d, M);

    // Set build-time parameter (ef_search is set per-sweep below)
    index.hnsw.ef_construction = ef_construction;

    // Train (not strictly needed for HNSW, but can be called)
    std::cout << "Training index..." << std::endl;
    index.Train(n, database.data());

    // Add vectors to index
    std::cout << "Adding " << n << " vectors to index (building graph)..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    index.Add(n, database.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Index size: " << index.n_total << ", build time: " << add_ms << " ms" << std::endl;

    // Sweep ef_search to find recall vs speed trade-off
    std::cout << "\n================================================" << std::endl;
    std::cout << "ef_search   Recall@" << k << "    Search ms   ms/query" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::vector<float> hnsw_distances(nb * k);
    std::vector<idx_t> hnsw_labels(nb * k);

    for (int ef : {16, 32, 64, 128, 256, 512}) {
        index.hnsw.ef_search = ef;

        auto s0 = std::chrono::high_resolution_clock::now();
        index.Search(nb, query.data(), k, hnsw_distances.data(), hnsw_labels.data());
        auto s1 = std::chrono::high_resolution_clock::now();
        double search_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();

        // Calculate recall rate
        int correct = 0;
        int total = nb * k;
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < k; j++) {
                for (int l = 0; l < k; l++) {
                    if (hnsw_labels[i * k + j] == gt_labels[i * k + l]) {
                        correct++;
                        break;
                    }
                }
            }
        }
        float recall = (float)correct / (float)total;
        printf("  %4d        %5.1f%%    %8.2f    %8.3f\n",
               ef, recall * 100.0f, search_ms, search_ms / nb);
    }
    std::cout << "================================================" << std::endl;

    std::cout << "\nDone!" << std::endl;
    return 0;
}
