/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.

 * HNSW Graph Search Demo
 * This demo demonstrates HNSW index using graph-based Search (not brute-force)
 */

#include <index/hnsw/index_hnsw.h>
#include <index/flat/index_flat.h>
#include <utils/utils.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>

using namespace hypervec;

int main() {
    std::cout << "HNSW Graph Search Demo" << std::endl;
    std::cout << "=====================" << std::endl;

    // Parameters
    int d = 128;           // dimension
    int n = 50000;        // number of vectors in database
    int nb = 100;         // number of query vectors
    int M = 64;           // HNSW parameter: number of connections (increased from 32)
    int ef_search = 256;   // HNSW parameter: Search width (larger = higher recall, slower)
    int k = 10;           // number of nearest neighbors to Search

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimension: " << d << std::endl;
    std::cout << "  Database size: " << n << std::endl;
    std::cout << "  Query count: " << nb << std::endl;
    std::cout << "  HNSW M: " << M << std::endl;
    std::cout << "  HNSW ef_search: " << ef_search << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << std::endl;

    // Generate random vectors
    std::cout << "Generating " << n << " random vectors of dimension " << d << std::endl;
    std::vector<float> database(n * d);
    std::vector<float> query(nb * d);

    FloatRand(database.data(), n * d, 1234);
    FloatRand(query.data(), nb * d, 5678);

    // Create ground truth using brute-force Search (IndexFlatL2)
    std::cout << "Computing ground truth with brute-force Search..." << std::endl;
    IndexFlatL2 ground_truth(d);
    ground_truth.Add(n, database.data());

    auto gt_start = std::chrono::high_resolution_clock::now();
    std::vector<float> gt_distances(nb * k);
    std::vector<idx_t> gt_labels(nb * k);
    ground_truth.Search(nb, query.data(), k, gt_distances.data(), gt_labels.data());
    auto gt_end = std::chrono::high_resolution_clock::now();
    double gt_time = std::chrono::duration<double, std::milli>(gt_end - gt_start).count();
    std::cout << "Ground truth Search time: " << gt_time << " ms" << std::endl << std::endl;

    // Create HNSW index with graph-based Search
    std::cout << "Creating HNSW index (graph-based Search)..." << std::endl;
    IndexHNSWFlat index(d, M);

    // Set Search parameters
    index.hnsw.ef_search = ef_search;

    std::cout << "Training index..." << std::endl;
    index.Train(n, database.data());

    auto add_start = std::chrono::high_resolution_clock::now();
    std::cout << "Adding " << n << " vectors to index (building graph)..." << std::endl;
    index.Add(n, database.data());
    auto add_end = std::chrono::high_resolution_clock::now();
    double add_time = std::chrono::duration<double, std::milli>(add_end - add_start).count();
    std::cout << "Index build time: " << add_time << " ms" << std::endl;
    std::cout << "Index size: " << index.n_total << std::endl;

    // Search using HNSW graph
    std::cout << "\nSearching using HNSW graph..." << std::endl;
    std::vector<float> hnsw_distances(nb * k);
    std::vector<idx_t> hnsw_labels(nb * k);

    auto search_start = std::chrono::high_resolution_clock::now();
    index.Search(nb, query.data(), k, hnsw_distances.data(), hnsw_labels.data());
    auto search_end = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double, std::milli>(search_end - search_start).count();

    std::cout << "HNSW Search time: " << search_time << " ms" << std::endl;
    std::cout << "Average per query: " << search_time / nb << " ms" << std::endl;

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
    float recall = (float)correct / total;

    // Print results
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Recall@" << k << ": " << recall * 100 << "%" << std::endl;
    std::cout << "Correct matches: " << correct << "/" << total << std::endl;
    std::cout << std::endl;
    std::cout << "Performance comparison:" << std::endl;
    std::cout << "  Brute-force: " << gt_time << " ms" << std::endl;
    std::cout << "  HNSW graph:  " << search_time << " ms" << std::endl;
    std::cout << "  Speedup:     " << gt_time / search_time << "x" << std::endl;

    std::cout << "\nFirst 3 queries (ground truth vs HNSW):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "Query " << i << ":" << std::endl;
        std::cout << "  GT:    ";
        for (int j = 0; j < k; j++) {
            std::cout << gt_labels[i * k + j] << "(" << gt_distances[i * k + j] << ") ";
        }
        std::cout << std::endl;
        std::cout << "  HNSW:  ";
        for (int j = 0; j < k; j++) {
            std::cout << hnsw_labels[i * k + j] << "(" << hnsw_distances[i * k + j] << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}