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
#include <iostream>

using namespace hypervec;

int main() {
    std::cout << "HNSW Index Demo" << std::endl;
    std::cout << "===============" << std::endl;

    // Parameters
    int d = 128;           // dimension
    int n = 10000;         // number of vectors in database
    int nb = 100;          // number of query vectors
    int M = 32;            // HNSW parameter: number of connections
    int ef_search = 64;     // HNSW parameter: Search width
    int k = 10;            // number of nearest neighbors to Search

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
    std::cout << "Creating HNSW index with M=" << M << ", ef_search=" << ef_search
              << std::endl;
    IndexHNSWFlat index(d, M);

    // Set Search parameters
    index.hnsw.ef_search = ef_search;

    // Train (not strictly needed for HNSW, but can be called)
    std::cout << "Training index..." << std::endl;
    index.Train(n, database.data());

    // Add vectors to index
    std::cout << "Adding " << n << " vectors to index..." << std::endl;
    index.Add(n, database.data());
    std::cout << "Index size: " << index.n_total << std::endl;

    // Search
    std::cout << "Searching for " << k << " nearest neighbors for " << nb
              << " queries..." << std::endl;
    std::vector<float> hnsw_distances(nb * k);
    std::vector<idx_t> hnsw_labels(nb * k);

    index.Search(nb, query.data(), k, hnsw_distances.data(), hnsw_labels.data());

    // Calculate recall rate
    int correct = 0;
    int total = nb * k;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < k; j++) {
            // Check if HNSW result matches ground truth
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

    std::cout << "\nFirst 5 queries (ground truth vs HNSW):" << std::endl;
    for (int i = 0; i < 5; i++) {
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
