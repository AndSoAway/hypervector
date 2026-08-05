/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * IndexHNSWPQ demo — train, add, search, and measure recall against a
 * brute-force IndexFlatL2 ground truth. Demonstrates the dual-storage build
 * mode (raw scaffold + PQ codes) and Freeze() to release raw vectors after
 * bulk add.
 */

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <utils/structures/random.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace hypervec;

static float ComputeRecall(const std::vector<idx_t>& result,
                           const std::vector<idx_t>& ground_truth, int nq,
                           int k) {
  int correct = 0;
  for (int i = 0; i < nq; i++) {
    for (int j = 0; j < k; j++) {
      for (int l = 0; l < k; l++) {
        if (result[i * k + j] == ground_truth[i * k + l]) {
          correct++;
          break;
        }
      }
    }
  }
  return static_cast<float>(correct) / static_cast<float>(nq * k);
}

int main() {
  std::cout << "IndexHNSWPQ Demo (dual-storage build)" << std::endl;
  std::cout << "=====================================" << std::endl;

  // Unified params (d=128, 100K vectors, 1K queries per teacher spec)
  const int d = 128;
  const int nb = 100000;  // 10万向量
  const int nq = 1000;
  const int M_pq = 128;   // PQ subquantizers; dsub = 1 (near lossless)
  const int nbits = 8;    // ksub = 256
  const int M_hnsw = 64;  // HNSW out-degree (more connections = better recall)
  const int k = 10;

  std::cout << "d=" << d << " nb=" << nb << " nq=" << nq
            << " M_pq=" << M_pq << " nbits=" << nbits
            << " M_hnsw=" << M_hnsw << " (dsub=" << d / M_pq
            << " ksub=" << (1 << nbits) << ")" << std::endl;

  std::cout << "Generating random vectors..." << std::endl;
  std::vector<float> database(nb * d);
  std::vector<float> query(nq * d);
  FloatRand(database.data(), nb * d, 1234);
  FloatRand(query.data(), nq * d, 5678);

  std::cout << "Computing ground truth (IndexFlatL2)..." << std::endl;
  IndexFlatL2 gt_index(d);
  gt_index.Add(nb, database.data());
  std::vector<float> gt_distances(nq * k);
  std::vector<idx_t> gt_labels(nq * k);
  gt_index.Search(nq, query.data(), k, gt_distances.data(), gt_labels.data());

  std::cout << "\nBuilding IndexHNSWPQ..." << std::endl;
  IndexHNSWPQ idx(d, M_pq, nbits, M_hnsw);

  auto t0 = std::chrono::high_resolution_clock::now();
  idx.Train(nb, database.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  const double train_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Train time: " << train_ms << " ms" << std::endl;

  t0 = std::chrono::high_resolution_clock::now();
  idx.Add(nb, database.data());
  t1 = std::chrono::high_resolution_clock::now();
  const double add_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Add time:   " << add_ms << " ms (n_total=" << idx.n_total
            << ")" << std::endl;

  std::cout << "  Memory before Freeze: raw_storage retained" << std::endl;
  idx.Freeze();
  std::cout << "  After Freeze: raw scaffold released, index is read-only"
            << std::endl;

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "ef_search   Recall@" << k << "    Search ms" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  std::vector<float> ds(nq * k);
  std::vector<idx_t> ls(nq * k);
  for (int ef : {16, 32, 64, 128, 256, 512, 1024}) {
    SearchParametersHNSW params;
    params.ef_search = ef;

    t0 = std::chrono::high_resolution_clock::now();
    idx.Search(nq, query.data(), k, ds.data(), ls.data(), &params);
    t1 = std::chrono::high_resolution_clock::now();
    const double search_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

    const float recall = ComputeRecall(ls, gt_labels, nq, k);
    printf("  %4d        %5.1f%%      %7.2f\n", ef, recall * 100.0f,
           search_ms);
  }
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "(Recall is bounded by PQ quantization loss; raise M_pq or "
               "nbits to push the ceiling.)"
            << std::endl;

  std::cout << "\nDone!" << std::endl;
  return 0;
}
