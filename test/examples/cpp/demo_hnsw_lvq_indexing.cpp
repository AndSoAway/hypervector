/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * IndexHNSWLVQ demo — train, add, sweep ef_search, measure recall vs brute-force.
 */

#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <utils/structures/random.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace hypervec;

int main() {
  std::cout << "HNSW+LVQ Index Demo" << std::endl;
  std::cout << "===================" << std::endl;

  const int d = 128;
  const int nb = 100000;
  const int nq = 1000;
  const int nlocal = 16;   // local centroids → 16×1024=16K distance table entries
  const int nbits = 10;    // ksub = 1024 (16K codewords → 6.25:1 compression)
  const int M_hnsw = 16;   // good graph quality
  const int k = 10;

  std::cout << "d=" << d << " nb=" << nb << " nq=" << nq
            << " nlocal=" << nlocal << " nbits=" << nbits
            << " M_hnsw=" << M_hnsw << " (ksub=" << (1 << nbits) << ")"
            << std::endl;

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

  std::cout << "\nBuilding IndexHNSWLVQ..." << std::endl;
  IndexHNSWLVQ idx(d, nlocal, nbits, M_hnsw);
  idx.hnsw.ef_construction = 20;  // balanced graph quality

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

  idx.Freeze();
  std::cout << "  Freeze done (raw storage released)" << std::endl;
  std::cout << "  Code bytes per vector: " << idx.SaCodeSize() << std::endl;

  std::cout << "\n------------------------------------------------" << std::endl;
  std::cout << "ef_search   Recall@" << k << "    Search ms   ms/query" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::vector<float> ds(nq * k);
  std::vector<idx_t> ls(nq * k);
  for (int ef : {64, 128, 256, 512, 1024}) {
    SearchParametersHNSW params;
    params.ef_search = ef;

    auto s0 = std::chrono::high_resolution_clock::now();
    idx.Search(nq, query.data(), k, ds.data(), ls.data(), &params);
    auto s1 = std::chrono::high_resolution_clock::now();
    double search_ms =
      std::chrono::duration<double, std::milli>(s1 - s0).count();

    int correct = 0;
    int total = nq * k;
    for (int i = 0; i < nq; i++) {
      for (int j = 0; j < k; j++) {
        for (int l = 0; l < k; l++) {
          if (ls[i * k + j] == gt_labels[i * k + l]) {
            correct++;
            break;
          }
        }
      }
    }
    float recall = (float)correct / (float)total;
    printf("  %4d        %5.1f%%    %8.2f    %8.3f\n",
           ef, recall * 100.0f, search_ms, search_ms / nq);
  }
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << "\nDone!" << std::endl;
  return 0;
}
