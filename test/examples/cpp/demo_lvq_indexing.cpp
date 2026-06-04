/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * IndexLVQ demo: train, add, search, and measure recall against brute force.
 */

#include <index/flat/index_flat.h>
#include <quantization/lvq/index_lvq.h>
#include <utils/structures/random.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace hypervec;

int main() {
  std::cout << "LVQ Index Demo" << std::endl;
  std::cout << "==============" << std::endl;

  const int d = 128;
  const int nb = 100000;
  const int nq = 1000;
  const int nlocal = 256;
  const int nbits = 12;
  const int k = 10;

  std::cout << "d=" << d << " nb=" << nb << " nq=" << nq
            << " nlocal=" << nlocal << " nbits=" << nbits
            << " (ksub=" << (1 << nbits) << ")" << std::endl;

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

  std::cout << "\nBuilding IndexLVQ (brute-force scan)..." << std::endl;
  IndexLVQ idx(d, nlocal, nbits);

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
  std::cout << "  Code bytes per vector: " << idx.SaCodeSize() << std::endl;

  std::cout << "\nSearching..." << std::endl;
  std::vector<float> ds(nq * k);
  std::vector<idx_t> ls(nq * k);

  auto s0 = std::chrono::high_resolution_clock::now();
  idx.Search(nq, query.data(), k, ds.data(), ls.data());
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
  float recall = static_cast<float>(correct) / static_cast<float>(total);

  std::cout << "\n========================================" << std::endl;
  std::cout << "Results:" << std::endl;
  std::cout << "  Recall@" << k << ":  " << recall * 100.0f << "%"
            << std::endl;
  std::cout << "  Search time: " << search_ms << " ms" << std::endl;
  std::cout << "========================================" << std::endl;

  std::cout << "\nDone!" << std::endl;
  return 0;
}
