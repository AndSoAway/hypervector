/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * IVF index demo - demonstration of IndexIVFFlat functionality
 * Shows Train / Add / Search and measures recall vs brute-force ground truth
 * for different nprobe values.
 */

#include <index/flat/index_flat.h>
#include <index/ivf/index_ivf_flat.h>
#include <utils/structures/random.h>
#include <utils/utils.h>

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
  return (float)correct / (float)(nq * k);
}

int main() {
  std::cout << "IVF Flat Index Demo" << std::endl;
  std::cout << "===================" << std::endl;

  // Unified params (d=128, 100K vectors, 1K queries per teacher spec)
  const int d = 128;      // vector dimension
  const int nb = 100000;  // database size
  const int nq = 1000;    // number of queries
  const int nlist = 256;  // number of IVF clusters (sqrt(nb) ≈ 316)
  const int k = 10;       // nearest neighbours to retrieve

  // Generate random vectors
  std::cout << "Generating " << nb << " database vectors and " << nq
            << " query vectors (d=" << d << ")..." << std::endl;
  std::vector<float> database(nb * d);
  std::vector<float> query(nq * d);
  FloatRand(database.data(), nb * d, 1234);
  FloatRand(query.data(), nq * d, 5678);

  // Ground truth via brute-force flat search
  std::cout << "Computing ground truth (brute-force)..." << std::endl;
  IndexFlatL2 gt_index(d);
  gt_index.Add(nb, database.data());
  std::vector<float> gt_distances(nq * k);
  std::vector<idx_t> gt_labels(nq * k);
  gt_index.Search(nq, query.data(), k, gt_distances.data(), gt_labels.data());

  // Build IVF Flat index
  std::cout << "Training IndexIVFFlat (nlist=" << nlist << ")..." << std::endl;
  IndexIVFFlat ivf(d, nlist);

  auto t0 = std::chrono::high_resolution_clock::now();
  ivf.Train(nb, database.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  double train_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Train time: " << train_ms << " ms" << std::endl;

  std::cout << "Adding " << nb << " vectors..." << std::endl;
  t0 = std::chrono::high_resolution_clock::now();
  ivf.Add(nb, database.data());
  t1 = std::chrono::high_resolution_clock::now();
  double add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Add time:   " << add_ms << " ms" << std::endl;
  std::cout << "  Index size: " << ivf.n_total << " vectors" << std::endl;

  // Search with several nprobe values
  std::cout << "\n========================================" << std::endl;
  std::cout << "nprobe  Recall@" << k << "   Search time" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  IVFSearchParameters params;
  std::vector<float> ivf_distances(nq * k);
  std::vector<idx_t> ivf_labels(nq * k);

  for (int nprobe : {1, 4, 8, 16, 32, 64, 128, nlist}) {
    params.nprobe = nprobe;

    t0 = std::chrono::high_resolution_clock::now();
    ivf.Search(nq, query.data(), k, ivf_distances.data(), ivf_labels.data(),
               &params);
    t1 = std::chrono::high_resolution_clock::now();
    double search_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

    float recall = ComputeRecall(ivf_labels, gt_labels, nq, k);
    printf("  %4d    %6.2f%%     %.2f ms\n", nprobe, recall * 100.0f,
           search_ms);
  }

  // Correctness sanity check: nprobe=nlist must be exact
  params.nprobe = nlist;
  ivf.Search(nq, query.data(), k, ivf_distances.data(), ivf_labels.data(),
             &params);
  float exact_recall = ComputeRecall(ivf_labels, gt_labels, nq, k);
  std::cout << "========================================" << std::endl;
  if (exact_recall >= 0.99f) {
    std::cout << "PASS: nprobe=nlist recall " << exact_recall * 100.0f
              << "% >= 99% (expected near-exact)" << std::endl;
  } else {
    std::cout << "FAIL: nprobe=nlist recall " << exact_recall * 100.0f
              << "% is too low" << std::endl;
    return 1;
  }

  // Show first 3 queries at nprobe=nlist
  std::cout << "\nFirst 3 queries at nprobe=nlist (GT vs IVF):" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << "  Query " << i << " GT:  ";
    for (int j = 0; j < k; j++) {
      printf("%5lld(%.3f) ", (long long)gt_labels[i * k + j],
             gt_distances[i * k + j]);
    }
    std::cout << std::endl;
    std::cout << "  Query " << i << " IVF: ";
    for (int j = 0; j < k; j++) {
      printf("%5lld(%.3f) ", (long long)ivf_labels[i * k + j],
             ivf_distances[i * k + j]);
    }
    std::cout << std::endl;
  }

  std::cout << "\nDone!" << std::endl;
  return 0;
}
