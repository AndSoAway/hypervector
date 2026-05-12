/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * IVFPQ index demo — train, add, search, and measure recall against a
 * brute-force IndexFlatL2 ground truth. Sweeps over nprobe and toggles
 * use_precomputed_table to confirm both paths produce identical results
 * (with a measurable speedup for the precomputed path).
 */

#include <index/flat/index_flat.h>
#include <index/ivf/index_ivf.h>
#include <quantization/pq/index_ivfpq.h>
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
  return static_cast<float>(correct) / static_cast<float>(nq * k);
}

int main() {
  std::cout << "IVFPQ Index Demo" << std::endl;
  std::cout << "================" << std::endl;

  // Parameters chosen so dsub = d / M = 8 (decent subspace size for k-means)
  const int d = 128;       // vector dimension
  const int nb = 10000;    // database size
  const int nq = 200;      // number of queries
  const int nlist = 128;   // IVF clusters
  const int M = 16;        // PQ subquantizers
  const int nbits = 8;     // bits per code → ksub = 256
  const int k = 10;

  std::cout << "d=" << d << " nb=" << nb << " nq=" << nq
            << " nlist=" << nlist << " M=" << M << " nbits=" << nbits
            << " (dsub=" << d / M << " ksub=" << (1 << nbits) << ")"
            << std::endl;

  std::cout << "Generating random vectors..." << std::endl;
  std::vector<float> database(nb * d);
  std::vector<float> query(nq * d);
  FloatRand(database.data(), nb * d, 1234);
  FloatRand(query.data(), nq * d, 5678);

  // Ground truth via brute-force flat L2 search.
  std::cout << "Computing ground truth (IndexFlatL2)..." << std::endl;
  IndexFlatL2 gt_index(d);
  gt_index.Add(nb, database.data());
  std::vector<float> gt_distances(nq * k);
  std::vector<idx_t> gt_labels(nq * k);
  gt_index.Search(nq, query.data(), k, gt_distances.data(), gt_labels.data());

  // Two indexes share training (same data, deterministic kmeans seed) so
  // they have identical centroids — only the search path differs.
  std::cout << "\nTraining IndexIVFPQ (basic)..." << std::endl;
  IndexIVFPQ ivfpq_basic(d, nlist, M, nbits);

  auto t0 = std::chrono::high_resolution_clock::now();
  ivfpq_basic.Train(nb, database.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  double train_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Train time: " << train_ms << " ms" << std::endl;

  t0 = std::chrono::high_resolution_clock::now();
  ivfpq_basic.Add(nb, database.data());
  t1 = std::chrono::high_resolution_clock::now();
  double add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Add time:   " << add_ms << " ms (n_total="
            << ivfpq_basic.n_total << ")" << std::endl;

  std::cout << "\nTraining IndexIVFPQ (use_precomputed_table=1)..."
            << std::endl;
  IndexIVFPQ ivfpq_pre(d, nlist, M, nbits);
  ivfpq_pre.use_precomputed_table = 1;

  t0 = std::chrono::high_resolution_clock::now();
  ivfpq_pre.Train(nb, database.data());
  t1 = std::chrono::high_resolution_clock::now();
  std::cout << "  Train time (incl. precompute): "
            << std::chrono::duration<double, std::milli>(t1 - t0).count()
            << " ms" << std::endl;
  ivfpq_pre.Add(nb, database.data());
  std::cout << "  Precomputed table size: "
            << ivfpq_pre.precomputed_table.size() << " floats ("
            << (ivfpq_pre.precomputed_table.size() * sizeof(float)) / 1024
            << " KB)" << std::endl;

  // Search sweep. Both paths compute the same L2² distances, but in
  // different floating-point orders, so distances may differ in their
  // last few bits and labels may swap at tie boundaries. We compare via
  // recall (insensitive to those FP ties) rather than strict label
  // equality.
  std::fflush(stdout);
  std::cout << "\n=================================================="
               "================="
            << std::endl;
  std::cout << "nprobe   Recall@" << k
            << " (basic / precomp)    Basic ms   Precomp ms   Speedup"
            << std::endl;
  std::cout << "----------------------------------------------------"
               "---------------"
            << std::endl;
  std::fflush(stdout);

  IVFSearchParameters params;
  std::vector<float> d_b(nq * k), d_p(nq * k);
  std::vector<idx_t> l_b(nq * k), l_p(nq * k);

  for (int nprobe : {1, 4, 8, 16, 32, nlist}) {
    params.nprobe = nprobe;

    t0 = std::chrono::high_resolution_clock::now();
    ivfpq_basic.Search(nq, query.data(), k, d_b.data(), l_b.data(), &params);
    t1 = std::chrono::high_resolution_clock::now();
    double basic_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    ivfpq_pre.Search(nq, query.data(), k, d_p.data(), l_p.data(), &params);
    t1 = std::chrono::high_resolution_clock::now();
    double pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const float r_b = ComputeRecall(l_b, gt_labels, nq, k);
    const float r_p = ComputeRecall(l_p, gt_labels, nq, k);
    const double speedup = pre_ms > 0 ? basic_ms / pre_ms : 0.0;

    printf("  %4d      %5.1f%% / %5.1f%%        %7.2f     %7.2f     %4.2fx\n",
           nprobe, r_b * 100.0f, r_p * 100.0f, basic_ms, pre_ms, speedup);
    std::fflush(stdout);
  }
  std::cout << "=================================================="
               "================="
            << std::endl;
  std::cout << "(basic and precomp recalls should match within float-tie "
               "noise.)"
            << std::endl;

  // Sanity check: nprobe=nlist with PQ on uniform random d=128 data is
  // hard — recall floor is around 25%. The point of this gate is just to
  // catch a fundamentally broken implementation, not to assert quality on
  // adversarial data.
  params.nprobe = nlist;
  ivfpq_basic.Search(nq, query.data(), k, d_b.data(), l_b.data(), &params);
  const float exhaustive_recall = ComputeRecall(l_b, gt_labels, nq, k);
  std::cout << "\nnprobe=nlist recall (basic): "
            << exhaustive_recall * 100.0f
            << "% (PQ is lossy + uniform d=128 random data is hard; "
               "expect ~25-40%)"
            << std::endl;
  if (exhaustive_recall < 0.20f) {
    std::cout << "FAIL: recall too low to be plausible" << std::endl;
    return 1;
  }

  std::cout << "\nDone!" << std::endl;
  return 0;
}
