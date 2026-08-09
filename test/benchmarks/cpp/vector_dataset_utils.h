/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <index/index.h>

#include <cstdint>
#include <string>
#include <vector>

namespace hypervec {

/** Read a .fvecs file (float vector format).
 *
 *  Format: [dim(int32)] [vec[0](float)] ... [vec[dim-1](float)]
 *          repeated N times.
 *
 *  @param path     file path
 *  @param n_out    output: number of vectors
 *  @param d_out    output: vector dimension
 *  @return         flat float array, size n_out * d_out
 */
std::vector<float> ReadFvecs(const std::string& path,
                             size_t& n_out, size_t& d_out);

/** Read a .ivecs file (int ground-truth format).
 *
 *  Format: [k(int32)] [id[0](int32)] ... [id[k-1](int32)]
 *          repeated N times.
 *
 *  @param path     file path
 *  @param n_out    output: number of queries
 *  @param k_out    output: number of neighbours per query
 *  @return         flat int32 array, size n_out * k_out
 */
std::vector<int> ReadIvecs(const std::string& path,
                           size_t& n_out, size_t& k_out);

/** Read a .bvecs file (byte vector format).
 *
 *  Format: [dim(int32)] [b[0](uint8)] ... [b[dim-1](uint8)]
 *          repeated N times.  Values are converted to float32.
 *
 *  @param path     file path
 *  @param n_out    output: number of vectors
 *  @param d_out    output: vector dimension
 *  @return         flat float array, size n_out * d_out
 */
std::vector<float> ReadBvecs(const std::string& path,
                             size_t& n_out, size_t& d_out);

/** Compute ground truth via brute-force L2 search.
 *
 *  Used when a ground-truth .ivecs file is not available.
 *
 *  @param data       database vectors, n * d
 *  @param n          number of database vectors
 *  @param d          vector dimension
 *  @param queries    query vectors, nq * d
 *  @param nq         number of queries
 *  @param k          neighbours per query
 *  @return           flat idx_t array, size nq * k
 */
std::vector<idx_t> ComputeGroundTruth(
    const float* data, size_t n, size_t d,
    const float* queries, size_t nq, size_t k);

}  // namespace hypervec