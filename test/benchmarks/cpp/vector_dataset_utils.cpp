/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <test/benchmarks/cpp/vector_dataset_utils.h>

#include <index/flat/index_flat.h>
#include <utils/distances/distances.h>
#include <utils/structures/heap.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace hypervec {

std::vector<float> ReadFvecs(const std::string& path,
                             size_t& n_out, size_t& d_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("ReadFvecs: cannot open " + path);
    }

    // Read the first int32 to get dimension
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!in || dim <= 0) {
        throw std::runtime_error("ReadFvecs: invalid dimension");
    }
    d_out = static_cast<size_t>(dim);

    // Seek to end to determine file size
    in.seekg(0, std::ios::end);
    size_t file_bytes = static_cast<size_t>(in.tellg());
    size_t vec_bytes = sizeof(int32_t) + d_out * sizeof(float);
    size_t n = file_bytes / vec_bytes;
    n_out = n;

    // Rewind and read all vectors
    in.seekg(0, std::ios::beg);
    std::vector<float> data(n * d_out);

    for (size_t i = 0; i < n; i++) {
        int32_t vec_dim = 0;
        in.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        if (static_cast<size_t>(vec_dim) != d_out) {
            throw std::runtime_error("ReadFvecs: inconsistent dimension");
        }
        in.read(reinterpret_cast<char*>(data.data() + i * d_out),
                static_cast<std::streamsize>(d_out * sizeof(float)));
    }

    return data;
}

std::vector<int> ReadIvecs(const std::string& path,
                           size_t& n_out, size_t& k_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("ReadIvecs: cannot open " + path);
    }

    // Read the first int32 to get k
    int32_t k = 0;
    in.read(reinterpret_cast<char*>(&k), sizeof(int32_t));
    if (!in || k <= 0) {
        throw std::runtime_error("ReadIvecs: invalid k");
    }
    k_out = static_cast<size_t>(k);

    // Seek to end to determine file size
    in.seekg(0, std::ios::end);
    size_t file_bytes = static_cast<size_t>(in.tellg());
    size_t entry_bytes = sizeof(int32_t) + k_out * sizeof(int32_t);
    size_t n = file_bytes / entry_bytes;
    n_out = n;

    // Rewind and read all entries
    in.seekg(0, std::ios::beg);
    std::vector<int> data(n * k_out);

    for (size_t i = 0; i < n; i++) {
        int32_t entry_k = 0;
        in.read(reinterpret_cast<char*>(&entry_k), sizeof(int32_t));
        if (static_cast<size_t>(entry_k) != k_out) {
            throw std::runtime_error("ReadIvecs: inconsistent k");
        }
        in.read(reinterpret_cast<char*>(data.data() + i * k_out),
                static_cast<std::streamsize>(k_out * sizeof(int32_t)));
    }

    return data;
}

std::vector<float> ReadBvecs(const std::string& path,
                             size_t& n_out, size_t& d_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("ReadBvecs: cannot open " + path);
    }

    // Read the first int32 to get dimension
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!in || dim <= 0) {
        throw std::runtime_error("ReadBvecs: invalid dimension");
    }
    d_out = static_cast<size_t>(dim);

    // Seek to end to determine file size
    in.seekg(0, std::ios::end);
    size_t file_bytes = static_cast<size_t>(in.tellg());
    size_t vec_bytes = sizeof(int32_t) + d_out * sizeof(uint8_t);
    size_t n = file_bytes / vec_bytes;
    n_out = n;

    // Rewind and read all vectors, converting uint8 → float32
    in.seekg(0, std::ios::beg);
    std::vector<float> data(n * d_out);

    for (size_t i = 0; i < n; i++) {
        int32_t vec_dim = 0;
        in.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        if (static_cast<size_t>(vec_dim) != d_out) {
            throw std::runtime_error("ReadBvecs: inconsistent dimension");
        }
        std::vector<uint8_t> raw(d_out);
        in.read(reinterpret_cast<char*>(raw.data()),
                static_cast<std::streamsize>(d_out * sizeof(uint8_t)));
        for (size_t j = 0; j < d_out; j++) {
            data[i * d_out + j] = static_cast<float>(raw[j]);
        }
    }

    return data;
}

std::vector<idx_t> ComputeGroundTruth(
    const float* data, size_t n, size_t d,
    const float* queries, size_t nq, size_t k) {
    // Use IndexFlatL2 for brute-force search
    IndexFlatL2 flat(static_cast<idx_t>(d));
    flat.Add(static_cast<idx_t>(n), data);
    flat.is_trained = true;

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    flat.Search(static_cast<idx_t>(nq), queries, static_cast<idx_t>(k),
                distances.data(), labels.data());

    return labels;
}

}  // namespace hypervec