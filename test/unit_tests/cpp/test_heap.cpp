/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <utils/structures/heap.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>

using namespace hypervec;

TEST(Heap, AddnWithIds) {
    size_t n = 1000;
    size_t k = 1;
    std::vector<int64_t> heap_labels(n, -1);
    std::vector<float> heap_distances(n, 0);
    float_minheap_array_t heaps = {
            n, k, heap_labels.data(), heap_distances.data()};
    heaps.Heapify();
    std::vector<int64_t> labels(n, 1);
    std::vector<float> distances(n, 0.0f);
    std::vector<int64_t> subset(n);
    std::iota(subset.begin(), subset.end(), 0);
    heaps.AddnWithIds(1, distances.data(), labels.data(), 1);
    heaps.Reorder();
    EXPECT_TRUE(
            std::all_of(heap_labels.begin(), heap_labels.end(), [](int64_t i) {
                return i == 1;
            }));
}

TEST(Heap, AddnQuerySubsetWithIds) {
    size_t n = 20000000; // more than 2^24
    size_t k = 1;
    std::vector<int64_t> heap_labels(n, -1);
    std::vector<float> heap_distances(n, 0);
    float_minheap_array_t heaps = {
            n, k, heap_labels.data(), heap_distances.data()};
    heaps.Heapify();
    std::vector<int64_t> labels(n, 1);
    std::vector<float> distances(n, 0.0f);
    std::vector<int64_t> subset(n);
    std::iota(subset.begin(), subset.end(), 0);
    heaps.AddnQuerySubsetWithIds(
            n, subset.data(), 1, distances.data(), labels.data(), 1);
    heaps.Reorder();
    EXPECT_TRUE(
            std::all_of(heap_labels.begin(), heap_labels.end(), [](int64_t i) {
                return i == 1;
            }));
}
