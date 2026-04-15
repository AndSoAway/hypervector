/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include <hypervec/Index.h>
#include <hypervec/IndexFlat.h>
#include <hypervec/IndexNSG.h>

#include <hypervec/utils/random.h>

using namespace hypervec;

TEST(IoCorrupted, NSGNeighborIdEqualsNtotal) {
    // Build an NSG index
    int d = 8, nb = 500;
    std::vector<float> xb(nb * d);
    hypervec::rand_smooth_vectors(nb, d, xb.data(), 1234);

    IndexNSGFlat index(d, 16);
    index.Add(nb, xb.data());

    // Verify that NSG::search_on_graph() skips corrupt neighbor entries
    // that would trigger access outside/above the bounds of the index by
    // setting the LAST valid neighbor of every node to ntotal.
    // get_neighbors returns neighbors until it hits a -1 sentinel,
    // so setting the last non-sentinel entry preserves graph connectivity
    // while ensuring the inner Search loop encounters id == ntotal.
    auto* graph = index.nsg.final_graph.get();
    for (int i = 0; i < nb; i++) {
        // Find the last valid (non-sentinel) neighbor slot
        int last = -1;
        for (int j = 0; j < graph->K; j++) {
            if (graph->at(i, j) >= 0) {
                last = j;
            } else {
                break;
            }
        }
        if (last >= 0) {
            graph->at(i, last) = nb; // inject ntotal
        }
    }

    std::vector<float> xq(d);
    hypervec::rand_smooth_vectors(1, d, xq.data(), 5678);
    std::vector<idx_t> I(1);
    std::vector<float> D(1);

    // Safely skipped without an exception or ASAN failure.
    EXPECT_NO_THROW(index.Search(1, xq.data(), 1, D.data(), I.data()));

    // Result must be a valid node ID
    EXPECT_GE(I[0], 0);
    EXPECT_LT(I[0], nb);
}
