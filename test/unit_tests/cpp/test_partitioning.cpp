/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <hypervec/utils/AlignedTable.h>
#include <hypervec/utils/partitioning.h>

using namespace hypervec;

using AlignedTableUint16 = AlignedTable<uint16_t>;

// TODO: This test fails when hypervec is compiled with
// GCC 13.2 from conda-forge with AVX2 enabled. This may be
// a GCC bug that needs to be investigated further.
// As of 16-AUG-2023 the hypervec conda packages are built
// with GCC 11.2, so the published binaries are not affected.
TEST(TestPartitioning, TestPartitioningBigRange) {
    auto n = 1024;
    AlignedTableUint16 tab(n);
    for (auto i = 0; i < n; i++) {
        tab[i] = i * 64;
    }
    int32_t hist[16]{};
    simd_histogram_16(tab.get(), n, 0, 12, hist);
    for (auto i = 0; i < 16; i++) {
        ASSERT_EQ(hist[i], 64);
    }
}
