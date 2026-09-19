/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

// IndexFactory and IndexIdToString are not compiled in the HNSW-only build.
TEST(TestFactoryTools, TestReverseIndexFactory) {
    GTEST_SKIP() << "Full IndexFactory not available in HNSW-only build.";
}
