/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <hypervec/utils/utils.h>

TEST(Threading, openmp) {
    EXPECT_TRUE(hypervec::check_openmp());
}
