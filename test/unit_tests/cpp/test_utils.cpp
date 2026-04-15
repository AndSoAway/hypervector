/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <core/index.h>
#include <utils/utils.h>

TEST(TestUtils, GetVersion) {
    std::string version = std::to_string(HYPERVEC_VERSION_MAJOR) + "." +
            std::to_string(HYPERVEC_VERSION_MINOR) + "." +
            std::to_string(HYPERVEC_VERSION_PATCH);

    EXPECT_EQ(version, hypervec::GetVersion());
}
