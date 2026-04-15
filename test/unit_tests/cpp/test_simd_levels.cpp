/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

// Core SIMDConfig API tests - works in both static and DD modes.
// Hardware execution tests (DD-only) are in separate files:
// - test_simd_levels_x86_avx2.cpp (compiled with AVX2 flags)
// - test_simd_levels_x86_avx512.cpp (compiled with AVX512 flags)

#include <gtest/gtest.h>

#include <hypervec/impl/HypervecException.h>
#include <hypervec/utils/simd_levels.h>
#include <hypervec/utils/utils.h>

// Helper to check if we're in DD mode
static bool is_dd_mode() {
    return hypervec::get_compile_options().find("DD") != std::string::npos;
}

TEST(SIMDConfig, get_level_returns_valid_level) {
    // Works in both static and DD modes
    hypervec::SIMDLevel level = hypervec::SIMDConfig::get_level();
    EXPECT_NE(level, hypervec::SIMDLevel::COUNT);
    EXPECT_GE(static_cast<int>(level), 0);
    EXPECT_LT(
            static_cast<int>(level), static_cast<int>(hypervec::SIMDLevel::COUNT));
}

TEST(SIMDConfig, supported_simd_levels_not_empty) {
    // Works in both static and DD modes
    // Current level should always be in supported levels
    EXPECT_TRUE(
            hypervec::SIMDConfig::is_simd_level_available(
                    hypervec::SIMDConfig::get_level()));
}

TEST(SIMDConfig, set_level_to_supported_level_succeeds) {
    // Works in both static and DD modes
    hypervec::SIMDLevel original_level = hypervec::SIMDConfig::get_level();

    // Setting to any supported level should succeed
    for (int i = 0; i < static_cast<int>(hypervec::SIMDLevel::COUNT); ++i) {
        auto level = static_cast<hypervec::SIMDLevel>(i);
        if (hypervec::SIMDConfig::is_simd_level_available(level)) {
            EXPECT_NO_THROW(hypervec::SIMDConfig::set_level(level))
                    << "set_level(" << hypervec::to_string(level)
                    << ") should succeed";
            EXPECT_EQ(hypervec::SIMDConfig::get_level(), level);
        }
    }

    // Restore original level
    hypervec::SIMDConfig::set_level(original_level);
}

TEST(SIMDConfig, set_level_to_unsupported_level_throws) {
    // Works in both static and DD modes
    // Find a level that's NOT supported
    hypervec::SIMDLevel unsupported = hypervec::SIMDLevel::COUNT;
    for (int i = 0; i < static_cast<int>(hypervec::SIMDLevel::COUNT); ++i) {
        auto level = static_cast<hypervec::SIMDLevel>(i);
        if (!hypervec::SIMDConfig::is_simd_level_available(level)) {
            unsupported = level;
            break;
        }
    }

    if (unsupported != hypervec::SIMDLevel::COUNT) {
        EXPECT_THROW(
                hypervec::SIMDConfig::set_level(unsupported),
                hypervec::HypervecException)
                << "set_level(" << hypervec::to_string(unsupported)
                << ") should throw";
    }
}

TEST(SIMDConfig, static_mode_has_single_level) {
    // Static mode should have exactly 1 level: the compiled-in level
    if (is_dd_mode()) {
        GTEST_SKIP() << "DD build - has multiple levels";
    }

    int count = 0;
    for (int i = 0; i < static_cast<int>(hypervec::SIMDLevel::COUNT); ++i) {
        if (hypervec::SIMDConfig::is_simd_level_available(
                    static_cast<hypervec::SIMDLevel>(i))) {
            ++count;
        }
    }
    EXPECT_EQ(count, 1)
            << "Static mode should have exactly 1 level (compiled-in)";
}

TEST(SIMDConfig, get_level_name_matches_level) {
    // Works in both static and DD modes
    hypervec::SIMDLevel level = hypervec::SIMDConfig::get_level();
    std::string name = hypervec::SIMDConfig::get_level_name();
    std::string expected = hypervec::to_string(level);
    EXPECT_EQ(name, expected);
}

TEST(SIMDConfig, get_dispatched_level_matches_get_level) {
    // Works in both static and DD modes
    // Verifies that dispatch mechanism returns the same level as get_level()
    hypervec::SIMDLevel level = hypervec::SIMDConfig::get_level();
    hypervec::SIMDLevel dispatched = hypervec::SIMDConfig::get_dispatched_level();
    EXPECT_EQ(level, dispatched)
            << "get_level() returned " << hypervec::to_string(level)
            << " but get_dispatched_level() returned "
            << hypervec::to_string(dispatched);
}

TEST(SIMDLevel, to_string_all_levels) {
    EXPECT_EQ("NONE", hypervec::to_string(hypervec::SIMDLevel::NONE));
    EXPECT_EQ("AVX2", hypervec::to_string(hypervec::SIMDLevel::AVX2));
    EXPECT_EQ("AVX512", hypervec::to_string(hypervec::SIMDLevel::AVX512));
    EXPECT_EQ("AVX512_SPR", hypervec::to_string(hypervec::SIMDLevel::AVX512_SPR));
    EXPECT_EQ("ARM_NEON", hypervec::to_string(hypervec::SIMDLevel::ARM_NEON));
    EXPECT_EQ("ARM_SVE", hypervec::to_string(hypervec::SIMDLevel::ARM_SVE));

    // COUNT should throw
    EXPECT_THROW(
            hypervec::to_string(hypervec::SIMDLevel::COUNT), hypervec::HypervecException);
}

TEST(SIMDLevel, to_simd_level_all_strings) {
    EXPECT_EQ(hypervec::SIMDLevel::NONE, hypervec::to_simd_level("NONE"));
    EXPECT_EQ(hypervec::SIMDLevel::AVX2, hypervec::to_simd_level("AVX2"));
    EXPECT_EQ(hypervec::SIMDLevel::AVX512, hypervec::to_simd_level("AVX512"));
    EXPECT_EQ(hypervec::SIMDLevel::AVX512_SPR, hypervec::to_simd_level("AVX512_SPR"));
    EXPECT_EQ(hypervec::SIMDLevel::ARM_NEON, hypervec::to_simd_level("ARM_NEON"));
    EXPECT_EQ(hypervec::SIMDLevel::ARM_SVE, hypervec::to_simd_level("ARM_SVE"));

    // Invalid strings should throw
    EXPECT_THROW(hypervec::to_simd_level("INVALID"), hypervec::HypervecException);
    EXPECT_THROW(hypervec::to_simd_level(""), hypervec::HypervecException);
}

TEST(SIMDConfig, modern_hardware_has_simd_support) {
    // In DD mode, verify modern hardware detects SIMD support
    if (!is_dd_mode()) {
        GTEST_SKIP() << "Static build - level is fixed at compile time";
    }

    hypervec::SIMDLevel detected = hypervec::SIMDConfig::auto_detect_simd_level();

#if defined(__x86_64__) || defined(_M_X64)
    // All modern x86_64 machines (Haswell 2013+) support at least AVX2
    EXPECT_NE(detected, hypervec::SIMDLevel::NONE)
            << "x86_64 machines should support at least AVX2";
#elif defined(__aarch64__) || defined(_M_ARM64)
    // NEON is mandatory on aarch64
    EXPECT_NE(detected, hypervec::SIMDLevel::NONE)
            << "ARM64 machines should support at least NEON";
#endif
}

TEST(CompileOptions, lists_expected_levels) {
    std::string options = hypervec::get_compile_options();

    // All supported levels (except NONE) should be in compile options
    for (int i = 0; i < static_cast<int>(hypervec::SIMDLevel::COUNT); ++i) {
        auto level = static_cast<hypervec::SIMDLevel>(i);
        if (!hypervec::SIMDConfig::is_simd_level_available(level)) {
            continue;
        }
        if (level == hypervec::SIMDLevel::NONE) {
            continue; // NONE is not reported in options
        }
        std::string name = hypervec::to_string(level);
        EXPECT_NE(options.find(name), std::string::npos)
                << "Supported level " << name
                << " should be in compile options: " << options;
    }

    // DD mode should have "DD" marker
    if (is_dd_mode()) {
        EXPECT_NE(options.find("DD"), std::string::npos)
                << "DD mode should have 'DD' in compile options: " << options;
    }
}
