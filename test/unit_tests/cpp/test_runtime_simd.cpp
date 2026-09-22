/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <utils/distances/distances.h>
#include <utils/log/exception.h>
#include <utils/simd/simd_levels.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace hypervec {
namespace {

struct RestoreLevel {
  SIMDLevel saved = SIMDConfig::get_level();
  ~RestoreLevel() { SIMDConfig::set_level(saved); }
};

TEST(RuntimeSimd, Environment) {
#ifdef HYPERVEC_ENABLE_DD
  const char* env = std::getenv("HYPERVEC_SIMD_LEVEL");
  if (env != nullptr &&
      (std::string(env) == "INVALID" || std::string(env) == "AVX512" ||
       (std::string(env) == "AVX2" &&
        !SIMDConfig::is_simd_level_available(SIMDLevel::AVX2)))) {
    EXPECT_THROW(SIMDConfig::get_level(), HypervecException);
    return;
  }
  if (env != nullptr && std::string(env) == "NONE") {
    EXPECT_EQ(SIMDConfig::get_level(), SIMDLevel::NONE);
  }
  if (env != nullptr && std::string(env) == "AVX2") {
    EXPECT_EQ(SIMDConfig::get_level(), SIMDLevel::AVX2);
  }
#endif
  EXPECT_EQ(SIMDConfig::get_level(), SIMDConfig::get_dispatched_level());
}

TEST(RuntimeSimd, RejectsInvalidLevelsWithoutChangingSelection) {
  const RestoreLevel restore;
  const SIMDLevel before = SIMDConfig::get_level();
  for (auto level : {static_cast<SIMDLevel>(-1), SIMDLevel::COUNT,
                     static_cast<SIMDLevel>(100)}) {
    EXPECT_FALSE(SIMDConfig::is_simd_level_available(level));
    EXPECT_THROW(SIMDConfig::set_level(level), HypervecException);
    EXPECT_EQ(SIMDConfig::get_level(), before);
  }
#ifdef HYPERVEC_ENABLE_DD
  const char* unavailable = "AVX512";
  EXPECT_THROW(SIMDConfig config(&unavailable), HypervecException);
  EXPECT_EQ(SIMDConfig::get_level(), before);
#endif
}

TEST(RuntimeSimd, DistancesTailsUnalignedAndBatch) {
  const RestoreLevel restore;
  std::mt19937 random(123);
  std::uniform_real_distribution<float> value(-1.0F, 1.0F);
  std::vector<size_t> dimensions = {99, 128, 299, 300, 301, 768};
  for (size_t d = 0; d <= 65; ++d) dimensions.push_back(d);
  for (auto level : {SIMDLevel::NONE, SIMDLevel::AVX2}) {
    if (!SIMDConfig::is_simd_level_available(level)) continue;
    SIMDConfig::set_level(level);
    for (size_t d : dimensions) {
      for (size_t offset : {size_t{0}, size_t{1}, size_t{3}}) {
        SCOPED_TRACE(::testing::Message()
                     << to_string(level) << " d=" << d << " offset=" << offset);
        // End exactly at the allocation boundary so ASan detects tail
        // overreads.
        auto xs = std::make_unique<float[]>(d + offset + 1);
        auto ys = std::make_unique<float[]>(4 * d + offset + 1);
        float* x = xs.get() + offset + 1;
        float* y = ys.get() + offset + 1;
        for (size_t j = 0; j < d; ++j) x[j] = value(random);
        for (size_t j = 0; j < 4 * d; ++j) y[j] = value(random);
        float l2[4], ip[4], batch_l2[4], batch_ip[4];
        fvec_L2sqr_ny(l2, x, y, d, 4);
        fvec_inner_products_ny(ip, x, y, d, 4);
        fvec_L2sqr_batch_4(x, y, y + d, y + 2 * d, y + 3 * d, d, batch_l2[0],
                           batch_l2[1], batch_l2[2], batch_l2[3]);
        fvec_inner_product_batch_4(x, y, y + d, y + 2 * d, y + 3 * d, d,
                                   batch_ip[0], batch_ip[1], batch_ip[2],
                                   batch_ip[3]);
        for (size_t i = 0; i < 4; ++i) {
          double ref_l2 = 0, ref_ip = 0;
          for (size_t j = 0; j < d; ++j) {
            double diff = static_cast<double>(x[j]) - y[i * d + j];
            ref_l2 += diff * diff;
            ref_ip += static_cast<double>(x[j]) * y[i * d + j];
          }
          const double tolerance = 2e-6 * std::max(size_t{1}, d);
          EXPECT_NEAR(fvec_L2sqr(x, y + i * d, d), ref_l2, tolerance);
          EXPECT_NEAR(fvec_inner_product(x, y + i * d, d), ref_ip, tolerance);
          EXPECT_NEAR(l2[i], ref_l2, tolerance);
          EXPECT_NEAR(ip[i], ref_ip, tolerance);
          EXPECT_NEAR(batch_l2[i], ref_l2, tolerance);
          EXPECT_NEAR(batch_ip[i], ref_ip, tolerance);
        }
      }
    }
  }
}

TEST(RuntimeSimd, PreservesNonFiniteDistanceSemantics) {
  const RestoreLevel restore;
  for (auto level : {SIMDLevel::NONE, SIMDLevel::AVX2}) {
    if (!SIMDConfig::is_simd_level_available(level)) continue;
    SIMDConfig::set_level(level);
    std::vector<float> x(33, 1), y(33, 2);
    y[17] = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(std::isinf(fvec_L2sqr(x.data(), y.data(), x.size())));
    EXPECT_TRUE(std::isinf(fvec_inner_product(x.data(), y.data(), x.size())));
    y[17] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(std::isnan(fvec_L2sqr(x.data(), y.data(), x.size())));
    EXPECT_TRUE(std::isnan(fvec_inner_product(x.data(), y.data(), x.size())));
  }
}

TEST(RuntimeSimd, ConcurrentSelectionAndDistanceCalls) {
  const RestoreLevel restore;
  const auto fastest = SIMDConfig::auto_detect_simd_level();
  std::vector<std::thread> workers;
  for (int t = 0; t < 4; ++t) {
    workers.emplace_back([fastest] {
      const float x[9] = {}, y[9] = {1, 2, 3};
      for (int i = 0; i < 1000; ++i) {
        SIMDConfig::set_level(i % 2 == 0 ? SIMDLevel::NONE : fastest);
        EXPECT_EQ(fvec_L2sqr(x, y, 9), 14);
      }
    });
  }
  for (auto& worker : workers) worker.join();
}

}  // namespace
}  // namespace hypervec
