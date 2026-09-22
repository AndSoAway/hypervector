/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/lvq.h>
#include <quantization/lvq/lvq_distance_computer.h>
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

TEST(RuntimeSimd, LVQDistanceTailsAndBitWidths) {
  const RestoreLevel restore;
  std::mt19937 random(456);
  std::uniform_real_distribution<float> value(-3, 3);
  std::vector<size_t> dimensions = {99, 128, 299, 300, 301, 768};
  for (size_t d = 1; d <= 65; ++d) dimensions.push_back(d);
  for (size_t d : dimensions) {
    std::vector<float> base(4 * d), query(d), decoded(d);
    for (auto& v : base) v = value(random);
    for (auto& v : query) v = value(random);
    for (int bits = 1; bits <= 8; ++bits) {
      LocalVectorQuantizer lvq(d, bits);
      lvq.Train(4, base.data());
      auto buffer = std::make_unique<float[]>(d + 1);
      float* centered = buffer.get() + 1;
      lvq.ComputeDistanceTable(query.data(), centered);
      for (size_t offset : {size_t{1}, size_t{3}}) {
        auto storage = std::make_unique<uint8_t[]>(offset + lvq.code_size);
        uint8_t* code = storage.get() + offset;
        // Encoding the mean exercises step=0, including AVX and scalar tails.
        for (const float* source : {base.data(), lvq.mean.data()}) {
          lvq.ComputeCode(source, code);
          lvq.Decode(code, decoded.data());
          double expected = 0;
          for (size_t j = 0; j < d; ++j) {
            const double diff = static_cast<double>(query[j]) - decoded[j];
            expected += diff * diff;
          }
          const double tolerance = 1e-5 * std::max(1.0, expected);
          SIMDConfig::set_level(SIMDLevel::NONE);
          const auto baseline = lvq.GetDistanceKernel();
          EXPECT_NEAR(baseline(lvq, centered, code), expected, tolerance);
          for (auto level : {SIMDLevel::NONE, SIMDLevel::AVX2}) {
            if (!SIMDConfig::is_simd_level_available(level)) continue;
            SIMDConfig::set_level(level);
            SCOPED_TRACE(::testing::Message()
                         << to_string(level) << " d=" << d << " bits=" << bits);
            if (bits != 8) EXPECT_EQ(lvq.GetDistanceKernel(), baseline);
            if (bits == 8 && level == SIMDLevel::AVX2) {
              EXPECT_NE(lvq.GetDistanceKernel(), baseline);
            }
            EXPECT_NEAR(lvq.ApplyDistanceTable(centered, code), expected,
                        tolerance);
            LVQDistanceComputer dc(lvq, code, lvq.code_size);
            dc.SetQuery(query.data());
            // Captured kernels remain valid after the global selection changes.
            SIMDConfig::set_level(SIMDLevel::NONE);
            EXPECT_NEAR(dc(0), expected, tolerance);
          }
        }
      }
    }
  }
}

TEST(RuntimeSimd, LVQScanMatchesBaseline) {
  const RestoreLevel restore;
  constexpr idx_t d = 33, n = 127, nq = 4, k = 10;
  std::vector<float> base(n * d), queries(nq * d);
  std::mt19937 random(789);
  std::uniform_real_distribution<float> value(-1, 1);
  for (auto& v : base) v = value(random);
  for (auto& v : queries) v = value(random);
  LocalVectorQuantizer lvq(d, 8);
  lvq.Train(n, base.data());
  std::vector<uint8_t> codes(n * lvq.code_size);
  lvq.ComputeCodes(n, base.data(), codes.data());
  std::vector<float> expected(nq * k), actual(nq * k);
  std::vector<idx_t> expected_ids(nq * k), actual_ids(nq * k);
  SIMDConfig::set_level(SIMDLevel::NONE);
  lvq.SearchL2(nq, queries.data(), n, codes.data(), k, expected.data(),
               expected_ids.data());
  SIMDConfig::set_level(SIMDConfig::auto_detect_simd_level());
  lvq.SearchL2(nq, queries.data(), n, codes.data(), k, actual.data(),
               actual_ids.data());
  EXPECT_EQ(expected_ids, actual_ids);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-5 * std::max(1.0F, expected[i]));
  }
}

TEST(RuntimeSimd, IVFLVQ8ResidualAndRawScansMatchBaseline) {
  const RestoreLevel restore;
  constexpr idx_t d = 33, n = 256, nq = 4, k = 10;
  std::vector<float> base(n * d), queries(nq * d);
  std::mt19937 random(987);
  std::uniform_real_distribution<float> value(-1, 1);
  for (auto& v : base) v = value(random);
  for (auto& v : queries) v = value(random);
  for (bool residual : {false, true}) {
    SIMDConfig::set_level(SIMDLevel::NONE);
    IndexIVFLVQ index(d, 4, 8);
    index.by_residual = residual;
    index.nprobe = 4;
    index.Build(n, base.data());
    std::vector<float> expected(nq * k), actual(nq * k);
    std::vector<idx_t> expected_ids(nq * k), actual_ids(nq * k);
    index.Search(nq, queries.data(), k, expected.data(), expected_ids.data());
    SIMDConfig::set_level(SIMDConfig::auto_detect_simd_level());
    index.Search(nq, queries.data(), k, actual.data(), actual_ids.data());
    EXPECT_EQ(expected_ids, actual_ids);
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], 1e-5 * std::max(1.0F, expected[i]));
    }
  }
}

}  // namespace
}  // namespace hypervec
