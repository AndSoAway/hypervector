/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/pq.h>
#include <quantization/pq/pq_distance_computer.h>
#include <utils/common/range_search_result.h>
#include <utils/selector/id_selector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

TEST(PQLookup, AllWidthsUnalignedTailsAndSignedTables) {
  for (int bits = 1; bits <= 16; ++bits) {
    for (idx_t m : {1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 75, 100}) {
      SCOPED_TRACE(::testing::Message() << "bits=" << bits << " M=" << m);
      ProductQuantizer pq(m, m, bits);
      // Include negative entries: IVF precomputed residual tables are signed.
      std::vector<float> table(static_cast<size_t>(m * pq.ksub));
      for (size_t i = 0; i < table.size(); ++i) {
        table[i] = static_cast<float>(static_cast<int>(i % 101) - 50) * 0.137F;
      }
      for (size_t offset : {size_t{1}, size_t{3}}) {
        auto storage = std::make_unique<uint8_t[]>(offset + pq.code_size);
        uint8_t* code = storage.get() + offset;
        std::fill(code, code + pq.code_size, uint8_t{0});
        double expected = 0, absolute_sum = 0;
        for (idx_t j = 0; j < m; ++j) {
          const size_t value = (j % 2 == 0) ? static_cast<size_t>(pq.ksub - 1)
                                            : static_cast<size_t>(j) % pq.ksub;
          // Independent bit packing, including the maximum valid centroid.
          for (int bit = 0; bit < bits; ++bit) {
            const size_t position = static_cast<size_t>(j * bits + bit);
            if ((value & (size_t{1} << bit)) != 0) {
              code[position / 8] |= uint8_t{1} << (position % 8);
            }
          }
          const float term = table[static_cast<size_t>(j * pq.ksub) + value];
          expected += term;
          absolute_sum += std::abs(term);
        }
        // Ignore unused high bits, including odd-M PQ4 codes.
        const size_t remainder = static_cast<size_t>(m * bits) % 8;
        if (remainder != 0) code[pq.code_size - 1] |= 0xffU << remainder;
        const double tolerance = 2e-6 * std::max(1.0, absolute_sum);
        EXPECT_NEAR(pq.ApplyDistanceTable(table.data(), code), expected,
                    tolerance);
        EXPECT_NEAR(pq.GetDistanceKernel()(pq, table.data(), code), expected,
                    tolerance);
      }
    }
  }
}

TEST(PQLookup, NonFiniteEntriesKeepTheirMeaning) {
  for (int bits : {4, 8, 16}) {
    ProductQuantizer pq(5, 5, bits);
    std::vector<uint8_t> code(pq.code_size, 0);
    std::vector<float> table(static_cast<size_t>(pq.M * pq.ksub), 1);
    table[2 * pq.ksub] = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(std::isinf(pq.ApplyDistanceTable(table.data(), code.data())));
    table[2 * pq.ksub] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(std::isnan(pq.ApplyDistanceTable(table.data(), code.data())));
  }
}

TEST(PQLookup, ScanAndDistanceComputerMatchSequentialReference) {
  constexpr idx_t d = 18, m = 9, count = 127, k = 10;
  std::mt19937 random(42);
  std::uniform_real_distribution<float> value(-2, 2);
  for (int bits : {4, 8}) {
    ProductQuantizer pq(d, m, bits);
    for (auto& v : pq.centroids) v = value(random);
    pq.is_trained = true;  // A deterministic synthetic codebook, no training.
    std::vector<float> base(count * d), query(d), table(m * pq.ksub);
    for (auto& v : base) v = value(random);
    for (auto& v : query) v = value(random);
    std::vector<uint8_t> codes(count * pq.code_size);
    pq.ComputeCodes(count, base.data(), codes.data());
    pq.ComputeDistanceTable(query.data(), table.data());
    PQDistanceComputer dc(pq, codes.data(), pq.code_size);
    dc.SetQuery(query.data());
    std::vector<std::pair<double, idx_t>> expected;
    for (idx_t i = 0; i < count; ++i) {
      PQDecoderGeneric decoder(codes.data() + i * pq.code_size, bits);
      double sum = 0;
      for (idx_t j = 0; j < m; ++j)
        sum += table[j * pq.ksub + decoder.decode()];
      EXPECT_NEAR(dc(i), sum, 2e-6 * std::max(1.0, sum));
      expected.emplace_back(sum, i);
    }
    std::sort(expected.begin(), expected.end());
    std::vector<float> distances(k);
    std::vector<idx_t> labels(k);
    pq.SearchL2(1, query.data(), count, codes.data(), k, distances.data(),
                labels.data());
    for (idx_t j = 0; j < k; ++j) {
      EXPECT_EQ(labels[j], expected[j].second);
      EXPECT_NEAR(distances[j], expected[j].first,
                  2e-6 * std::max(1.0, expected[j].first));
    }
  }
}

TEST(PQLookup, IvfSpecializationsPreserveFilteringAndRangeSearch) {
  constexpr idx_t d = 18, count = 256, target = 17;
  std::mt19937 random(123);
  std::uniform_real_distribution<float> value(-2, 2);
  std::vector<float> base(count * d);
  for (auto& v : base) v = value(random);
  for (int bits : {4, 8}) {
    for (int mode = 0; mode < 3; ++mode) {
      IndexIVFPQ index(d, 4, 9, bits);
      index.by_residual = mode != 0;
      index.use_precomputed_table = mode == 2;
      index.Build(count, base.data());
      IDSelectorRange selector(target, target + 1);
      IVFSearchParameters params;
      params.nprobe = index.nlist;
      params.sel = &selector;
      const float* query = base.data() + target * d;
      float distance;
      idx_t label;
      index.Search(1, query, 1, &distance, &label, &params);
      EXPECT_EQ(label, target);
      RangeSearchResult range(1);
      index.RangeSearch(1, query, std::numeric_limits<float>::infinity(),
                        &range, &params);
      ASSERT_EQ(range.lims[1], 1U);
      EXPECT_EQ(range.labels[0], target);
      EXPECT_FLOAT_EQ(range.distances[0], distance);
    }
  }
}

}  // namespace
}  // namespace hypervec
