/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/graph/graph_validation.h>
#include <index/nsw/index_nsw.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <utils/common/result_handler.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

std::vector<float> MakeVectors(idx_t count, idx_t dimension) {
  std::vector<float> vectors(static_cast<size_t>(count * dimension));
  for (idx_t row = 0; row < count; ++row) {
    for (idx_t column = 0; column < dimension; ++column) {
      vectors[static_cast<size_t>(row * dimension + column)] =
          static_cast<float>(((row + 1) * (column * 17 + 13)) % 997) / 997.0F;
    }
  }
  return vectors;
}

std::unique_ptr<Index> Roundtrip(const Index& source) {
  VectorIOWriter writer;
  WriteIndex(&source, &writer);
  VectorIOReader reader;
  reader.data = std::move(writer.data);
  return ReadIndexUp(&reader);
}

TEST(IndexNSWFlat, RepeatedAddBuildsReachableSingleLayerGraph) {
  constexpr idx_t kDimension = 4;
  constexpr idx_t kCount = 48;
  const std::vector<float> vectors = MakeVectors(kCount, kDimension);
  IndexNSWFlat index(kDimension, 4);
  index.ef_construction = 24;
  index.ef_search = 24;

  index.Add(17, vectors.data());
  index.Add(kCount - 17, vectors.data() + 17 * kDimension);
  EXPECT_NO_THROW(index.CheckConsistency());
  EXPECT_EQ(index.n_total, kCount);
  EXPECT_EQ(index.graph.NodeCount(), static_cast<size_t>(kCount));
  EXPECT_EQ(index.graph.MaxDegree(), 8U);
  const GraphValidationReport report =
      ValidateGraph(index.graph, index.entry_point);
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, static_cast<size_t>(kCount));

  SearchParametersNSW params;
  params.ef_search = static_cast<int>(kCount);
  std::vector<float> distances(static_cast<size_t>(kCount));
  std::vector<idx_t> labels(static_cast<size_t>(kCount));
  index.Search(kCount, vectors.data(), 1, distances.data(), labels.data(),
               &params);
  for (idx_t row = 0; row < kCount; ++row) {
    EXPECT_EQ(labels[static_cast<size_t>(row)], row);
    EXPECT_NEAR(distances[static_cast<size_t>(row)], 0.0F, 1e-6F);
  }
}

TEST(IndexNSWFlat, SimilarityResultsAndFilteringUsePublicMetricSemantics) {
  const std::array<float, 8> vectors = {
      1.0F, 0.0F, 0.5F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F,
  };
  const std::array<float, 2> query = {1.0F, 0.0F};
  IndexNSWFlat index(2, 2, kMetricInnerProduct);
  index.Add(4, vectors.data());

  float distance = 0.0F;
  idx_t label = -1;
  SearchParametersNSW params;
  params.ef_search = 8;
  index.Search(1, query.data(), 1, &distance, &label, &params);
  EXPECT_EQ(label, 0);
  EXPECT_FLOAT_EQ(distance, 1.0F);

  using Handler = HeapBlockResultHandler<CMin<float, idx_t>>;
  Handler block(1, &distance, &label, 1);
  Handler::SingleResultHandler result(block);
  result.begin(0);
  index.Search1(query.data(), result, &params);
  result.end();
  EXPECT_EQ(label, 0);
  EXPECT_FLOAT_EQ(distance, 1.0F);

  IDSelectorRange exclude_zero(1, 4);
  params.sel = &exclude_zero;
  index.Search(1, query.data(), 1, &distance, &label, &params);
  EXPECT_EQ(label, 1);
  EXPECT_FLOAT_EQ(distance, 0.5F);
}

TEST(IndexNSWFlat, PersistenceRestoresGraphAndRuntimeDefaults) {
  constexpr idx_t kDimension = 3;
  constexpr idx_t kCount = 24;
  const std::vector<float> vectors = MakeVectors(kCount, kDimension);
  IndexNSWFlat source(kDimension, 3, kMetricL1);
  source.ef_construction = 19;
  source.ef_search = 11;
  source.Add(kCount, vectors.data());

  std::unique_ptr<Index> restored_base = Roundtrip(source);
  auto* restored = dynamic_cast<IndexNSWFlat*>(restored_base.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->graph_degree, 3);
  EXPECT_EQ(restored->ef_construction, 19);
  EXPECT_EQ(restored->ef_search, 11);
  EXPECT_NO_THROW(restored->CheckConsistency());

  SearchParametersNSW params;
  params.ef_search = static_cast<int>(kCount);
  float distance = -1.0F;
  idx_t label = -1;
  restored->Search(1, vectors.data() + 7 * kDimension, 1, &distance, &label,
                   &params);
  EXPECT_EQ(label, 7);
  EXPECT_FLOAT_EQ(distance, 0.0F);
}

TEST(IndexNSWFlat, ValidatesParametersAndResetRestoresEmptyState) {
  EXPECT_THROW(IndexNSWFlat(4, 1), HypervecException);
  IndexNSWFlat index(2, 2);
  const std::array<float, 4> vectors = {0.0F, 0.0F, 1.0F, 1.0F};
  index.Add(2, vectors.data());

  SearchParametersNSW invalid;
  invalid.ef_search = 0;
  float distance = 0.0F;
  idx_t label = -1;
  EXPECT_THROW(index.Search(1, vectors.data(), 1, &distance, &label, &invalid),
               HypervecException);

  index.Reset();
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.entry_point, kInvalidGraphId);
  EXPECT_EQ(index.graph.NodeCount(), 0U);
  EXPECT_NO_THROW(index.CheckConsistency());
}

}  // namespace
}  // namespace hypervec
