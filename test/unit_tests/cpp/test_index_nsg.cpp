/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/graph/graph_validation.h>
#include <index/nsg/index_nsg.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace {

hypervec::NSGIndexOptions ExhaustiveOptions() {
  hypervec::NSGIndexOptions options;
  options.knn_degree = 8;
  options.nn_descent_iterations = 8;
  options.nn_descent_convergence_threshold = 0.0;
  options.random_seed = 42;
  options.max_degree = 8;
  options.build_search_width = 8;
  options.candidate_pool_size = 16;
  options.ef_search = 16;
  return options;
}

void ExpectSameSearch(const hypervec::Index& expected,
                      const hypervec::Index& actual,
                      const std::vector<float>& queries, hypervec::idx_t k) {
  const hypervec::idx_t count =
      static_cast<hypervec::idx_t>(queries.size()) / expected.d;
  std::vector<float> expected_distances(static_cast<size_t>(count * k));
  std::vector<float> actual_distances(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> expected_labels(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> actual_labels(static_cast<size_t>(count * k));

  expected.Search(count, queries.data(), k, expected_distances.data(),
                  expected_labels.data());
  actual.Search(count, queries.data(), k, actual_distances.data(),
                actual_labels.data());

  EXPECT_EQ(actual_labels, expected_labels);
  ASSERT_EQ(actual_distances.size(), expected_distances.size());
  for (size_t result = 0; result < actual_distances.size(); ++result) {
    EXPECT_FLOAT_EQ(actual_distances[result], expected_distances[result]);
  }
}

}  // namespace

TEST(IndexNSG, FlatL2BuildsReachableGraphAndMatchesExhaustiveSearch) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  const std::vector<float> queries = {4.0F, 16.0F};
  hypervec::IndexFlatL2 expected(1);
  expected.Add(static_cast<hypervec::idx_t>(database.size()), database.data());
  hypervec::IndexNSGFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());

  index.Build(static_cast<hypervec::idx_t>(database.size()), database.data());

  EXPECT_EQ(index.n_total, 6);
  EXPECT_EQ(index.CodeStore().Size(), 6);
  EXPECT_EQ(index.Graph().NodeCount(), 6U);
  EXPECT_EQ(index.EntryPoint(), 3);
  EXPECT_EQ(index.BuildStats().candidate_graph.initial_distance_computations,
            30U);
  EXPECT_EQ(index.BuildStats().nsg.pruned_nodes, 6U);
  const auto report =
      hypervec::ValidateGraph(index.Graph(), index.EntryPoint());
  EXPECT_TRUE(report.IsStructurallyValid());
  EXPECT_EQ(report.reachable_nodes, 6U);
  ExpectSameSearch(expected, index, queries, 3);
}

TEST(IndexNSG, ParallelCandidateBuildPreservesSearchAndGraph) {
  constexpr hypervec::idx_t kCount = 128;
  std::vector<float> database(static_cast<size_t>(kCount));
  for (hypervec::idx_t node = 0; node < kCount; ++node) {
    database[static_cast<size_t>(node)] =
        static_cast<float>((node * 37) % kCount);
  }
  auto options = ExhaustiveOptions();
  hypervec::IndexNSGFlat baseline(1, hypervec::kMetricL2, options);
  baseline.Build(kCount, database.data());

  options.build_threads = 4;
  hypervec::IndexNSGFlat parallel(1, hypervec::kMetricL2, options);
  parallel.Build(kCount, database.data());
  EXPECT_EQ(parallel.EntryPoint(), baseline.EntryPoint());
  EXPECT_EQ(parallel.BuildStats().candidate_graph.neighbor_updates,
            baseline.BuildStats().candidate_graph.neighbor_updates);
  for (hypervec::idx_t node = 0; node < kCount; ++node) {
    const auto serial_neighbors = baseline.Graph().Neighbors(node);
    const auto parallel_neighbors = parallel.Graph().Neighbors(node);
    EXPECT_TRUE(std::equal(serial_neighbors.begin(), serial_neighbors.end(),
                           parallel_neighbors.begin(),
                           parallel_neighbors.end()));
  }
  ExpectSameSearch(baseline, parallel, database, 5);
  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&parallel, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  const auto restored = hypervec::ReadIndexUp(&reader);
  const auto* restored_nsg =
      dynamic_cast<const hypervec::IndexNSGFlat*>(restored.get());
  ASSERT_NE(restored_nsg, nullptr);
  EXPECT_EQ(restored_nsg->Options().build_threads, 1U);
  ExpectSameSearch(parallel, *restored_nsg, database, 5);
}

TEST(IndexNSG, SimilaritySearchRestoresExternalDistanceDirection) {
  const std::vector<float> database = {
      1.0F, 0.0F, 0.0F, 2.0F, 2.0F, 1.0F, 3.0F, 4.0F, 1.0F, 3.0F, 6.0F, 2.0F,
  };
  const std::vector<float> queries = {1.0F, 1.0F, 2.0F, 0.5F};
  hypervec::IndexFlatIP expected(2);
  expected.Add(6, database.data());
  hypervec::IndexNSGFlat index(2, hypervec::kMetricInnerProduct,
                               ExhaustiveOptions());

  index.Build(6, database.data());

  ExpectSameSearch(expected, index, queries, 3);
  std::unique_ptr<hypervec::DistanceComputer> distance(
      index.GetDistanceComputer());
  distance->SetQuery(queries.data());
  EXPECT_FLOAT_EQ((*distance)(5), 8.0F);
}

TEST(IndexNSG, PersistenceRoundtripPreservesStaticState) {
  hypervec::NSGIndexOptions options = ExhaustiveOptions();
  options.knn_degree = 5;
  options.nn_descent_iterations = 7;
  options.nn_descent_convergence_threshold = 0.125;
  options.random_seed = 123456789;
  options.max_degree = 4;
  options.build_search_width = 6;
  options.candidate_pool_size = 9;
  options.ef_search = 7;
  options.check_relative_distance = false;
  const std::vector<float> database = {
      1.0F, 0.0F, 0.0F, 2.0F, 2.0F, 1.0F, 3.0F, 4.0F, 1.0F, 3.0F, 6.0F, 2.0F,
  };
  const std::vector<float> queries = {1.0F, 1.0F, 2.0F, 0.5F};
  hypervec::IndexNSGFlat source(2, hypervec::kMetricInnerProduct, options);
  source.Build(6, database.data());

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> restored_base =
      hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexNSGFlat*>(restored_base.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->d, source.d);
  EXPECT_EQ(restored->n_total, source.n_total);
  EXPECT_EQ(restored->metric_type, source.metric_type);
  EXPECT_EQ(restored->EntryPoint(), source.EntryPoint());
  EXPECT_EQ(restored->Options().knn_degree, options.knn_degree);
  EXPECT_EQ(restored->Options().nn_descent_iterations,
            options.nn_descent_iterations);
  EXPECT_DOUBLE_EQ(restored->Options().nn_descent_convergence_threshold,
                   options.nn_descent_convergence_threshold);
  EXPECT_EQ(restored->Options().random_seed, options.random_seed);
  EXPECT_EQ(restored->Options().max_degree, options.max_degree);
  EXPECT_EQ(restored->Options().build_search_width, options.build_search_width);
  EXPECT_EQ(restored->Options().candidate_pool_size,
            options.candidate_pool_size);
  EXPECT_EQ(restored->Options().ef_search, options.ef_search);
  EXPECT_EQ(restored->Options().check_relative_distance,
            options.check_relative_distance);
  ASSERT_EQ(restored->Graph().NodeCount(), source.Graph().NodeCount());
  EXPECT_EQ(restored->Graph().MaxDegree(), source.Graph().MaxDegree());
  for (size_t node = 0; node < source.Graph().NodeCount(); ++node) {
    const auto source_neighbors =
        source.Graph().Neighbors(static_cast<hypervec::GraphId>(node));
    const auto restored_neighbors =
        restored->Graph().Neighbors(static_cast<hypervec::GraphId>(node));
    EXPECT_TRUE(std::equal(source_neighbors.begin(), source_neighbors.end(),
                           restored_neighbors.begin(),
                           restored_neighbors.end()));
  }
  EXPECT_EQ(restored->BuildStats().candidate_graph.iterations, 0U);
  EXPECT_EQ(restored->BuildStats().nsg.pruned_nodes, 0U);
  ExpectSameSearch(source, *restored, queries, 3);
  EXPECT_THROW(restored->Add(1, database.data()), hypervec::HypervecException);
}

TEST(IndexNSG, PersistenceRoundtripPreservesEmptyLpState) {
  hypervec::IndexNSGFlat source(2, hypervec::kMetricLp, ExhaustiveOptions(),
                                3.0F);

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  hypervec::VectorIOReader reader;
  reader.data = writer.data;
  std::unique_ptr<hypervec::Index> restored_base =
      hypervec::ReadIndexUp(&reader);
  auto* restored = dynamic_cast<hypervec::IndexNSGFlat*>(restored_base.get());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->n_total, 0);
  EXPECT_EQ(restored->EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(restored->Graph().NodeCount(), 0U);
  EXPECT_EQ(restored->metric_type, hypervec::kMetricLp);
  EXPECT_FLOAT_EQ(restored->metric_arg, 3.0F);

  const std::array<float, 4> database = {1.0F, 2.0F, 3.0F, 4.0F};
  restored->Build(2, database.data());
  EXPECT_EQ(restored->n_total, 2);
}

TEST(IndexNSG, ApproximatePipelineRetainsHighRecall) {
  constexpr size_t kCount = 128;
  constexpr hypervec::idx_t kQueryCount = 16;
  constexpr hypervec::idx_t k = 5;
  std::vector<float> database(kCount * 2);
  for (size_t node = 0; node < kCount; ++node) {
    database[node * 2] = static_cast<float>((node * 37) % kCount);
    database[node * 2 + 1] =
        static_cast<float>((node * 53) % (kCount - 1)) * 0.01F;
  }
  std::vector<float> queries(static_cast<size_t>(kQueryCount) * 2);
  for (hypervec::idx_t query = 0; query < kQueryCount; ++query) {
    const size_t source = static_cast<size_t>(query * 7) % kCount;
    queries[static_cast<size_t>(query) * 2] = database[source * 2] + 0.2F;
    queries[static_cast<size_t>(query) * 2 + 1] =
        database[source * 2 + 1] - 0.15F;
  }

  hypervec::IndexFlatL2 expected(2);
  expected.Add(kCount, database.data());
  hypervec::NSGIndexOptions options;
  options.knn_degree = 10;
  options.nn_descent_iterations = 15;
  options.random_seed = 42;
  options.max_degree = 8;
  options.build_search_width = 20;
  options.candidate_pool_size = 30;
  options.ef_search = 24;
  hypervec::IndexNSGFlat index(2, hypervec::kMetricL2, options);
  index.Build(kCount, database.data());

  std::vector<float> expected_distances(kQueryCount * k);
  std::vector<float> actual_distances(kQueryCount * k);
  std::vector<hypervec::idx_t> expected_labels(kQueryCount * k);
  std::vector<hypervec::idx_t> actual_labels(kQueryCount * k);
  expected.Search(kQueryCount, queries.data(), k, expected_distances.data(),
                  expected_labels.data());
  index.Search(kQueryCount, queries.data(), k, actual_distances.data(),
               actual_labels.data());

  size_t matches = 0;
  for (hypervec::idx_t query = 0; query < kQueryCount; ++query) {
    const auto expected_begin =
        expected_labels.begin() + static_cast<size_t>(query * k);
    for (hypervec::idx_t result = 0; result < k; ++result) {
      const hypervec::idx_t actual =
          actual_labels[static_cast<size_t>(query * k + result)];
      matches +=
          static_cast<size_t>(std::find(expected_begin, expected_begin + k,
                                        actual) != expected_begin + k);
    }
  }
  EXPECT_GE(static_cast<double>(matches) / static_cast<double>(kQueryCount * k),
            0.90);
}

TEST(IndexNSG, EnforcesStaticLifecycleAndSupportsResetRebuild) {
  const std::array<float, 4> database = {1.0F, 2.0F, 3.0F, 4.0F};
  hypervec::IndexNSGFlat index(2, hypervec::kMetricL2, ExhaustiveOptions());

  EXPECT_THROW(index.Add(2, database.data()), hypervec::HypervecException);
  index.Build(2, database.data());
  EXPECT_THROW(index.Build(2, database.data()), hypervec::HypervecException);

  std::array<float, 2> reconstructed;
  index.Reconstruct(1, reconstructed.data());
  EXPECT_EQ(reconstructed, (std::array<float, 2>{3.0F, 4.0F}));
  const auto capabilities = index.GetCapabilities();
  EXPECT_FALSE(capabilities.requires_training);
  EXPECT_TRUE(capabilities.supports_reconstruct);
  EXPECT_FALSE(capabilities.supports_range_search);

  index.Reset();
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.CodeStore().Size(), 0);
  EXPECT_EQ(index.Graph().NodeCount(), 0U);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(index.BuildStats().nsg.pruned_nodes, 0U);
  index.Build(2, database.data(), 1, database.data());
  EXPECT_EQ(index.n_total, 2);
}

TEST(IndexNSG, SearchParametersFilterResultsAndValidateInputs) {
  const std::vector<float> database = {0.0F, 2.0F, 5.0F, 9.0F, 14.0F, 20.0F};
  hypervec::IndexNSGFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());
  index.Build(6, database.data());

  hypervec::IDSelectorRange selector(2, 5);
  hypervec::SearchParametersNSG params;
  params.ef_search = 8;
  params.sel = &selector;
  std::array<float, 4> distances;
  std::array<hypervec::idx_t, 4> labels;
  const float query = 1.0F;
  index.Search(1, &query, 4, distances.data(), labels.data(), &params);
  EXPECT_EQ(labels[0], 2);
  EXPECT_EQ(labels[1], 3);
  EXPECT_EQ(labels[2], 4);
  EXPECT_EQ(labels[3], -1);
  EXPECT_TRUE(std::isinf(distances[3]));

  params.ef_search = 0;
  EXPECT_THROW(
      index.Search(1, &query, 1, distances.data(), labels.data(), &params),
      hypervec::HypervecException);
  EXPECT_THROW(index.Search(-1, nullptr, 1, nullptr, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.Search(1, nullptr, 1, distances.data(), labels.data()),
               hypervec::HypervecException);
  EXPECT_THROW(index.Reconstruct(6, distances.data()),
               hypervec::HypervecException);
}

TEST(IndexNSG, FailedBuildDoesNotPublishCodesGraphOrStatistics) {
  const std::array<float, 3> database = {
      0.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
  hypervec::IndexNSGFlat index(1, hypervec::kMetricL2, ExhaustiveOptions());

  EXPECT_THROW(index.Build(3, database.data()), hypervec::HypervecException);
  EXPECT_EQ(index.n_total, 0);
  EXPECT_EQ(index.CodeStore().Size(), 0);
  EXPECT_EQ(index.Graph().NodeCount(), 0U);
  EXPECT_EQ(index.EntryPoint(), hypervec::kInvalidGraphId);
  EXPECT_EQ(index.BuildStats().candidate_graph.iterations, 0U);
  EXPECT_EQ(index.BuildStats().nsg.pruned_nodes, 0U);
}

TEST(IndexNSG, ValidatesConstructionBuildAndQueryAwareInputs) {
  hypervec::NSGIndexOptions zero_search = ExhaustiveOptions();
  zero_search.ef_search = 0;
  EXPECT_THROW((hypervec::IndexNSGFlat(2, hypervec::kMetricL2, zero_search)),
               hypervec::HypervecException);

  hypervec::NSGIndexOptions narrow_pool = ExhaustiveOptions();
  narrow_pool.max_degree = 8;
  narrow_pool.candidate_pool_size = 4;
  EXPECT_THROW((hypervec::IndexNSGFlat(2, hypervec::kMetricL2, narrow_pool)),
               hypervec::HypervecException);
  EXPECT_THROW((hypervec::IndexNSG(nullptr, ExhaustiveOptions())),
               hypervec::HypervecException);

  hypervec::IndexNSGFlat index(2, hypervec::kMetricL2, ExhaustiveOptions());
  const std::array<float, 4> database = {1.0F, 2.0F, 3.0F, 4.0F};
  EXPECT_THROW(index.Build(0, database.data()), hypervec::HypervecException);
  EXPECT_THROW(index.Build(2, nullptr), hypervec::HypervecException);
  EXPECT_THROW(index.Build(2, database.data(), -1, nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(index.Build(2, database.data(), 1, nullptr),
               hypervec::HypervecException);

  float empty_distance = 0.0F;
  hypervec::idx_t empty_label = 0;
  index.Search(1, database.data(), 1, &empty_distance, &empty_label);
  EXPECT_TRUE(std::isinf(empty_distance));
  EXPECT_EQ(empty_label, -1);
}

TEST(NSGIndexBuildStats, ResetAndCombineCoverBothBuildPhases) {
  hypervec::NSGIndexBuildStats aggregate;
  hypervec::NSGIndexBuildStats update;
  update.candidate_graph.iterations = 2;
  update.nsg.pruned_nodes = 3;

  aggregate.Combine(update);
  EXPECT_EQ(aggregate.candidate_graph.iterations, 2U);
  EXPECT_EQ(aggregate.nsg.pruned_nodes, 3U);

  aggregate.Reset();
  EXPECT_EQ(aggregate.candidate_graph.iterations, 0U);
  EXPECT_EQ(aggregate.nsg.pruned_nodes, 0U);
}
