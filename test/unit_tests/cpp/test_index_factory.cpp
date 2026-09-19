/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/index_factory.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/lsh/index_lsh.h>
#include <index/nsg/index_nsg.h>
#include <index/nsw/index_nsw.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename IndexType>
void ExpectBuiltIn(std::string name,
                   hypervec::MetricType metric = hypervec::kMetricL2) {
  hypervec::IndexConfig config(std::move(name), 4, metric);
  config.SetInteger("nlist", 2)
      .SetInteger("m_pq", 2)
      .SetInteger("nlocal", 2)
      .SetInteger("nbits", 2)
      .SetInteger("m_hnsw", 4)
      .SetInteger("table_count", 3)
      .SetInteger("bits_per_table", 4)
      .SetInteger("probe_count", 3)
      .SetInteger("candidate_limit", 8)
      .SetInteger("knn_degree", 4)
      .SetInteger("nn_descent_iterations", 3)
      .SetDouble("nn_descent_convergence_threshold", 0.01)
      .SetInteger("random_seed", 7)
      .SetInteger("max_degree", 4)
      .SetInteger("build_search_width", 4)
      .SetInteger("candidate_pool_size", 8)
      .SetInteger("ef_construction", 8)
      .SetInteger("ef_search", 4)
      .SetBoolean("check_relative_distance", true)
      .SetBoolean("fill_to_max_degree", true);

  const auto allowed = hypervec::GetIndexRegistry().List();
  const auto type = std::find_if(allowed.begin(), allowed.end(),
                                 [&](const hypervec::IndexDescriptor& item) {
                                   return item.name == config.index_type;
                                 });
  ASSERT_NE(type, allowed.end());

  hypervec::IndexConfig filtered(config.index_type, config.dimension,
                                 config.metric_type);
  for (const std::string& parameter : type->parameter_names) {
    if (parameter == "check_relative_distance" ||
        parameter == "fill_to_max_degree") {
      filtered.SetBoolean(parameter, config.GetBoolean(parameter, false));
    } else if (parameter == "nn_descent_convergence_threshold") {
      filtered.SetDouble(parameter, config.GetDouble(parameter, 0.0));
    } else {
      filtered.SetInteger(parameter, config.GetInteger(parameter, 0));
    }
  }
  std::unique_ptr<hypervec::Index> index = hypervec::CreateIndex(filtered);
  EXPECT_NE(dynamic_cast<IndexType*>(index.get()), nullptr);
}

}  // namespace

TEST(IndexConfig, StoresTypedParametersAndSortedNames) {
  hypervec::IndexConfig config("custom", 8, hypervec::kMetricInnerProduct);
  config.SetString("label", "alpha")
      .SetBoolean("enabled", true)
      .SetDouble("ratio", 0.25)
      .SetInteger("count", 7);

  EXPECT_EQ(config.index_type, "custom");
  EXPECT_EQ(config.dimension, 8);
  EXPECT_EQ(config.metric_type, hypervec::kMetricInnerProduct);
  EXPECT_EQ(config.GetString("label", ""), "alpha");
  EXPECT_TRUE(config.GetBoolean("enabled", false));
  EXPECT_DOUBLE_EQ(config.GetDouble("ratio", 0.0), 0.25);
  EXPECT_EQ(config.GetInteger("count", 0), 7);
  EXPECT_EQ(config.GetInteger("missing", 11), 11);
  EXPECT_EQ(config.ParameterNames(),
            (std::vector<std::string>{"count", "enabled", "label", "ratio"}));
}

TEST(IndexConfig, RejectsTypeMismatchAndEmptyNames) {
  hypervec::IndexConfig config("flat", 4);
  config.SetString("value", "not-an-integer");

  EXPECT_THROW(config.GetInteger("value", 0), hypervec::HypervecException);
  EXPECT_THROW(config.SetInteger("", 1), hypervec::HypervecException);
}

TEST(IndexRegistry, ListsAllBuiltInIndexesDeterministically) {
  const auto descriptors = hypervec::GetIndexRegistry().List();
  std::vector<std::string> names;
  for (const auto& descriptor : descriptors) {
    names.push_back(descriptor.name);
  }

  EXPECT_EQ(names, (std::vector<std::string>{"flat", "hnsw_flat", "hnsw_lvq",
                                             "hnsw_pq", "ivf_flat", "ivf_lvq",
                                             "ivf_pq", "lsh", "lvq", "nsg_flat",
                                             "nsw_flat", "pq"}));
}

TEST(IndexRegistry, CreatesEveryBuiltInIndex) {
  ExpectBuiltIn<hypervec::IndexFlat>("flat");
  ExpectBuiltIn<hypervec::IndexPQ>("pq");
  ExpectBuiltIn<hypervec::IndexLVQ>("lvq");
  ExpectBuiltIn<hypervec::IndexIVFFlat>("ivf_flat");
  ExpectBuiltIn<hypervec::IndexIVFPQ>("ivf_pq");
  ExpectBuiltIn<hypervec::IndexIVFLVQ>("ivf_lvq");
  ExpectBuiltIn<hypervec::IndexHNSWFlat>("hnsw_flat");
  ExpectBuiltIn<hypervec::IndexHNSWPQ>("hnsw_pq");
  ExpectBuiltIn<hypervec::IndexHNSWLVQ>("hnsw_lvq");
  ExpectBuiltIn<hypervec::IndexLSH>("lsh", hypervec::kMetricInnerProduct);
  ExpectBuiltIn<hypervec::IndexNSGFlat>("nsg_flat");
  ExpectBuiltIn<hypervec::IndexNSWFlat>("nsw_flat");
}

TEST(IndexRegistry, AppliesAlgorithmParameters) {
  hypervec::IndexConfig ivf_config("ivf_pq", 12);
  ivf_config.SetInteger("nlist", 7)
      .SetInteger("m_pq", 3)
      .SetInteger("nbits", 4);
  auto ivf_base = hypervec::CreateIndex(ivf_config);
  auto* ivf = dynamic_cast<hypervec::IndexIVFPQ*>(ivf_base.get());
  ASSERT_NE(ivf, nullptr);
  EXPECT_EQ(ivf->nlist, 7);
  EXPECT_EQ(ivf->pq.M, 3);
  EXPECT_EQ(ivf->pq.nbits, 4);

  hypervec::IndexConfig hnsw_config("hnsw_flat", 12);
  hnsw_config.SetInteger("m_hnsw", 11);
  auto hnsw_base = hypervec::CreateIndex(hnsw_config);
  auto* hnsw = dynamic_cast<hypervec::IndexHNSWFlat*>(hnsw_base.get());
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->hnsw.NbNeighbors(0), 22);
  EXPECT_EQ(hnsw->hnsw.NbNeighbors(1), 11);

  hypervec::IndexConfig nsw_config("nsw_flat", 12);
  nsw_config.SetInteger("max_degree", 7)
      .SetInteger("ef_construction", 15)
      .SetInteger("ef_search", 9)
      .SetBoolean("check_relative_distance", false)
      .SetBoolean("fill_to_max_degree", false);
  auto nsw_base = hypervec::CreateIndex(nsw_config);
  auto* nsw = dynamic_cast<hypervec::IndexNSWFlat*>(nsw_base.get());
  ASSERT_NE(nsw, nullptr);
  EXPECT_EQ(nsw->Options().max_degree, 7U);
  EXPECT_EQ(nsw->Options().ef_construction, 15U);
  EXPECT_EQ(nsw->Options().ef_search, 9U);
  EXPECT_FALSE(nsw->Options().check_relative_distance);
  EXPECT_FALSE(nsw->Options().fill_to_max_degree);

  hypervec::IndexConfig nsg_config("nsg_flat", 12);
  nsg_config.SetInteger("knn_degree", 9)
      .SetInteger("nn_descent_iterations", 6)
      .SetDouble("nn_descent_convergence_threshold", 0.02)
      .SetInteger("random_seed", 17)
      .SetInteger("max_degree", 7)
      .SetInteger("build_search_width", 11)
      .SetInteger("candidate_pool_size", 13)
      .SetInteger("ef_search", 15)
      .SetBoolean("check_relative_distance", false);
  auto nsg_base = hypervec::CreateIndex(nsg_config);
  auto* nsg = dynamic_cast<hypervec::IndexNSGFlat*>(nsg_base.get());
  ASSERT_NE(nsg, nullptr);
  EXPECT_EQ(nsg->Options().knn_degree, 9U);
  EXPECT_EQ(nsg->Options().nn_descent_iterations, 6U);
  EXPECT_DOUBLE_EQ(nsg->Options().nn_descent_convergence_threshold, 0.02);
  EXPECT_EQ(nsg->Options().random_seed, 17U);
  EXPECT_EQ(nsg->Options().max_degree, 7U);
  EXPECT_EQ(nsg->Options().build_search_width, 11U);
  EXPECT_EQ(nsg->Options().candidate_pool_size, 13U);
  EXPECT_EQ(nsg->Options().ef_search, 15U);
  EXPECT_FALSE(nsg->Options().check_relative_distance);

  hypervec::IndexConfig lsh_config("lsh", 12, hypervec::kMetricInnerProduct);
  lsh_config.SetInteger("table_count", 7)
      .SetInteger("bits_per_table", 9)
      .SetInteger("probe_count", 5)
      .SetInteger("candidate_limit", 17)
      .SetInteger("random_seed", 23);
  auto lsh_base = hypervec::CreateIndex(lsh_config);
  auto* lsh = dynamic_cast<hypervec::IndexLSH*>(lsh_base.get());
  ASSERT_NE(lsh, nullptr);
  EXPECT_EQ(lsh->Options().table_count, 7U);
  EXPECT_EQ(lsh->Options().bits_per_table, 9U);
  EXPECT_EQ(lsh->Options().probe_count, 5U);
  EXPECT_EQ(lsh->Options().candidate_limit, 17U);
  EXPECT_EQ(lsh->Options().random_seed, 23U);
}

TEST(IndexRegistry, AppliesMetricArgumentsToCompositeStorage) {
  hypervec::IndexConfig config("hnsw_flat", 4, hypervec::kMetricLp);
  config.metric_arg = 3.0F;
  config.SetInteger("m_hnsw", 4);

  auto base = hypervec::CreateIndex(config);
  auto* hnsw = dynamic_cast<hypervec::IndexHNSWFlat*>(base.get());
  ASSERT_NE(hnsw, nullptr);
  EXPECT_FLOAT_EQ(hnsw->metric_arg, 3.0F);
  ASSERT_NE(hnsw->storage, nullptr);
  EXPECT_FLOAT_EQ(hnsw->storage->metric_arg, 3.0F);
}

TEST(IndexRegistry, ResolvesAliasesCaseInsensitively) {
  hypervec::IndexConfig ivf_config("InDeXiVfFlAt", 4,
                                   hypervec::kMetricInnerProduct);
  ivf_config.SetInteger("nlist", 2);
  auto ivf = hypervec::CreateIndex(ivf_config);
  EXPECT_NE(dynamic_cast<hypervec::IndexIVFFlat*>(ivf.get()), nullptr);

  hypervec::IndexConfig hnsw_config("HNSW", 4);
  hnsw_config.SetInteger("m_hnsw", 4);
  auto hnsw = hypervec::CreateIndex(hnsw_config);
  EXPECT_NE(dynamic_cast<hypervec::IndexHNSWFlat*>(hnsw.get()), nullptr);

  hypervec::IndexConfig nsw_config("InDeXnSwFlAt", 4);
  nsw_config.SetInteger("max_degree", 4).SetInteger("ef_construction", 8);
  auto nsw = hypervec::CreateIndex(nsw_config);
  EXPECT_NE(dynamic_cast<hypervec::IndexNSWFlat*>(nsw.get()), nullptr);

  hypervec::IndexConfig nsg_config("NsG", 4);
  nsg_config.SetInteger("knn_degree", 4)
      .SetInteger("max_degree", 4)
      .SetInteger("candidate_pool_size", 8);
  auto nsg = hypervec::CreateIndex(nsg_config);
  EXPECT_NE(dynamic_cast<hypervec::IndexNSGFlat*>(nsg.get()), nullptr);

  hypervec::IndexConfig lsh_config("InDeXlSh", 4,
                                   hypervec::kMetricInnerProduct);
  auto lsh = hypervec::CreateIndex(lsh_config);
  EXPECT_NE(dynamic_cast<hypervec::IndexLSH*>(lsh.get()), nullptr);
}

TEST(IndexRegistry, CanComposeAnOwningIdMap) {
  hypervec::IndexConfig config("flat", 4);
  config.use_id_map = true;
  auto base = hypervec::CreateIndex(config);

  auto* mapped = dynamic_cast<hypervec::IndexIDMap*>(base.get());
  ASSERT_NE(mapped, nullptr);
  EXPECT_TRUE(mapped->own_fields);
  EXPECT_NE(dynamic_cast<hypervec::IndexFlat*>(mapped->index), nullptr);
  EXPECT_TRUE(mapped->GetCapabilities().supports_add_with_ids);
}

TEST(IndexRegistry, RejectsInvalidBuiltInConfigurations) {
  EXPECT_THROW(hypervec::CreateIndex(hypervec::IndexConfig("flat", 0)),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::CreateIndex(hypervec::IndexConfig("unknown", 4)),
               hypervec::HypervecException);

  hypervec::IndexConfig unknown_parameter("flat", 4);
  unknown_parameter.SetInteger("typo", 1);
  EXPECT_THROW(hypervec::CreateIndex(unknown_parameter),
               hypervec::HypervecException);

  hypervec::IndexConfig wrong_type("hnsw_flat", 4);
  wrong_type.SetString("m_hnsw", "four");
  EXPECT_THROW(hypervec::CreateIndex(wrong_type), hypervec::HypervecException);

  hypervec::IndexConfig bad_degree("hnsw_flat", 4);
  bad_degree.SetInteger("m_hnsw", 1);
  EXPECT_THROW(hypervec::CreateIndex(bad_degree), hypervec::HypervecException);

  hypervec::IndexConfig bad_nsw("nsw_flat", 4);
  bad_nsw.SetInteger("max_degree", 8).SetInteger("ef_construction", 4);
  EXPECT_THROW(hypervec::CreateIndex(bad_nsw), hypervec::HypervecException);

  hypervec::IndexConfig bad_nsg("nsg_flat", 4);
  bad_nsg.SetDouble("nn_descent_convergence_threshold", 2.0);
  EXPECT_THROW(hypervec::CreateIndex(bad_nsg), hypervec::HypervecException);

  EXPECT_THROW(hypervec::CreateIndex(hypervec::IndexConfig("lsh", 4)),
               hypervec::HypervecException);
  hypervec::IndexConfig bad_lsh("lsh", 4, hypervec::kMetricInnerProduct);
  bad_lsh.SetInteger("bits_per_table", 64);
  EXPECT_THROW(hypervec::CreateIndex(bad_lsh), hypervec::HypervecException);
  bad_lsh = hypervec::IndexConfig("lsh", 4, hypervec::kMetricInnerProduct);
  bad_lsh.SetInteger("candidate_limit", -1);
  EXPECT_THROW(hypervec::CreateIndex(bad_lsh), hypervec::HypervecException);

  hypervec::IndexConfig bad_pq("pq", 5);
  bad_pq.SetInteger("m_pq", 2);
  EXPECT_THROW(hypervec::CreateIndex(bad_pq), hypervec::HypervecException);

  EXPECT_THROW(hypervec::CreateIndex(
                   hypervec::IndexConfig("ivf_flat", 4, hypervec::kMetricL1)),
               hypervec::HypervecException);

  hypervec::IndexConfig bad_lp("flat", 4, hypervec::kMetricLp);
  EXPECT_THROW(hypervec::CreateIndex(bad_lp), hypervec::HypervecException);
}

TEST(IndexRegistry, SupportsIsolatedCustomRegistrations) {
  hypervec::IndexRegistry registry;
  registry.Register(
      {"custom_flat", {"custom-alias"}, {"marker"}},
      [](const hypervec::IndexConfig& config) {
        EXPECT_EQ(config.GetInteger("marker", 0), 42);
        return std::make_unique<hypervec::IndexFlatL2>(config.dimension);
      });

  EXPECT_TRUE(registry.Contains("CUSTOM-ALIAS"));
  hypervec::IndexConfig config("custom-alias", 6);
  config.SetInteger("marker", 42);
  auto index = registry.Create(config);
  EXPECT_NE(dynamic_cast<hypervec::IndexFlatL2*>(index.get()), nullptr);

  EXPECT_THROW(
      registry.Register({"other", {"CUSTOM-ALIAS"}, {}},
                        [](const hypervec::IndexConfig& custom_config) {
                          return std::make_unique<hypervec::IndexFlatL2>(
                              custom_config.dimension);
                        }),
      hypervec::HypervecException);
}
