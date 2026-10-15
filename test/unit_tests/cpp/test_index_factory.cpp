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
void ExpectBuiltIn(std::string name) {
  hypervec::IndexConfig config(std::move(name), 4);
  config.SetInteger("nlist", 2)
      .SetInteger("m_pq", 2)
      .SetInteger("nlocal", 2)
      .SetInteger("nbits", 2)
      .SetInteger("m_hnsw", 4);

  const auto allowed = hypervec::GetIndexRegistry().List();
  const auto type = std::find_if(allowed.begin(), allowed.end(),
                                 [&](const hypervec::IndexDescriptor& item) {
                                   return item.name == config.index_type;
                                 });
  ASSERT_NE(type, allowed.end());

  hypervec::IndexConfig filtered(config.index_type, config.dimension);
  for (const std::string& parameter : type->parameter_names) {
    filtered.SetInteger(parameter, config.GetInteger(parameter, 0));
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
                                             "ivf_pq", "lvq", "pq"}));
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
