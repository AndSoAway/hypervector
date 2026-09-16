/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/engine.h>
#include <factory/index_registry.h>
#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hypervec {

TEST(TestFactoryTools, ParsesStrictTypedConfig) {
  const IndexConfig config = ParseIndexConfig(R"({
    "type": "HNSW",
    "dimension": 12,
    "metric": "l2",
    "backend": "CPU",
    "storage": {"type": "PQ", "pq_m": 3, "nbits": 4},
    "build": {"graph_degree": 24, "ef_construction": 80},
    "defaults": {"ef_search": 32},
    "id_mode": "external"
  })");

  EXPECT_EQ(config.type, "HNSW");
  EXPECT_EQ(config.dimension, 12);
  EXPECT_EQ(config.metric, kMetricL2);
  EXPECT_EQ(config.backend, "CPU");
  EXPECT_EQ(config.storage.type, "PQ");
  EXPECT_EQ(config.storage.pq_m, 3);
  EXPECT_EQ(config.storage.nbits, 4);
  EXPECT_EQ(config.build.graph_degree, 24);
  EXPECT_EQ(config.build.ef_construction, 80);
  EXPECT_EQ(config.defaults.ef_search, 32);
  EXPECT_EQ(config.id_mode, IdMode::External);
}

TEST(TestFactoryTools, RejectsMalformedAndAmbiguousJson) {
  EXPECT_THROW(ParseIndexConfig(""), HypervecException);
  EXPECT_THROW(ParseIndexConfig("[]"), HypervecException);
  EXPECT_THROW(ParseIndexConfig(R"({"type":"flat"})"), HypervecException);
  EXPECT_THROW(ParseIndexConfig(R"({"dimension":4})"), HypervecException);
  EXPECT_THROW(ParseIndexConfig(R"({"type":"flat","dimension":4,"extra":1})"),
               HypervecException);
  EXPECT_THROW(ParseIndexConfig(R"({"type":"flat","type":"pq","dimension":4})"),
               HypervecException);
  EXPECT_THROW(ParseIndexConfig(R"({"type":"flat","dimension":4} trailing)"),
               HypervecException);
  EXPECT_THROW(ParseIndexConfig(R"({"type":"flat","dimension":4,"metric":1})"),
               HypervecException);
  EXPECT_THROW(
      ParseIndexConfig(R"({"type":"flat","dimension":4,"metric":"cosine"})"),
      HypervecException);
}

TEST(TestFactoryTools, RegistersBuiltinsAndResolvesAliases) {
  Engine engine;
  const std::vector<IndexDescriptor> descriptors =
      IndexRegistry::Global().List();
  const std::array<std::string, 9> expected = {
      "flat",    "hnsw_flat", "hnsw_lvq", "hnsw_pq", "ivf_flat",
      "ivf_lvq", "ivf_pq",    "lvq",      "pq"};
  for (const std::string& name : expected) {
    EXPECT_NE(std::find_if(descriptors.begin(), descriptors.end(),
                           [&name](const IndexDescriptor& descriptor) {
                             return descriptor.canonical_name == name;
                           }),
              descriptors.end());
  }

  std::unique_ptr<Index> alias =
      engine.CreateIndex(R"({"type":"INDEXHNSWFLAT","dimension":8})");
  EXPECT_EQ(alias->TypeName(), "hnsw_flat");
  std::unique_ptr<Index> hyphen =
      engine.CreateIndex(R"({"type":"hnsw-flat","dimension":8})");
  EXPECT_EQ(hyphen->TypeName(), "hnsw_flat");
}

TEST(TestFactoryTools, RegistryRejectsDuplicateNamesTransactionally) {
  IndexRegistry registry;
  IndexDescriptor descriptor{
      "custom",
      {"custom_alias"},
      ToCapabilities(IndexCapability::KnnSearch),
      {kMetricL2},
      {"cpu"},
      [](IndexConfig&) {},
      [](const IndexConfig& config, const ResourceContext&) {
        return std::make_unique<IndexFlatL2>(config.dimension);
      }};
  registry.Register(descriptor);
  EXPECT_THROW(registry.Register(descriptor), HypervecException);
  ASSERT_TRUE(registry.Find("CUSTOM-ALIAS").has_value());
  EXPECT_EQ(registry.Find("custom_alias")->canonical_name, "custom");
  EXPECT_EQ(registry.List().size(), 1);
}

TEST(TestFactoryTools, CreatesEveryRegisteredBuiltinVariant) {
  Engine engine;
  const std::vector<std::pair<std::string, std::string>> cases = {
      {R"({"type":"flat","dimension":8})", "flat"},
      {R"({"type":"pq","dimension":8,"storage":{"pq_m":2,"nbits":2}})", "pq"},
      {R"({"type":"lvq","dimension":8,"storage":{"nlocal":2,"nbits":2}})",
       "lvq"},
      {R"({"type":"ivf","dimension":8,"storage":{"type":"flat"},"build":{"nlist":4},"defaults":{"nprobe":2}})",
       "ivf_flat"},
      {R"({"type":"ivf","dimension":8,"storage":{"type":"pq","pq_m":2,"nbits":2},"build":{"nlist":4},"defaults":{"nprobe":2}})",
       "ivf_pq"},
      {R"({"type":"ivf","dimension":8,"storage":{"type":"lvq","nlocal":2,"nbits":2},"build":{"nlist":4},"defaults":{"nprobe":2}})",
       "ivf_lvq"},
      {R"({"type":"hnsw","dimension":8,"storage":{"type":"flat"},"build":{"graph_degree":4}})",
       "hnsw_flat"},
      {R"({"type":"hnsw","dimension":8,"storage":{"type":"pq","pq_m":2,"nbits":2},"build":{"graph_degree":4}})",
       "hnsw_pq"},
      {R"({"type":"hnsw","dimension":8,"storage":{"type":"lvq","nlocal":2,"nbits":2},"build":{"graph_degree":4}})",
       "hnsw_lvq"},
  };

  for (const auto& [json, type] : cases) {
    SCOPED_TRACE(type);
    std::unique_ptr<Index> index = engine.CreateIndex(json);
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->TypeName(), type);
    EXPECT_EQ(index->d, 8);
  }
}

TEST(TestFactoryTools, AppliesAlgorithmParametersToConcreteIndex) {
  Engine engine;
  std::unique_ptr<Index> hnsw_index = engine.CreateIndex(R"({
    "type":"hnsw", "dimension":8,
    "storage":{"type":"pq","pq_m":2,"nbits":3},
    "build":{"graph_degree":7,"ef_construction":91},
    "defaults":{"ef_search":37}
  })");
  const auto* hnsw = dynamic_cast<const IndexHNSWPQ*>(hnsw_index.get());
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->hnsw.NbNeighbors(0), 14);
  EXPECT_EQ(hnsw->hnsw.ef_construction, 91);
  EXPECT_EQ(hnsw->hnsw.ef_search, 37);
  const auto* hnsw_pq = dynamic_cast<const IndexPQ*>(hnsw->storage);
  ASSERT_NE(hnsw_pq, nullptr);
  EXPECT_EQ(hnsw_pq->pq.M, 2);
  EXPECT_EQ(hnsw_pq->pq.nbits, 3);

  std::unique_ptr<Index> ivf_index = engine.CreateIndex(R"({
    "type":"ivf_pq", "dimension":8,
    "storage":{"pq_m":4,"nbits":2},
    "build":{"nlist":5,"by_residual":false,"use_precomputed_table":1},
    "defaults":{"nprobe":3}
  })");
  const auto* ivf = dynamic_cast<const IndexIVFPQ*>(ivf_index.get());
  ASSERT_NE(ivf, nullptr);
  EXPECT_EQ(ivf->nlist, 5);
  EXPECT_EQ(ivf->nprobe, 3);
  EXPECT_EQ(ivf->pq.M, 4);
  EXPECT_EQ(ivf->pq.nbits, 2);
  EXPECT_FALSE(ivf->by_residual);
  EXPECT_EQ(ivf->use_precomputed_table, 1);
}

TEST(TestFactoryTools, NormalizesDefaultsAfterSelectingConcreteType) {
  Engine engine;
  IndexConfig request;
  request.type = "IVF";
  request.dimension = 16;
  request.storage.type = "PQ";

  const IndexConfig normalized = engine.NormalizeConfig(request);
  EXPECT_EQ(normalized.type, "ivf_pq");
  EXPECT_EQ(normalized.storage.type, "pq");
  EXPECT_EQ(normalized.storage.pq_m, 8);
  EXPECT_EQ(normalized.storage.nbits, 8);
  EXPECT_EQ(normalized.build.nlist, 1024);
  EXPECT_EQ(normalized.defaults.nprobe, 1);
  EXPECT_EQ(normalized.build.by_residual, true);
  EXPECT_EQ(normalized.build.use_precomputed_table, 0);
}

TEST(TestFactoryTools, RejectsUnsupportedOrMisplacedParameters) {
  Engine engine;
  EXPECT_THROW(engine.CreateIndex(R"({"type":"unknown","dimension":8})"),
               HypervecException);
  EXPECT_THROW(engine.CreateIndex(R"({"type":"flat","dimension":0})"),
               HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(R"({"type":"flat","dimension":8,"backend":"cuda"})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"hnsw","dimension":8,"storage":{"type":"binary"}})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(R"({"type":"pq","dimension":7,"storage":{"pq_m":2}})"),
      HypervecException);
  EXPECT_THROW(engine.CreateIndex(
                   R"({"type":"pq","dimension":8,"storage":{"nbits":17}})"),
               HypervecException);
  EXPECT_THROW(engine.CreateIndex(
                   R"({"type":"flat","dimension":8,"storage":{"nbits":4}})"),
               HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"flat","dimension":8,"build":{"graph_degree":8}})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"hnsw_flat","dimension":8,"defaults":{"nprobe":1}})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"ivf_flat","dimension":8,"build":{"nlist":2},"defaults":{"nprobe":3}})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"ivf_pq","dimension":8,"metric":"inner_product"})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"ivf_pq","dimension":8,"build":{"use_precomputed_table":2}})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"ivf_flat","dimension":8,"build":{"nlist":9223372036854775807}})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(
          R"({"type":"hnsw_flat","dimension":8,"storage":{"type":"pq"}})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(R"({"type":"flat","dimension":8,"metric_arg":3})"),
      HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(R"({"type":"flat","dimension":8,"metric":"lp"})"),
      HypervecException);
}

TEST(TestFactoryTools, CreatesLpMetricAndExternalIdWrapper) {
  Engine engine;
  std::unique_ptr<Index> lp = engine.CreateIndex(
      R"({"type":"flat","dimension":8,"metric":"lp","metric_arg":3})");
  EXPECT_EQ(lp->metric_type, kMetricLp);
  EXPECT_FLOAT_EQ(lp->metric_arg, 3.0F);

  std::unique_ptr<Index> external = engine.CreateIndex(
      R"({"type":"hnsw","dimension":8,"id_mode":"external"})");
  auto* id_map = dynamic_cast<IndexIDMap*>(external.get());
  ASSERT_NE(id_map, nullptr);
  EXPECT_TRUE(id_map->OwnsIndex());
  EXPECT_NE(id_map->get_index<IndexHNSWFlat>(), nullptr);
  EXPECT_TRUE(
      HasCapability(id_map->Capabilities(), IndexCapability::ExternalIds));
}

TEST(TestFactoryTools, RegistryCapabilitiesAreGuaranteedByInstances) {
  Engine engine;
  for (const IndexDescriptor& descriptor : IndexRegistry::Global().List()) {
    SCOPED_TRACE(descriptor.canonical_name);
    IndexConfig config;
    config.type = descriptor.canonical_name;
    config.dimension = 8;
    config.storage.nbits = 2;
    if (descriptor.canonical_name.find("pq") != std::string::npos ||
        descriptor.canonical_name == "pq") {
      config.storage.pq_m = 2;
    } else if (descriptor.canonical_name.find("lvq") != std::string::npos ||
               descriptor.canonical_name == "lvq") {
      config.storage.nlocal = 2;
    } else {
      config.storage.nbits.reset();
    }
    if (descriptor.canonical_name.starts_with("ivf_")) {
      config.build.nlist = 4;
      config.defaults.nprobe = 2;
    }
    if (descriptor.canonical_name.starts_with("hnsw_")) {
      config.build.graph_degree = 4;
    }

    const std::unique_ptr<Index> index = engine.CreateIndex(config);
    EXPECT_EQ(descriptor.guaranteed_capabilities & ~index->Capabilities(), 0);
  }
}

TEST(TestFactoryTools, FactoryFlatMatchesDirectConstruction) {
  Engine engine;
  std::unique_ptr<Index> factory =
      engine.CreateIndex(R"({"type":"flat","dimension":2,"metric":"l2"})");
  IndexFlatL2 direct(2);
  const std::array<float, 8> vectors = {0.0F, 0.0F, 1.0F, 0.0F,
                                        0.0F, 2.0F, 3.0F, 3.0F};
  const DenseDatasetView data{4, 2, vectors.data(), nullptr};
  factory->Build(data);
  direct.Build(data);

  const std::array<float, 2> query = {0.5F, 0.25F};
  std::array<float, 4> factory_distances{};
  std::array<float, 4> direct_distances{};
  std::array<idx_t, 4> factory_labels{};
  std::array<idx_t, 4> direct_labels{};
  factory->Search(1, query.data(), 4, factory_distances.data(),
                  factory_labels.data());
  direct.Search(1, query.data(), 4, direct_distances.data(),
                direct_labels.data());

  EXPECT_EQ(factory_labels, direct_labels);
  EXPECT_EQ(factory_distances, direct_distances);
}

}  // namespace hypervec
