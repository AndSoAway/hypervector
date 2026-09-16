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
#include <index/hnsw/hnsw.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf.h>
#include <index/ivf/index_ivf_flat.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <utils/log/exception.h>

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
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

TEST(TestFactoryTools, AcceptsUtf8Bom) {
  const IndexConfig config =
      ParseIndexConfig("\xEF\xBB\xBF{\"type\":\"flat\",\"dimension\":4}");
  EXPECT_EQ(config.type, "flat");
  EXPECT_EQ(config.dimension, 4);
}

TEST(TestFactoryTools, ParsesStrictRuntimeSearchConfig) {
  const SearchConfig config =
      ParseSearchConfig(R"({"nprobe":4,"ef_search":32})");
  EXPECT_EQ(config.nprobe, 4);
  EXPECT_EQ(config.ef_search, 32);

  const SearchConfig empty = ParseSearchConfig("{}");
  EXPECT_FALSE(empty.nprobe.has_value());
  EXPECT_FALSE(empty.ef_search.has_value());

  EXPECT_THROW(ParseSearchConfig("[]"), HypervecException);
  EXPECT_THROW(ParseSearchConfig(R"({"nprobe":1,"extra":2})"),
               HypervecException);
  EXPECT_THROW(ParseSearchConfig(R"({"nprobe":1,"nprobe":2})"),
               HypervecException);
  EXPECT_THROW(ParseSearchConfig(R"({"nprobe":1.5})"), HypervecException);
  EXPECT_THROW(ParseSearchConfig(R"({"ef_search":2147483648})"),
               HypervecException);
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
  EXPECT_THROW(
      ParseIndexConfig(
          R"({"type":"flat","dimension":4,"metric":"lp","metric_arg":1e999})"),
      HypervecException);
}

TEST(TestFactoryTools, RegistersBuiltinsAndResolvesAliases) {
  Engine engine;
  const std::vector<IndexDescriptorPtr> descriptors =
      IndexRegistry::Global().List();
  const std::array<std::string, 9> expected = {
      "flat",    "hnsw_flat", "hnsw_lvq", "hnsw_pq", "ivf_flat",
      "ivf_lvq", "ivf_pq",    "lvq",      "pq"};
  for (const std::string& name : expected) {
    EXPECT_NE(std::find_if(descriptors.begin(), descriptors.end(),
                           [&name](const IndexDescriptorPtr& descriptor) {
                             return descriptor->canonical_name == name;
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
      },
      [](const IndexConfig&, SearchConfig&) {},
      [](const IndexConfig&, const SearchConfig&) {
        return std::make_unique<SearchParameters>();
      }};
  registry.Register(descriptor);
  EXPECT_THROW(registry.Register(descriptor), HypervecException);
  const IndexDescriptorPtr by_alias = registry.Find("CUSTOM-ALIAS");
  const IndexDescriptorPtr by_canonical = registry.Find("custom");
  ASSERT_NE(by_alias, nullptr);
  ASSERT_NE(by_canonical, nullptr);
  EXPECT_EQ(by_alias->canonical_name, "custom");
  EXPECT_EQ(by_alias.get(), by_canonical.get());
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

TEST(TestFactoryTools, CreatesRuntimeSearchParametersWithoutMutatingIndex) {
  Engine engine;
  const IndexConfig ivf_config = ParseIndexConfig(R"({
    "type":"ivf", "dimension":8,
    "build":{"nlist":5}, "defaults":{"nprobe":2}
  })");
  const SearchConfig ivf_defaults = engine.NormalizeSearchConfig(ivf_config);
  EXPECT_EQ(ivf_defaults.nprobe, 2);
  EXPECT_FALSE(ivf_defaults.ef_search.has_value());

  SearchConfig ivf_override;
  ivf_override.nprobe = 4;
  std::unique_ptr<SearchParameters> ivf_parameters =
      engine.CreateSearchParameters(ivf_config, ivf_override);
  const auto* concrete_ivf =
      dynamic_cast<const IVFSearchParameters*>(ivf_parameters.get());
  ASSERT_NE(concrete_ivf, nullptr);
  EXPECT_EQ(concrete_ivf->nprobe, 4);

  std::unique_ptr<Index> ivf_index = engine.CreateIndex(ivf_config);
  const auto* ivf = dynamic_cast<const IndexIVFFlat*>(ivf_index.get());
  ASSERT_NE(ivf, nullptr);
  EXPECT_EQ(ivf->nprobe, 2);

  const IndexConfig hnsw_config = ParseIndexConfig(R"({
    "type":"hnsw", "dimension":8, "defaults":{"ef_search":24}
  })");
  const SearchConfig hnsw_defaults = engine.NormalizeSearchConfig(hnsw_config);
  EXPECT_EQ(hnsw_defaults.ef_search, 24);
  EXPECT_FALSE(hnsw_defaults.nprobe.has_value());

  SearchConfig hnsw_override;
  hnsw_override.ef_search = 48;
  std::unique_ptr<SearchParameters> hnsw_parameters =
      engine.CreateSearchParameters(hnsw_config, hnsw_override);
  const auto* concrete_hnsw =
      dynamic_cast<const SearchParametersHNSW*>(hnsw_parameters.get());
  ASSERT_NE(concrete_hnsw, nullptr);
  EXPECT_EQ(concrete_hnsw->ef_search, 48);

  IndexConfig flat_config;
  flat_config.type = "flat";
  flat_config.dimension = 8;
  EXPECT_NE(engine.CreateSearchParameters(flat_config), nullptr);
}

TEST(TestFactoryTools, RuntimeSearchParametersUseTheCommonIndexApi) {
  Engine engine;
  const std::array<float, 8> vectors = {0.0F, 0.0F, 1.0F, 0.0F,
                                        0.0F, 2.0F, 3.0F, 3.0F};
  const std::array<float, 2> query = {1.0F, 0.0F};

  const IndexConfig ivf_config = ParseIndexConfig(R"({
    "type":"ivf_flat", "dimension":2, "build":{"nlist":2}
  })");
  std::unique_ptr<Index> ivf = engine.CreateIndex(ivf_config);
  ivf->Build({4, 2, vectors.data(), nullptr});
  SearchConfig ivf_search;
  ivf_search.nprobe = 2;
  std::unique_ptr<SearchParameters> ivf_parameters =
      engine.CreateSearchParameters(ivf_config, ivf_search);
  std::array<float, 1> distances{};
  std::array<idx_t, 1> labels{};
  ivf->Search(1, query.data(), 1, distances.data(), labels.data(),
              ivf_parameters.get());
  EXPECT_EQ(labels[0], 1);
  EXPECT_FLOAT_EQ(distances[0], 0.0F);

  const IndexConfig hnsw_config = ParseIndexConfig(R"({
    "type":"hnsw_flat", "dimension":2,
    "build":{"graph_degree":2}, "id_mode":"external"
  })");
  std::unique_ptr<Index> hnsw = engine.CreateIndex(hnsw_config);
  const std::array<idx_t, 4> ids = {10, 20, 30, 40};
  hnsw->Build({4, 2, vectors.data(), ids.data()});
  SearchConfig hnsw_search;
  hnsw_search.ef_search = 8;
  std::unique_ptr<SearchParameters> hnsw_parameters =
      engine.CreateSearchParameters(hnsw_config, hnsw_search);
  hnsw->Search(1, query.data(), 1, distances.data(), labels.data(),
               hnsw_parameters.get());
  EXPECT_EQ(labels[0], 20);
  EXPECT_FLOAT_EQ(distances[0], 0.0F);
}

TEST(TestFactoryTools, RejectsInvalidRuntimeSearchParameters) {
  Engine engine;
  IndexConfig flat;
  flat.type = "flat";
  flat.dimension = 8;

  SearchConfig nprobe;
  nprobe.nprobe = 1;
  EXPECT_THROW(engine.CreateSearchParameters(flat, nprobe), HypervecException);

  IndexConfig ivf;
  ivf.type = "ivf_flat";
  ivf.dimension = 8;
  ivf.build.nlist = 4;
  SearchConfig ef_search;
  ef_search.ef_search = 10;
  EXPECT_THROW(engine.CreateSearchParameters(ivf, ef_search),
               HypervecException);
  nprobe.nprobe = 0;
  EXPECT_THROW(engine.CreateSearchParameters(ivf, nprobe), HypervecException);
  nprobe.nprobe = 5;
  EXPECT_THROW(engine.CreateSearchParameters(ivf, nprobe), HypervecException);

  IndexConfig hnsw;
  hnsw.type = "hnsw_flat";
  hnsw.dimension = 8;
  EXPECT_THROW(engine.CreateSearchParameters(hnsw, nprobe), HypervecException);
  ef_search.ef_search = 0;
  EXPECT_THROW(engine.CreateSearchParameters(hnsw, ef_search),
               HypervecException);
}

TEST(TestFactoryTools, SerializesCanonicalIndexAndSearchConfig) {
  Engine engine;
  const IndexConfig normalized_hnsw =
      engine.NormalizeConfig(ParseIndexConfig(R"({
    "type":"hnsw", "dimension":8,
    "storage":{"type":"pq","pq_m":2,"nbits":3},
    "build":{"graph_degree":7,"ef_construction":91},
    "defaults":{"ef_search":37}
  })"));
  const std::string hnsw_json = SerializeIndexConfig(normalized_hnsw);
  EXPECT_EQ(
      hnsw_json,
      R"({"type":"hnsw_pq","dimension":8,"metric":"l2","backend":"cpu","storage":{"type":"pq","pq_m":2,"nbits":3},"build":{"graph_degree":7,"ef_construction":91},"defaults":{"ef_search":37},"id_mode":"sequential"})");
  EXPECT_EQ(
      SerializeIndexConfig(engine.NormalizeConfig(ParseIndexConfig(hnsw_json))),
      hnsw_json);

  const IndexConfig normalized_ivf =
      engine.NormalizeConfig(ParseIndexConfig(R"({
    "type":"ivf", "dimension":16, "id_mode":"external",
    "storage":{"type":"pq","pq_m":4,"nbits":6},
    "build":{"nlist":32,"by_residual":false,"use_precomputed_table":1},
    "defaults":{"nprobe":7}
  })"));
  const std::string ivf_json = SerializeIndexConfig(normalized_ivf);
  EXPECT_EQ(
      ivf_json,
      R"({"type":"ivf_pq","dimension":16,"metric":"l2","backend":"cpu","storage":{"type":"pq","pq_m":4,"nbits":6},"build":{"nlist":32,"by_residual":false,"use_precomputed_table":1},"defaults":{"nprobe":7},"id_mode":"external"})");
  EXPECT_EQ(
      SerializeIndexConfig(engine.NormalizeConfig(ParseIndexConfig(ivf_json))),
      ivf_json);

  const IndexConfig normalized_flat = engine.NormalizeConfig(ParseIndexConfig(
      R"({"type":"flat","dimension":3,"metric":"lp","metric_arg":1.25})"));
  const std::string flat_json = SerializeIndexConfig(normalized_flat);
  EXPECT_EQ(
      flat_json,
      R"({"type":"flat","dimension":3,"metric":"lp","metric_arg":1.25,"backend":"cpu","storage":{"type":"flat"},"build":{},"defaults":{},"id_mode":"sequential"})");
  EXPECT_EQ(
      SerializeIndexConfig(engine.NormalizeConfig(ParseIndexConfig(flat_json))),
      flat_json);

  const IndexConfig unresolved =
      ParseIndexConfig(R"({"type":"hnsw","dimension":3})");
  std::unique_ptr<Index> roundtrip =
      engine.CreateIndex(ParseIndexConfig(SerializeIndexConfig(unresolved)));
  ASSERT_NE(roundtrip, nullptr);
  EXPECT_EQ(roundtrip->TypeName(), "hnsw_flat");

  SearchConfig search;
  search.ef_search = 64;
  EXPECT_EQ(SerializeSearchConfig(search), R"({"ef_search":64})");
  EXPECT_EQ(SerializeSearchConfig(ParseSearchConfig("{}")), "{}");
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
  EXPECT_THROW(engine.CreateIndex(R"({"type":"flat!","dimension":8})"),
               HypervecException);
  EXPECT_THROW(
      engine.CreateIndex(R"({"type":"flat","dimension":8,"backend":"c pu"})"),
      HypervecException);
}

TEST(TestFactoryTools, ValidatesTypedNumericBoundaries) {
  Engine engine;
  IndexConfig config;
  config.type = "flat";
  config.dimension = static_cast<idx_t>(INT_MAX);
  EXPECT_EQ(engine.NormalizeConfig(config).dimension,
            static_cast<idx_t>(INT_MAX));

  config.dimension = static_cast<idx_t>(INT_MAX) + 1;
  EXPECT_THROW(engine.NormalizeConfig(config), HypervecException);

  config.dimension = 8;
  config.metric = kMetricLp;
  config.metric_arg = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(engine.NormalizeConfig(config), HypervecException);
  config.metric_arg = std::numeric_limits<float>::infinity();
  EXPECT_THROW(engine.NormalizeConfig(config), HypervecException);
}

TEST(TestFactoryTools, CreatesLpMetricAndExternalIdWrapper) {
  Engine engine;
  std::unique_ptr<Index> sequential =
      engine.CreateIndex(R"({"type":"flat","dimension":8})");
  EXPECT_EQ(sequential->TypeName(), "flat");
  EXPECT_EQ(dynamic_cast<IndexIDMap*>(sequential.get()), nullptr);

  std::unique_ptr<Index> lp = engine.CreateIndex(
      R"({"type":"flat","dimension":8,"metric":"lp","metric_arg":3})");
  EXPECT_EQ(lp->metric_type, kMetricLp);
  EXPECT_FLOAT_EQ(lp->metric_arg, 3.0F);

  std::unique_ptr<Index> external = engine.CreateIndex(
      R"({"type":"hnsw","dimension":8,"id_mode":"external"})");
  EXPECT_EQ(external->TypeName(), "idmap");
  auto* id_map = dynamic_cast<IndexIDMap*>(external.get());
  ASSERT_NE(id_map, nullptr);
  EXPECT_TRUE(id_map->OwnsIndex());
  EXPECT_NE(id_map->get_index<IndexHNSWFlat>(), nullptr);
  EXPECT_TRUE(
      HasCapability(id_map->Capabilities(), IndexCapability::ExternalIds));
}

TEST(TestFactoryTools, RegistryCapabilitiesAreGuaranteedByInstances) {
  Engine engine;
  for (const IndexDescriptorPtr& descriptor : IndexRegistry::Global().List()) {
    SCOPED_TRACE(descriptor->canonical_name);
    IndexConfig config;
    config.type = descriptor->canonical_name;
    config.dimension = 8;
    config.storage.nbits = 2;
    if (descriptor->canonical_name.find("pq") != std::string::npos ||
        descriptor->canonical_name == "pq") {
      config.storage.pq_m = 2;
    } else if (descriptor->canonical_name.find("lvq") != std::string::npos ||
               descriptor->canonical_name == "lvq") {
      config.storage.nlocal = 2;
    } else {
      config.storage.nbits.reset();
    }
    if (descriptor->canonical_name.starts_with("ivf_")) {
      config.build.nlist = 4;
      config.defaults.nprobe = 2;
    }
    if (descriptor->canonical_name.starts_with("hnsw_")) {
      config.build.graph_degree = 4;
    }

    const std::unique_ptr<Index> index = engine.CreateIndex(config);
    EXPECT_EQ(descriptor->guaranteed_capabilities & ~index->Capabilities(), 0);
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
