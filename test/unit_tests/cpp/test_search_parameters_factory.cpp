/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/diskann/index_diskann.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/lsh/index_lsh.h>
#include <index/nsg/index_nsg.h>
#include <index/nsw/index_nsw.h>
#include <index/pretransform/index_pre_transform.h>
#include <index/search_parameters_factory.h>
#include <index/vamana/index_vamana.h>
#include <transform/vector_transform.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

TEST(SearchConfig, StoresTypedValuesAndSortedNames) {
  hypervec::SearchConfig config;
  config.SetBoolean("enabled", true).SetInteger("width", 17);

  EXPECT_TRUE(config.HasParameter("enabled"));
  EXPECT_TRUE(config.GetBoolean("enabled", false));
  EXPECT_EQ(config.GetInteger("width", 0), 17);
  EXPECT_EQ(config.GetInteger("missing", 23), 23);
  EXPECT_EQ(config.ParameterNames(),
            (std::vector<std::string>{"enabled", "width"}));

  EXPECT_THROW(config.GetInteger("enabled", 0), hypervec::HypervecException);
  EXPECT_THROW(config.SetInteger("", 1), hypervec::HypervecException);
}

TEST(SearchParametersFactory, CreatesGraphParameterSubtypes) {
  hypervec::DiskAnnIndexOptions diskann_options;
  diskann_options.search_width = 71;
  diskann_options.check_relative_distance = false;
  hypervec::IndexDiskANNFlat diskann(4, hypervec::kMetricL2, diskann_options);
  auto diskann_base = hypervec::CreateSearchParameters(diskann);
  auto* diskann_params =
      dynamic_cast<hypervec::SearchParametersDiskANN*>(diskann_base.get());
  ASSERT_NE(diskann_params, nullptr);
  EXPECT_EQ(diskann_params->search_width, 71U);
  EXPECT_FALSE(diskann_params->check_relative_distance);

  hypervec::IndexHNSWFlat hnsw(4, 8);
  hnsw.hnsw.ef_search = 29;
  hnsw.hnsw.check_relative_distance = false;
  hnsw.hnsw.search_bounded_queue = false;
  hypervec::SearchConfig hnsw_config;
  hnsw_config.SetInteger("ef", 43).SetBoolean("bounded_queue", true);
  auto hnsw_base = hypervec::CreateSearchParameters(hnsw, hnsw_config);
  auto* hnsw_params =
      dynamic_cast<hypervec::SearchParametersHNSW*>(hnsw_base.get());
  ASSERT_NE(hnsw_params, nullptr);
  EXPECT_EQ(hnsw_params->ef_search, 43);
  EXPECT_FALSE(hnsw_params->check_relative_distance);
  EXPECT_TRUE(hnsw_params->bounded_queue);

  hypervec::NSWIndexOptions nsw_options;
  nsw_options.ef_search = 31;
  nsw_options.check_relative_distance = false;
  hypervec::IndexNSWFlat nsw(4, hypervec::kMetricL2, nsw_options);
  hypervec::SearchConfig nsw_config;
  nsw_config.SetInteger("ef_search", 47)
      .SetBoolean("check_relative_distance", true);
  auto nsw_base = hypervec::CreateSearchParameters(nsw, nsw_config);
  auto* nsw_params =
      dynamic_cast<hypervec::SearchParametersNSW*>(nsw_base.get());
  ASSERT_NE(nsw_params, nullptr);
  EXPECT_EQ(nsw_params->ef_search, 47U);
  EXPECT_TRUE(nsw_params->check_relative_distance);

  hypervec::NSGIndexOptions nsg_options;
  nsg_options.ef_search = 37;
  hypervec::IndexNSGFlat nsg(4, hypervec::kMetricL2, nsg_options);
  auto nsg_base = hypervec::CreateSearchParameters(nsg);
  auto* nsg_params =
      dynamic_cast<hypervec::SearchParametersNSG*>(nsg_base.get());
  ASSERT_NE(nsg_params, nullptr);
  EXPECT_EQ(nsg_params->ef_search, 37U);

  hypervec::VamanaIndexOptions vamana_options;
  vamana_options.search_width = 53;
  hypervec::IndexVamanaFlat vamana(4, hypervec::kMetricL2, vamana_options);
  auto vamana_base = hypervec::CreateSearchParameters(vamana);
  auto* vamana_params =
      dynamic_cast<hypervec::SearchParametersVamana*>(vamana_base.get());
  ASSERT_NE(vamana_params, nullptr);
  EXPECT_EQ(vamana_params->search_width, 53U);
}

TEST(SearchParametersFactory, CreatesIvfAndLshParameters) {
  hypervec::IndexIVFFlat ivf(4, 8);
  ivf.nprobe = 3;
  hypervec::SearchConfig ivf_config;
  ivf_config.SetInteger("nprobe", 5);
  auto ivf_base = hypervec::CreateSearchParameters(ivf, ivf_config);
  auto* ivf_params =
      dynamic_cast<hypervec::IVFSearchParameters*>(ivf_base.get());
  ASSERT_NE(ivf_params, nullptr);
  EXPECT_EQ(ivf_params->nprobe, 5);

  hypervec::IndexLSH lsh(4);
  hypervec::SearchConfig lsh_config;
  lsh_config.SetInteger("probe_count", 7).SetInteger("candidate_limit", 19);
  auto lsh_base = hypervec::CreateSearchParameters(lsh, lsh_config);
  auto* lsh_params =
      dynamic_cast<hypervec::SearchParametersLSH*>(lsh_base.get());
  ASSERT_NE(lsh_params, nullptr);
  EXPECT_EQ(lsh_params->probe_count, 7U);
  EXPECT_EQ(lsh_params->candidate_limit, 19U);
}

TEST(SearchParametersFactory, UnwrapsIdMapAndCarriesSelector) {
  hypervec::NSWIndexOptions options;
  options.ef_search = 23;
  auto nsw =
      std::make_unique<hypervec::IndexNSWFlat>(4, hypervec::kMetricL2, options);
  auto transform = std::make_unique<hypervec::LinearTransform>(4, 4);
  transform->SetIdentity();
  hypervec::IndexPreTransform transformed(std::move(transform), std::move(nsw));
  hypervec::IndexIDMap mapped(&transformed);
  hypervec::IDSelectorRange selector(100, 200);
  hypervec::SearchConfig config;
  config.selector = &selector;

  const auto descriptor = hypervec::DescribeSearchParameters(mapped);
  EXPECT_EQ(descriptor.name, "nsw");
  EXPECT_EQ(descriptor.parameter_names,
            (std::vector<std::string>{"check_relative_distance", "ef_search"}));

  auto base = hypervec::CreateSearchParameters(mapped, config);
  auto* params = dynamic_cast<hypervec::SearchParametersNSW*>(base.get());
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->ef_search, 23U);
  EXPECT_EQ(params->sel, &selector);
}

TEST(SearchParametersFactory, CreatesGenericParametersForFlatIndex) {
  hypervec::IndexFlatL2 flat(4);
  hypervec::IDSelectorRange selector(0, 10);
  hypervec::SearchConfig config;
  config.selector = &selector;

  const auto descriptor = hypervec::DescribeSearchParameters(flat);
  EXPECT_EQ(descriptor.name, "generic");
  EXPECT_TRUE(descriptor.parameter_names.empty());

  auto parameters = hypervec::CreateSearchParameters(flat, config);
  ASSERT_NE(parameters, nullptr);
  EXPECT_EQ(parameters->sel, &selector);
  EXPECT_EQ(typeid(*parameters), typeid(hypervec::SearchParameters));
}

TEST(SearchParametersFactory, RejectsInvalidAndAmbiguousValues) {
  hypervec::IndexHNSWFlat hnsw(4, 8);

  hypervec::SearchConfig unknown;
  unknown.SetInteger("misspelled", 1);
  EXPECT_THROW(hypervec::CreateSearchParameters(hnsw, unknown),
               hypervec::HypervecException);

  hypervec::SearchConfig invalid;
  invalid.SetInteger("ef_search", 0);
  EXPECT_THROW(hypervec::CreateSearchParameters(hnsw, invalid),
               hypervec::HypervecException);

  hypervec::SearchConfig wrong_type;
  wrong_type.SetBoolean("ef_search", true);
  EXPECT_THROW(hypervec::CreateSearchParameters(hnsw, wrong_type),
               hypervec::HypervecException);

  hypervec::SearchConfig ambiguous;
  ambiguous.SetInteger("ef", 10).SetInteger("ef_search", 20);
  EXPECT_THROW(hypervec::CreateSearchParameters(hnsw, ambiguous),
               hypervec::HypervecException);

  hypervec::IndexFlatL2 flat(4);
  EXPECT_THROW(hypervec::CreateSearchParameters(flat, unknown),
               hypervec::HypervecException);
}
