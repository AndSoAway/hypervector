/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/index_registry.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/hnsw.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/ivf/index_ivf.h>
#include <index/ivf/index_ivf_flat.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <utils/log/assert.h>
#include <utils/log/exception.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

constexpr idx_t kDefaultNlist = 1024;
constexpr idx_t kDefaultNprobe = 1;
constexpr idx_t kDefaultPqM = 8;
constexpr idx_t kDefaultNlocal = 16;
constexpr int kDefaultNbits = 8;
constexpr int kDefaultGraphDegree = 32;
constexpr int kDefaultEfConstruction = 40;
constexpr int kDefaultEfSearch = 16;

enum class IndexFamily {
  Scan,
  IVF,
  HNSW,
};

enum class StorageKind {
  Flat,
  PQ,
  LVQ,
};

void RejectParameter(bool is_set, const char* parameter,
                     const std::string& type) {
  if (is_set) {
    throw HypervecException(std::string(parameter) +
                            " is not valid for index type '" + type + "'");
  }
}

void RejectIvfParameters(const IndexConfig& config) {
  RejectParameter(config.build.nlist.has_value(), "build.nlist", config.type);
  RejectParameter(config.build.by_residual.has_value(), "build.by_residual",
                  config.type);
  RejectParameter(config.build.use_precomputed_table.has_value(),
                  "build.use_precomputed_table", config.type);
  RejectParameter(config.defaults.nprobe.has_value(), "defaults.nprobe",
                  config.type);
}

void RejectHnswParameters(const IndexConfig& config) {
  RejectParameter(config.build.graph_degree.has_value(), "build.graph_degree",
                  config.type);
  RejectParameter(config.build.ef_construction.has_value(),
                  "build.ef_construction", config.type);
  RejectParameter(config.defaults.ef_search.has_value(), "defaults.ef_search",
                  config.type);
}

void ValidateNbits(int nbits, int maximum) {
  if (nbits < 1 || nbits > maximum) {
    throw HypervecException("storage.nbits must be in [1, " +
                            std::to_string(maximum) + "]");
  }
}

void ValidateNprobe(idx_t nprobe, idx_t nlist, const char* parameter) {
  if (nprobe <= 0 || nprobe > nlist) {
    throw HypervecException(std::string(parameter) +
                            " must be in [1, build.nlist]");
  }
}

void ValidateEfSearch(int ef_search, const char* parameter) {
  if (ef_search <= 0) {
    throw HypervecException(std::string(parameter) + " must be positive");
  }
}

size_t ToSize(idx_t value, const char* parameter) {
  if (value < 0 ||
      static_cast<uint64_t>(value) >
          static_cast<uint64_t>((std::numeric_limits<size_t>::max)())) {
    throw HypervecException(std::string(parameter) +
                            " is outside the platform size range");
  }
  return static_cast<size_t>(value);
}

void NormalizePQ(IndexConfig* config) {
  config->storage.pq_m = config->storage.pq_m.value_or(kDefaultPqM);
  config->storage.nbits = config->storage.nbits.value_or(kDefaultNbits);
  if (*config->storage.pq_m <= 0) {
    throw HypervecException("storage.pq_m must be positive");
  }
  if (config->dimension % *config->storage.pq_m != 0) {
    throw HypervecException("dimension must be divisible by storage.pq_m");
  }
  ValidateNbits(*config->storage.nbits, HYPERVEC_PQ_MAX_NBITS);
  const size_t codewords = size_t{1} << *config->storage.nbits;
  mul_no_overflow(ToSize(config->dimension, "dimension"), codewords,
                  "PQ centroid count");
}

void NormalizeLVQ(IndexConfig* config) {
  config->storage.nlocal = config->storage.nlocal.value_or(kDefaultNlocal);
  config->storage.nbits = config->storage.nbits.value_or(kDefaultNbits);
  if (*config->storage.nlocal <= 0 ||
      *config->storage.nlocal > static_cast<idx_t>(INT_MAX)) {
    throw HypervecException("storage.nlocal must be in [1, INT_MAX]");
  }
  ValidateNbits(*config->storage.nbits, HYPERVEC_LVQ_MAX_NBITS);
  const size_t local_count = ToSize(*config->storage.nlocal, "storage.nlocal");
  const size_t codewords = size_t{1} << *config->storage.nbits;
  const size_t codebook_count =
      mul_no_overflow(local_count, codewords, "LVQ codebook count");
  mul_no_overflow(codebook_count, ToSize(config->dimension, "dimension"),
                  "LVQ codebook count");
}

void NormalizeIVF(IndexConfig* config) {
  config->build.nlist = config->build.nlist.value_or(kDefaultNlist);
  config->defaults.nprobe = config->defaults.nprobe.value_or(kDefaultNprobe);
  if (*config->build.nlist <= 0) {
    throw HypervecException("build.nlist must be positive");
  }
  mul_no_overflow(ToSize(*config->build.nlist, "build.nlist"),
                  ToSize(config->dimension, "dimension"), "IVF centroid count");
  ValidateNprobe(*config->defaults.nprobe, *config->build.nlist,
                 "defaults.nprobe");
}

void NormalizeHNSW(IndexConfig* config) {
  config->build.graph_degree =
      config->build.graph_degree.value_or(kDefaultGraphDegree);
  config->build.ef_construction =
      config->build.ef_construction.value_or(kDefaultEfConstruction);
  config->defaults.ef_search =
      config->defaults.ef_search.value_or(kDefaultEfSearch);
  if (*config->build.graph_degree < 2 ||
      *config->build.graph_degree > INT_MAX / 2) {
    throw HypervecException("build.graph_degree must be in [2, INT_MAX / 2]");
  }
  if (*config->build.ef_construction <= 0) {
    throw HypervecException("build.ef_construction must be positive");
  }
  ValidateEfSearch(*config->defaults.ef_search, "defaults.ef_search");
}

const char* StorageName(StorageKind storage) {
  switch (storage) {
    case StorageKind::Flat:
      return "flat";
    case StorageKind::PQ:
      return "pq";
    case StorageKind::LVQ:
      return "lvq";
    default:
      HYPERVEC_THROW_MSG("unsupported built-in storage kind");
  }
}

void NormalizeBuiltin(IndexConfig* config, IndexFamily family,
                      StorageKind storage) {
  const std::string expected_storage = StorageName(storage);
  if (config->storage.type.empty()) {
    config->storage.type = expected_storage;
  } else if (config->storage.type != expected_storage) {
    throw HypervecException("storage.type '" + config->storage.type +
                            "' conflicts with index type '" + config->type +
                            "'");
  }

  if (family == IndexFamily::IVF) {
    NormalizeIVF(config);
  } else {
    RejectIvfParameters(*config);
  }
  if (family == IndexFamily::HNSW) {
    NormalizeHNSW(config);
  } else {
    RejectHnswParameters(*config);
  }

  if (storage == StorageKind::PQ) {
    RejectParameter(config->storage.nlocal.has_value(), "storage.nlocal",
                    config->type);
    NormalizePQ(config);
  } else if (storage == StorageKind::LVQ) {
    RejectParameter(config->storage.pq_m.has_value(), "storage.pq_m",
                    config->type);
    NormalizeLVQ(config);
  } else {
    RejectParameter(config->storage.pq_m.has_value(), "storage.pq_m",
                    config->type);
    RejectParameter(config->storage.nlocal.has_value(), "storage.nlocal",
                    config->type);
    RejectParameter(config->storage.nbits.has_value(), "storage.nbits",
                    config->type);
  }

  if (family == IndexFamily::IVF && storage != StorageKind::Flat) {
    config->build.by_residual = config->build.by_residual.value_or(true);
  } else {
    RejectParameter(config->build.by_residual.has_value(), "build.by_residual",
                    config->type);
  }

  if (family == IndexFamily::IVF && storage == StorageKind::PQ) {
    config->build.use_precomputed_table =
        config->build.use_precomputed_table.value_or(0);
    const int table_mode = *config->build.use_precomputed_table;
    if (table_mode != 0 && table_mode != 1) {
      throw HypervecException("build.use_precomputed_table must be 0 or 1");
    }
    if (table_mode == 1) {
      const size_t list_subquantizers =
          mul_no_overflow(ToSize(*config->build.nlist, "build.nlist"),
                          ToSize(*config->storage.pq_m, "storage.pq_m"),
                          "IVF-PQ precomputed table count");
      mul_no_overflow(list_subquantizers, size_t{1} << *config->storage.nbits,
                      "IVF-PQ precomputed table count");
    }
  } else {
    RejectParameter(config->build.use_precomputed_table.has_value(),
                    "build.use_precomputed_table", config->type);
  }
}

void NormalizeFlat(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::Scan, StorageKind::Flat);
}

void NormalizeScanPQ(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::Scan, StorageKind::PQ);
}

void NormalizeScanLVQ(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::Scan, StorageKind::LVQ);
}

void NormalizeIVFFlat(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::IVF, StorageKind::Flat);
}

void NormalizeIVFPQ(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::IVF, StorageKind::PQ);
}

void NormalizeIVFLVQ(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::IVF, StorageKind::LVQ);
}

void NormalizeHNSWFlat(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::HNSW, StorageKind::Flat);
}

void NormalizeHNSWPQ(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::HNSW, StorageKind::PQ);
}

void NormalizeHNSWLVQ(IndexConfig& config) {
  NormalizeBuiltin(&config, IndexFamily::HNSW, StorageKind::LVQ);
}

void NormalizeScanSearch(const IndexConfig& index_config,
                         SearchConfig& search_config) {
  RejectParameter(search_config.nprobe.has_value(), "nprobe",
                  index_config.type);
  RejectParameter(search_config.ef_search.has_value(), "ef_search",
                  index_config.type);
}

void NormalizeIVFSearch(const IndexConfig& index_config,
                        SearchConfig& search_config) {
  RejectParameter(search_config.ef_search.has_value(), "ef_search",
                  index_config.type);
  HYPERVEC_THROW_IF_NOT_MSG(index_config.defaults.nprobe.has_value() &&
                                index_config.build.nlist.has_value(),
                            "normalized IVF config is missing search defaults");
  search_config.nprobe =
      search_config.nprobe.value_or(*index_config.defaults.nprobe);
  ValidateNprobe(*search_config.nprobe, *index_config.build.nlist, "nprobe");
}

void NormalizeHNSWSearch(const IndexConfig& index_config,
                         SearchConfig& search_config) {
  RejectParameter(search_config.nprobe.has_value(), "nprobe",
                  index_config.type);
  HYPERVEC_THROW_IF_NOT_MSG(
      index_config.defaults.ef_search.has_value(),
      "normalized HNSW config is missing search defaults");
  search_config.ef_search =
      search_config.ef_search.value_or(*index_config.defaults.ef_search);
  ValidateEfSearch(*search_config.ef_search, "ef_search");
}

std::unique_ptr<SearchParameters> CreateScanSearchParameters(
    const IndexConfig&, const SearchConfig&) {
  return std::make_unique<SearchParameters>();
}

std::unique_ptr<SearchParameters> CreateIVFSearchParameters(
    const IndexConfig&, const SearchConfig& search_config) {
  auto parameters = std::make_unique<IVFSearchParameters>();
  parameters->nprobe = *search_config.nprobe;
  return parameters;
}

std::unique_ptr<SearchParameters> CreateHNSWSearchParameters(
    const IndexConfig&, const SearchConfig& search_config) {
  auto parameters = std::make_unique<SearchParametersHNSW>();
  parameters->ef_search = *search_config.ef_search;
  return parameters;
}

const std::vector<MetricType>& FlatMetrics() {
  static const std::vector<MetricType> metrics = {
      kMetricInnerProduct, kMetricL2,
      kMetricL1,           kMetricLinf,
      kMetricLp,           kMetricCanberra,
      kMetricBrayCurtis,   kMetricJensenShannon,
      kMetricJaccard,      kMetricNaNEuclidean,
      kMetricGower,
  };
  return metrics;
}

std::unique_ptr<Index> CreateFlat(const IndexConfig& config,
                                  const ResourceContext&) {
  std::unique_ptr<IndexFlat> index;
  if (config.metric == kMetricL2) {
    index = std::make_unique<IndexFlatL2>(config.dimension);
  } else if (config.metric == kMetricInnerProduct) {
    index = std::make_unique<IndexFlatIP>(config.dimension);
  } else {
    index = std::make_unique<IndexFlat>(config.dimension, config.metric);
  }
  if (config.metric_arg.has_value()) {
    index->metric_arg = *config.metric_arg;
  }
  return index;
}

std::unique_ptr<Index> CreatePQ(const IndexConfig& config,
                                const ResourceContext&) {
  return std::make_unique<IndexPQ>(config.dimension, *config.storage.pq_m,
                                   *config.storage.nbits, config.metric);
}

std::unique_ptr<Index> CreateLVQ(const IndexConfig& config,
                                 const ResourceContext&) {
  return std::make_unique<IndexLVQ>(config.dimension, *config.storage.nlocal,
                                    *config.storage.nbits, config.metric);
}

std::unique_ptr<Index> CreateIVFFlat(const IndexConfig& config,
                                     const ResourceContext&) {
  auto index = std::make_unique<IndexIVFFlat>(
      config.dimension, *config.build.nlist, config.metric);
  index->nprobe = *config.defaults.nprobe;
  return index;
}

std::unique_ptr<Index> CreateIVFPQ(const IndexConfig& config,
                                   const ResourceContext&) {
  auto index = std::make_unique<IndexIVFPQ>(
      config.dimension, *config.build.nlist, *config.storage.pq_m,
      *config.storage.nbits, config.metric);
  index->nprobe = *config.defaults.nprobe;
  index->by_residual = *config.build.by_residual;
  index->use_precomputed_table = *config.build.use_precomputed_table;
  return index;
}

std::unique_ptr<Index> CreateIVFLVQ(const IndexConfig& config,
                                    const ResourceContext&) {
  auto index = std::make_unique<IndexIVFLVQ>(
      config.dimension, *config.build.nlist, *config.storage.nlocal,
      *config.storage.nbits, config.metric);
  index->nprobe = *config.defaults.nprobe;
  index->by_residual = *config.build.by_residual;
  return index;
}

std::unique_ptr<Index> CreateHNSWFlat(const IndexConfig& config,
                                      const ResourceContext&) {
  auto index = std::make_unique<IndexHNSWFlat>(
      static_cast<int>(config.dimension), *config.build.graph_degree,
      config.metric);
  index->hnsw.ef_construction = *config.build.ef_construction;
  index->hnsw.ef_search = *config.defaults.ef_search;
  if (config.metric_arg.has_value()) {
    index->metric_arg = *config.metric_arg;
    index->storage->metric_arg = *config.metric_arg;
  }
  return index;
}

std::unique_ptr<Index> CreateHNSWPQ(const IndexConfig& config,
                                    const ResourceContext&) {
  auto index = std::make_unique<IndexHNSWPQ>(
      static_cast<int>(config.dimension),
      static_cast<int>(*config.storage.pq_m), *config.storage.nbits,
      *config.build.graph_degree, config.metric);
  index->hnsw.ef_construction = *config.build.ef_construction;
  index->hnsw.ef_search = *config.defaults.ef_search;
  return index;
}

std::unique_ptr<Index> CreateHNSWLVQ(const IndexConfig& config,
                                     const ResourceContext&) {
  auto index = std::make_unique<IndexHNSWLVQ>(
      static_cast<int>(config.dimension),
      static_cast<int>(*config.storage.nlocal), *config.storage.nbits,
      *config.build.graph_degree, config.metric);
  index->hnsw.ef_construction = *config.build.ef_construction;
  index->hnsw.ef_search = *config.defaults.ef_search;
  return index;
}

IndexCapabilities FlatCapabilities() {
  return IndexCapability::BatchBuild | IndexCapability::IncrementalAdd |
         IndexCapability::KnnSearch | IndexCapability::Filter |
         IndexCapability::ConcurrentSearch;
}

IndexCapabilities TrainedScanCapabilities(bool clone) {
  IndexCapabilities capabilities =
      IndexCapability::NeedsTrain | IndexCapability::BatchBuild |
      IndexCapability::IncrementalAdd | IndexCapability::KnnSearch |
      IndexCapability::ConcurrentSearch | IndexCapability::Serialize;
  if (clone) {
    capabilities |= IndexCapability::Clone;
  }
  return capabilities;
}

IndexCapabilities IVFCapabilities(bool range, bool clone) {
  IndexCapabilities capabilities =
      IndexCapability::NeedsTrain | IndexCapability::BatchBuild |
      IndexCapability::IncrementalAdd | IndexCapability::KnnSearch |
      IndexCapability::Filter | IndexCapability::ConcurrentSearch |
      IndexCapability::Serialize;
  if (range) {
    capabilities |= IndexCapability::RangeSearch;
  }
  if (clone) {
    capabilities |= IndexCapability::Clone;
  }
  return capabilities;
}

IndexCapabilities HNSWCapabilities(bool trained) {
  IndexCapabilities capabilities =
      IndexCapability::BatchBuild | IndexCapability::IncrementalAdd |
      IndexCapability::KnnSearch | IndexCapability::Filter |
      IndexCapability::ConcurrentSearch;
  if (trained) {
    capabilities |= IndexCapability::NeedsTrain;
    capabilities |= IndexCapability::Serialize;
  }
  return capabilities;
}

}  // namespace

void RegisterBuiltinIndexes() {
  static std::once_flag once;
  std::call_once(once, [] {
    IndexRegistry& registry = IndexRegistry::Global();
    const std::vector<MetricType> l2 = {kMetricL2};
    const std::vector<MetricType> ivf_flat_metrics = {kMetricL2,
                                                      kMetricInnerProduct};
    const std::vector<std::string> cpu = {"cpu"};

    registry.Register({"flat",
                       {"indexflat"},
                       FlatCapabilities(),
                       FlatMetrics(),
                       cpu,
                       NormalizeFlat,
                       CreateFlat,
                       NormalizeScanSearch,
                       CreateScanSearchParameters});
    registry.Register({"pq",
                       {"indexpq"},
                       TrainedScanCapabilities(false),
                       l2,
                       cpu,
                       NormalizeScanPQ,
                       CreatePQ,
                       NormalizeScanSearch,
                       CreateScanSearchParameters});
    registry.Register({"lvq",
                       {"indexlvq"},
                       TrainedScanCapabilities(true),
                       l2,
                       cpu,
                       NormalizeScanLVQ,
                       CreateLVQ,
                       NormalizeScanSearch,
                       CreateScanSearchParameters});
    registry.Register({"ivf_flat",
                       {"ivfflat", "indexivfflat"},
                       IVFCapabilities(true, false),
                       ivf_flat_metrics,
                       cpu,
                       NormalizeIVFFlat,
                       CreateIVFFlat,
                       NormalizeIVFSearch,
                       CreateIVFSearchParameters});
    registry.Register({"ivf_pq",
                       {"ivfpq", "indexivfpq"},
                       IVFCapabilities(true, false),
                       l2,
                       cpu,
                       NormalizeIVFPQ,
                       CreateIVFPQ,
                       NormalizeIVFSearch,
                       CreateIVFSearchParameters});
    registry.Register({"ivf_lvq",
                       {"ivflvq", "indexivflvq"},
                       IVFCapabilities(true, true),
                       l2,
                       cpu,
                       NormalizeIVFLVQ,
                       CreateIVFLVQ,
                       NormalizeIVFSearch,
                       CreateIVFSearchParameters});
    registry.Register({"hnsw_flat",
                       {"hnswflat", "indexhnswflat", "autoindex"},
                       HNSWCapabilities(false),
                       FlatMetrics(),
                       cpu,
                       NormalizeHNSWFlat,
                       CreateHNSWFlat,
                       NormalizeHNSWSearch,
                       CreateHNSWSearchParameters});
    registry.Register({"hnsw_pq",
                       {"hnswpq", "indexhnswpq"},
                       HNSWCapabilities(true),
                       l2,
                       cpu,
                       NormalizeHNSWPQ,
                       CreateHNSWPQ,
                       NormalizeHNSWSearch,
                       CreateHNSWSearchParameters});
    registry.Register({"hnsw_lvq",
                       {"hnswlvq", "indexhnswlvq"},
                       HNSWCapabilities(true),
                       l2,
                       cpu,
                       NormalizeHNSWLVQ,
                       CreateHNSWLVQ,
                       NormalizeHNSWSearch,
                       CreateHNSWSearchParameters});
  });
}

}  // namespace hypervec
