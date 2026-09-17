/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "persistence/index_file_header.h"

#include <factory/engine.h>
#include <index/flat/index_flat.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <utils/log/assert.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace hypervec {

namespace {

constexpr uint64_t kMaxIndexMetadataBytes = 1ULL << 20;

uint32_t MetadataChecksum(std::string_view bytes) {
  uint32_t crc = 0xFFFFFFFFU;
  for (const unsigned char byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0xEDB88320U : 0U);
    }
  }
  return crc ^ 0xFFFFFFFFU;
}

const Index* UnwrapIndex(const Index& index, IdMode* id_mode) {
  const auto* id_map = dynamic_cast<const IndexIDMap*>(&index);
  if (id_map == nullptr) {
    *id_mode = IdMode::Sequential;
    return &index;
  }
  HYPERVEC_THROW_IF_NOT_MSG(id_map->index != nullptr,
                            "index metadata: ID map has no underlying index");
  HYPERVEC_THROW_IF_NOT_MSG(
      dynamic_cast<const IndexIDMap*>(id_map->index) == nullptr,
      "index metadata: nested ID maps are not supported");
  id_map->check_consistency();
  *id_mode = IdMode::External;
  return id_map->index;
}

void ValidateHnswStorage(const IndexHNSW& hnsw) {
  HYPERVEC_THROW_IF_NOT_MSG(hnsw.storage != nullptr,
                            "index metadata: HNSW storage is null");
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.storage->d == hnsw.d && hnsw.storage->n_total == hnsw.n_total &&
          hnsw.storage->is_trained == hnsw.is_trained &&
          hnsw.storage->metric_type == hnsw.metric_type &&
          hnsw.storage->metric_arg == hnsw.metric_arg,
      "index metadata: HNSW storage metadata disagrees with its wrapper");
}

void DescribeHnsw(const IndexHNSW& hnsw, IndexConfig* config) {
  ValidateHnswStorage(hnsw);
  const int level_zero_neighbors = hnsw.hnsw.NbNeighbors(0);
  HYPERVEC_THROW_IF_NOT_MSG(
      level_zero_neighbors > 0 && level_zero_neighbors % 2 == 0,
      "index metadata: invalid HNSW level-0 degree");
  config->build.graph_degree = level_zero_neighbors / 2;
  config->build.ef_construction = hnsw.hnsw.ef_construction;
  config->defaults.ef_search = hnsw.hnsw.ef_search;

  if (config->type == "hnsw_flat") {
    HYPERVEC_THROW_IF_NOT_MSG(
        dynamic_cast<const IndexFlat*>(hnsw.storage) != nullptr,
        "index metadata: HNSW-Flat storage is not an IndexFlat");
    config->storage.type = "flat";
    return;
  }
  if (config->type == "hnsw_pq") {
    const auto* storage = dynamic_cast<const IndexPQ*>(hnsw.storage);
    HYPERVEC_THROW_IF_NOT_MSG(
        storage != nullptr,
        "index metadata: HNSW-PQ storage is not an IndexPQ");
    config->storage.type = "pq";
    config->storage.pq_m = storage->pq.M;
    config->storage.nbits = storage->pq.nbits;
    return;
  }
  if (config->type == "hnsw_lvq") {
    const auto* storage = dynamic_cast<const IndexLVQ*>(hnsw.storage);
    HYPERVEC_THROW_IF_NOT_MSG(
        storage != nullptr,
        "index metadata: HNSW-LVQ storage is not an IndexLVQ");
    config->storage.type = "lvq";
    config->storage.nlocal = storage->lvq.nlocal;
    config->storage.nbits = storage->lvq.nbits;
    return;
  }
  HYPERVEC_THROW_MSG("index metadata: generic HNSW is not serializable");
}

IndexConfig DescribeIndexConfig(const Index& index) {
  HYPERVEC_THROW_IF_NOT_MSG(
      HasCapability(index.Capabilities(), IndexCapability::Serialize),
      "index metadata: index does not advertise serialization support");

  IdMode id_mode = IdMode::Sequential;
  const Index* concrete = UnwrapIndex(index, &id_mode);
  IndexConfig config;
  config.type = std::string(concrete->TypeName());
  config.dimension = concrete->d;
  config.metric = concrete->metric_type;
  if (concrete->metric_type == kMetricLp) {
    config.metric_arg = concrete->metric_arg;
  }
  config.id_mode = id_mode;

  if (const auto* hnsw = dynamic_cast<const IndexHNSW*>(concrete)) {
    DescribeHnsw(*hnsw, &config);
  } else if (const auto* ivf_pq = dynamic_cast<const IndexIVFPQ*>(concrete)) {
    config.storage.type = "pq";
    config.storage.pq_m = ivf_pq->pq.M;
    config.storage.nbits = ivf_pq->pq.nbits;
    config.build.nlist = ivf_pq->nlist;
    config.build.by_residual = ivf_pq->by_residual;
    config.build.use_precomputed_table = ivf_pq->use_precomputed_table;
    config.defaults.nprobe = ivf_pq->nprobe;
  } else if (const auto* ivf_lvq = dynamic_cast<const IndexIVFLVQ*>(concrete)) {
    config.storage.type = "lvq";
    config.storage.nlocal = ivf_lvq->lvq.nlocal;
    config.storage.nbits = ivf_lvq->lvq.nbits;
    config.build.nlist = ivf_lvq->nlist;
    config.build.by_residual = ivf_lvq->by_residual;
    config.defaults.nprobe = ivf_lvq->nprobe;
  } else if (const auto* ivf_flat =
                 dynamic_cast<const IndexIVFFlat*>(concrete)) {
    config.storage.type = "flat";
    config.build.nlist = ivf_flat->nlist;
    config.defaults.nprobe = ivf_flat->nprobe;
  } else if (const auto* pq = dynamic_cast<const IndexPQ*>(concrete)) {
    config.storage.type = "pq";
    config.storage.pq_m = pq->pq.M;
    config.storage.nbits = pq->pq.nbits;
  } else if (const auto* lvq = dynamic_cast<const IndexLVQ*>(concrete)) {
    config.storage.type = "lvq";
    config.storage.nlocal = lvq->lvq.nlocal;
    config.storage.nbits = lvq->lvq.nbits;
  } else if (config.type == "flat") {
    config.storage.type = "flat";
  } else {
    HYPERVEC_THROW_MSG("index metadata: unsupported index type '" +
                       config.type + "'");
  }

  return Engine().NormalizeConfig(config);
}

Index* MutableConcreteIndex(Index* index, const IndexConfig& config) {
  if (config.id_mode == IdMode::External) {
    auto* id_map = dynamic_cast<IndexIDMap*>(index);
    HYPERVEC_THROW_IF_NOT_MSG(
        id_map != nullptr && id_map->index != nullptr,
        "index metadata does not match the persisted ID mode");
    return id_map->index;
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      dynamic_cast<IndexIDMap*>(index) == nullptr,
      "index metadata does not match the persisted ID mode");
  return index;
}

}  // namespace

IndexFileMetadata GetIndexFileMetadata(const Index& index) {
  IndexFileMetadata metadata;
  metadata.config = DescribeIndexConfig(index);
  return metadata;
}

namespace persistence_detail {

uint32_t IndexFileEnvelopeMagic() { return fourcc("HVX2"); }

void WriteIndexFileHeader(const IndexFileMetadata& metadata, IOWriter* f) {
  HYPERVEC_THROW_IF_NOT_MSG(
      metadata.format_major == kIndexFileFormatMajor &&
          metadata.format_minor == kIndexFileFormatMinor,
      "index metadata: unsupported format version for writing");
  HYPERVEC_THROW_IF_NOT_MSG(
      metadata.algorithm_version == kIndexAlgorithmVersion,
      "index metadata: unsupported algorithm version for writing");
  const std::string config_json = SerializeIndexConfig(metadata.config);
  HYPERVEC_THROW_IF_NOT_MSG(
      !config_json.empty() && config_json.size() <= kMaxIndexMetadataBytes,
      "index metadata: normalized config is too large");

  const uint32_t magic = IndexFileEnvelopeMagic();
  const uint64_t config_size = config_json.size();
  const uint32_t checksum = MetadataChecksum(config_json);
  WRITE1(magic);
  WRITE1(metadata.format_major);
  WRITE1(metadata.format_minor);
  WRITE1(metadata.algorithm_version);
  WRITE1(config_size);
  WRITEANDCHECK(config_json.data(), config_json.size());
  WRITE1(checksum);
}

IndexFileMetadata ReadIndexFileHeader(IOReader* f) {
  IndexFileMetadata metadata;
  READ1(metadata.format_major);
  READ1(metadata.format_minor);
  READ1(metadata.algorithm_version);
  HYPERVEC_THROW_IF_NOT_MSG(metadata.format_major == kIndexFileFormatMajor,
                            "index metadata: unsupported format major version");
  HYPERVEC_THROW_IF_NOT_MSG(metadata.format_minor <= kIndexFileFormatMinor,
                            "index metadata: unsupported format minor version");
  HYPERVEC_THROW_IF_NOT_MSG(
      metadata.algorithm_version == kIndexAlgorithmVersion,
      "index metadata: unsupported algorithm version");

  uint64_t config_size = 0;
  READ1(config_size);
  HYPERVEC_THROW_IF_NOT_MSG(
      config_size > 0 && config_size <= kMaxIndexMetadataBytes &&
          config_size <=
              static_cast<uint64_t>((std::numeric_limits<size_t>::max)()),
      "index metadata: invalid config size");
  std::string config_json(static_cast<size_t>(config_size), '\0');
  READANDCHECK(config_json.data(), config_json.size());
  uint32_t checksum = 0;
  READ1(checksum);
  HYPERVEC_THROW_IF_NOT_MSG(checksum == MetadataChecksum(config_json),
                            "index metadata: config checksum mismatch");

  Engine engine;
  metadata.config = engine.NormalizeConfig(ParseIndexConfig(config_json));
  HYPERVEC_THROW_IF_NOT_MSG(
      SerializeIndexConfig(metadata.config) == config_json,
      "index metadata: config is not in canonical normalized form");
  metadata.is_legacy = false;
  return metadata;
}

void ApplyAndValidateIndexFileMetadata(Index* index,
                                       const IndexFileMetadata& metadata) {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "index metadata: payload index is null");
  Index* concrete = MutableConcreteIndex(index, metadata.config);
  HYPERVEC_THROW_IF_NOT_MSG(
      concrete->TypeName() == metadata.config.type &&
          concrete->d == metadata.config.dimension &&
          concrete->metric_type == metadata.config.metric,
      "index metadata does not match the persisted payload");
  if (metadata.config.metric == kMetricLp) {
    HYPERVEC_THROW_IF_NOT_MSG(
        metadata.config.metric_arg.has_value() &&
            concrete->metric_arg == *metadata.config.metric_arg,
        "index metadata does not match the persisted metric argument");
  }

  if (auto* hnsw = dynamic_cast<IndexHNSW*>(concrete)) {
    HYPERVEC_THROW_IF_NOT_MSG(metadata.config.defaults.ef_search.has_value(),
                              "index metadata: HNSW ef_search is missing");
    hnsw->hnsw.ef_search = *metadata.config.defaults.ef_search;
  }

  const IndexFileMetadata actual = GetIndexFileMetadata(*index);
  HYPERVEC_THROW_IF_NOT_MSG(
      SerializeIndexConfig(actual.config) ==
          SerializeIndexConfig(metadata.config),
      "index metadata does not match the persisted payload configuration");
}

}  // namespace persistence_detail
}  // namespace hypervec
