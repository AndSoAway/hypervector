/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 *
 * HNSW-only index write implementation
 */

#include <index/diskann/index_diskann.h>
#include <index/flat/index_flat.h>
#include <index/graph/graph_validation.h>
#include <index/hnsw/hnsw_graph_storage.h>
#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/index_hnsw_lvq.h>
#include <index/hnsw/index_hnsw_pq.h>
#include <index/idmap/index_id_map.h>
#include <index/ivf/index_ivf_flat.h>
#include <index/lsh/index_lsh.h>
#include <index/nsg/index_nsg.h>
#include <index/nsw/index_nsw.h>
#include <index/pretransform/index_pre_transform.h>
#include <index/vamana/index_vamana.h>
#include <invlists/inverted_lists.h>
#include <persistence/index_io.h>
#include <persistence/index_io_builtins.h>
#include <persistence/index_io_registry.h>
#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/lvq/lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/pq/pq.h>
#include <quantization/rabitq/index_ivf_rabitq.h>
#include <transform/opq_matrix.h>
#include <transform/vector_transform.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <typeindex>
#include <vector>

namespace hypervec {

static void write_index_header(const Index& idx, IOWriter* f) {
  WRITE1(idx.d);
  WRITE1(idx.n_total);
  idx_t dummy = 1 << 20;
  WRITE1(dummy);
  WRITE1(dummy);
  WRITE1(idx.is_trained);
  int metric = static_cast<int>(idx.metric_type);
  WRITE1(metric);
  if (idx.metric_type > 1) {
    WRITE1(idx.metric_arg);
  }
}

static void write_pq(const ProductQuantizer& pq, IOWriter* f) {
  WRITE1(pq.d);
  WRITE1(pq.M);
  WRITE1(pq.nbits);
  WRITEVECTOR(pq.centroids);
}

static void write_lvq(const LocalVectorQuantizer& lvq, IOWriter* f) {
  WRITE1(lvq.d);
  WRITE1(lvq.nlocal);
  WRITE1(lvq.nbits);
  WRITEVECTOR(lvq.local_centroids);
  WRITEVECTOR(lvq.residual_codebooks);
}

static void write_linear_transform(const LinearTransform& transform,
                                   IOWriter* f) {
  HYPERVEC_THROW_IF_NOT_MSG(
      transform.is_trained,
      "LinearTransform serialize: transform must be trained");
  HYPERVEC_THROW_IF_NOT_MSG(
      transform.d_in > 0 && transform.d_out > 0,
      "LinearTransform serialize: dimensions must be positive");
  const size_t matrix_size = mul_no_overflow(
      static_cast<size_t>(transform.d_in), static_cast<size_t>(transform.d_out),
      "LinearTransform matrix");
  HYPERVEC_THROW_IF_NOT_MSG(
      transform.matrix.size() == matrix_size,
      "LinearTransform serialize: matrix size does not match dimensions");
  HYPERVEC_THROW_IF_NOT_MSG(
      transform.bias.empty() ||
          transform.bias.size() == static_cast<size_t>(transform.d_out),
      "LinearTransform serialize: bias size does not match output dimension");
  LinearTransform validated(transform.d_in, transform.d_out);
  validated.SetTransform(transform.matrix, transform.bias,
                         transform.is_orthonormal);

  WRITE1(transform.d_in);
  WRITE1(transform.d_out);
  const uint8_t is_orthonormal = transform.is_orthonormal;
  const uint8_t has_bias = !transform.bias.empty();
  WRITE1(is_orthonormal);
  WRITE1(has_bias);
  WRITEVECTOR(transform.matrix);
  if (has_bias) {
    WRITEVECTOR(transform.bias);
  }
}

static void write_transform(const VectorTransform& transform, IOWriter* f) {
  const auto* opq = dynamic_cast<const OPQMatrix*>(&transform);
  if (opq) {
    HYPERVEC_THROW_IF_NOT_MSG(
        opq->parameters.iterations > 0 &&
            opq->parameters.pq_parameters.niter > 0 &&
            opq->parameters.pq_parameters.nredo > 0,
        "OPQMatrix serialize: training parameters must be positive");
    HYPERVEC_THROW_IF_NOT_MSG(
        opq->d_in == opq->d_out && opq->is_orthonormal && opq->bias.empty(),
        "OPQMatrix serialize: transform must be an unbiased orthogonal "
        "rotation");
    uint32_t h = fourcc("OPQt");
    WRITE1(h);
    WRITE1(opq->subquantizer_count);
    WRITE1(opq->nbits);
    WRITE1(opq->parameters.iterations);
    WRITE1(opq->parameters.pq_parameters.niter);
    WRITE1(opq->parameters.pq_parameters.seed);
    WRITE1(opq->parameters.pq_parameters.nredo);
    const uint8_t verbose = opq->parameters.pq_parameters.verbose;
    WRITE1(verbose);
    write_linear_transform(*opq, f);
    return;
  }

  const auto* linear = dynamic_cast<const LinearTransform*>(&transform);
  if (linear) {
    uint32_t h = fourcc("LiTr");
    WRITE1(h);
    write_linear_transform(*linear, f);
    return;
  }

  HYPERVEC_THROW_MSG("unsupported vector transform type for writing");
}

static void write_HNSW(const HNSW& hnsw, IOWriter* f) {
  int M = hnsw.NbNeighbors(0);
  WRITE1(M);
  WRITE1(hnsw.ef_construction);
  WRITE1(hnsw.max_level);
  WRITE1(hnsw.entry_point);
  int nb_levels =
      hnsw.levels.size() > 0 ? hnsw.levels[hnsw.levels.size() - 1] : 0;
  WRITE1(nb_levels);
  WRITEVECTOR(hnsw.cum_nneighbor_per_level);
  WRITEVECTOR(hnsw.levels);
  WRITEVECTOR(hnsw.neighbors);
  WRITEVECTOR(hnsw.offsets);
}

static void write_random_access_payload(const RandomAccessReader& reader,
                                        IOWriter* f) {
  HYPERVEC_THROW_IF_NOT_MSG(
      reader.Size() <= std::numeric_limits<size_t>::max(),
      "random-access payload exceeds the serialization size limit");
  const size_t payload_size = static_cast<size_t>(reader.Size());
  WRITE1(payload_size);
  constexpr size_t kChunkSize = 1U << 20;
  std::vector<uint8_t> buffer(std::min(payload_size, kChunkSize));
  size_t offset = 0;
  while (offset < payload_size) {
    const size_t chunk = std::min(payload_size - offset, buffer.size());
    reader.ReadAt(static_cast<uint64_t>(offset), buffer.data(), chunk);
    WRITEANDCHECK(buffer.data(), chunk);
    offset += chunk;
  }
}

namespace persistence_internal {

namespace {

void WriteInvertedLists(const IndexIVF& index, size_t code_size, IOWriter* f) {
  HYPERVEC_THROW_IF_NOT_MSG(index.invlists != nullptr,
                            "IndexIVF serialize: inverted lists are null");
  for (size_t list_no = 0; list_no < static_cast<size_t>(index.nlist);
       ++list_no) {
    const size_t list_size = index.invlists->list_size(list_no);
    WRITE1(list_size);
    if (list_size == 0) {
      continue;
    }
    InvertedLists::ScopedIds ids(index.invlists, list_no);
    InvertedLists::ScopedCodes codes(index.invlists, list_no);
    WRITEANDCHECK(ids.get(), list_size);
    const size_t code_bytes =
        mul_no_overflow(list_size, code_size, "IndexIVF list codes");
    WRITEANDCHECK(codes.get(), code_bytes);
  }
}

}  // namespace

void WriteFlatL2Payload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& flat = static_cast<const IndexFlatL2&>(index);
  write_index_header(flat, f);
  WRITEVECTOR(flat.codes);
}

void WriteFlatIPPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& flat = static_cast<const IndexFlatIP&>(index);
  write_index_header(flat, f);
  WRITEVECTOR(flat.codes);
}

void WritePQPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& pq = static_cast<const IndexPQ&>(index);
  write_index_header(pq, f);
  write_pq(pq.pq, f);
  WRITEVECTOR(pq.codes);
}

void WriteLVQPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& lvq = static_cast<const IndexLVQ&>(index);
  write_index_header(lvq, f);
  write_lvq(lvq.lvq, f);
  WRITEVECTOR(lvq.codes);
}

void WriteIVFFlatPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& ivf = static_cast<const IndexIVFFlat&>(index);
  write_index_header(ivf, f);
  WRITE1(ivf.nlist);
  WRITE1(ivf.nprobe);
  WRITEVECTOR(ivf.centroids);
  const size_t code_size = mul_no_overflow(
      static_cast<size_t>(ivf.d), sizeof(float), "IndexIVFFlat code size");
  WriteInvertedLists(ivf, code_size, f);
}

void WriteIVFPQPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& ivf = static_cast<const IndexIVFPQ&>(index);
  write_index_header(ivf, f);
  WRITE1(ivf.nlist);
  WRITE1(ivf.nprobe);
  WRITEVECTOR(ivf.centroids);
  const int8_t by_residual = ivf.by_residual ? 1 : 0;
  const int precomputed_mode = ivf.use_precomputed_table;
  WRITE1(by_residual);
  WRITE1(precomputed_mode);
  write_pq(ivf.pq, f);
  WRITEVECTOR(ivf.precomputed_table);
  WriteInvertedLists(ivf, ivf.pq.code_size, f);
}

void WriteIVFLVQPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& ivf = static_cast<const IndexIVFLVQ&>(index);
  write_index_header(ivf, f);
  WRITE1(ivf.nlist);
  WRITE1(ivf.nprobe);
  WRITEVECTOR(ivf.centroids);
  const int8_t by_residual = ivf.by_residual ? 1 : 0;
  WRITE1(by_residual);
  write_lvq(ivf.lvq, f);
  WriteInvertedLists(ivf, ivf.lvq.code_size, f);
}

void ValidateIVFRaBitQForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& ivf = static_cast<const IndexIVFRaBitQ&>(index);
  HYPERVEC_THROW_IF_NOT_MSG(
      ivf.is_trained && ivf.metric_type == kMetricL2 && ivf.rabitq != nullptr &&
          ivf.invlists != nullptr && ivf.rabitq->Dimension() == ivf.d &&
          ivf.invlists->code_size == ivf.rabitq->CodeSize(),
      "IndexIVFRaBitQ serialize: index metadata is inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      ivf.nlist > 0 && ivf.nprobe > 0,
      "IndexIVFRaBitQ serialize: nlist and nprobe must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      ivf.n_total >= 0 && static_cast<uint64_t>(ivf.n_total) <=
                              std::numeric_limits<size_t>::max(),
      "IndexIVFRaBitQ serialize: n_total does not fit in size_t");
  const size_t expected_total = static_cast<size_t>(ivf.n_total);
  const size_t centroid_count =
      mul_no_overflow(static_cast<size_t>(ivf.nlist),
                      static_cast<size_t>(ivf.d), "IndexIVFRaBitQ centroids");
  HYPERVEC_THROW_IF_NOT_MSG(
      ivf.centroids.size() == centroid_count,
      "IndexIVFRaBitQ serialize: centroid count is inconsistent");
  for (float centroid : ivf.centroids) {
    HYPERVEC_THROW_IF_NOT_MSG(
        std::isfinite(centroid),
        "IndexIVFRaBitQ serialize: centroids must be finite");
  }

  size_t stored_total = 0;
  for (size_t list_no = 0; list_no < static_cast<size_t>(ivf.nlist);
       ++list_no) {
    const size_t list_size = ivf.invlists->list_size(list_no);
    stored_total =
        add_no_overflow(stored_total, list_size, "IndexIVFRaBitQ entry count");
    if (list_size == 0) {
      continue;
    }
    InvertedLists::ScopedCodes codes(ivf.invlists, list_no);
    const size_t factor_offset = ivf.rabitq->BitBytes();
    for (size_t offset = 0; offset < list_size; ++offset) {
      const uint8_t* code = codes.get() + offset * ivf.rabitq->CodeSize();
      float norm_squared;
      float scale;
      std::memcpy(&norm_squared, code + factor_offset, sizeof(float));
      std::memcpy(&scale, code + factor_offset + sizeof(float), sizeof(float));
      HYPERVEC_THROW_IF_NOT_MSG(
          std::isfinite(norm_squared) && norm_squared >= 0.0F &&
              std::isfinite(scale) && scale >= 0.0F,
          "IndexIVFRaBitQ serialize: code factors are invalid");
    }
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      stored_total == expected_total,
      "IndexIVFRaBitQ serialize: list entries do not match n_total");
}

void WriteIVFRaBitQPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& ivf = static_cast<const IndexIVFRaBitQ&>(index);
  write_index_header(ivf, f);
  WRITE1(ivf.nlist);
  WRITE1(ivf.nprobe);
  WRITEVECTOR(ivf.centroids);
  const uint8_t by_residual = ivf.by_residual;
  WRITE1(by_residual);
  const uint64_t random_seed = ivf.rabitq->Seed();
  const int rotation_rounds = ivf.rabitq->RotationRounds();
  WRITE1(random_seed);
  WRITE1(rotation_rounds);
  WriteInvertedLists(ivf, ivf.rabitq->CodeSize(), f);
}

void ValidateIDMapForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  static_cast<const IndexIDMap&>(index).check_consistency();
}

void WriteIDMapPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& id_map = static_cast<const IndexIDMap&>(index);
  write_index_header(id_map, f);
  WRITEVECTOR(id_map.rev_map);
  WriteIndex(id_map.index, f, 0);
}

void ValidatePreTransformForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& wrapper = static_cast<const IndexPreTransform&>(index);
  HYPERVEC_THROW_IF_NOT_MSG(
      wrapper.transform != nullptr && wrapper.index != nullptr,
      "IndexPreTransform serialize: components must not be null");
  HYPERVEC_THROW_IF_NOT_MSG(
      wrapper.is_trained && wrapper.transform->is_trained &&
          wrapper.index->is_trained,
      "IndexPreTransform serialize: all components must be trained");
  HYPERVEC_THROW_IF_NOT_MSG(
      wrapper.d == wrapper.transform->d_in &&
          wrapper.transform->d_out == wrapper.index->d &&
          wrapper.n_total == wrapper.index->n_total &&
          wrapper.metric_type == wrapper.index->metric_type &&
          wrapper.metric_arg == wrapper.index->metric_arg,
      "IndexPreTransform serialize: wrapper metadata is inconsistent");
}

void WritePreTransformPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& wrapper = static_cast<const IndexPreTransform&>(index);
  write_index_header(wrapper, f);
  write_transform(*wrapper.transform, f);
  WriteIndex(wrapper.index.get(), f, 0);
}

void ValidateLSHForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& lsh = static_cast<const IndexLSH&>(index);
  const size_t expected_code_size = mul_no_overflow(
      static_cast<size_t>(lsh.d), sizeof(float), "IndexLSH code size");
  const size_t expected_hyperplanes = mul_no_overflow(
      mul_no_overflow(lsh.Options().table_count, lsh.Options().bits_per_table,
                      "IndexLSH hyperplane count"),
      static_cast<size_t>(lsh.d), "IndexLSH hyperplane elements");
  HYPERVEC_THROW_IF_NOT_MSG(
      lsh.is_trained && lsh.metric_type == kMetricInnerProduct &&
          lsh.CodeStore().CodeSize() == expected_code_size &&
          lsh.CodeStore().Size() == lsh.n_total,
      "IndexLSH serialize: index metadata does not match stored vectors");
  HYPERVEC_THROW_IF_NOT_MSG(
      lsh.Hyperplanes().size() == expected_hyperplanes,
      "IndexLSH serialize: hyperplane count does not match the options");
}

void WriteLSHPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& lsh = static_cast<const IndexLSH&>(index);
  write_index_header(lsh, f);
  const LSHIndexOptions& options = lsh.Options();
  WRITE1(options.table_count);
  WRITE1(options.bits_per_table);
  WRITE1(options.probe_count);
  WRITE1(options.candidate_limit);
  WRITE1(options.random_seed);
  WRITEVECTOR(lsh.Hyperplanes());
  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(lsh.n_total),
                      lsh.CodeStore().CodeSize(), "IndexLSH codes");
  WRITE1(code_bytes);
  WRITEANDCHECK(lsh.CodeStore().Data(), code_bytes);
}

void ValidateNSWFlatForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& nsw = static_cast<const IndexNSWFlat&>(index);
  HYPERVEC_THROW_IF_NOT_MSG(
      nsw.CodeStore().Size() == nsw.n_total &&
          nsw.Graph().NodeCount() == static_cast<size_t>(nsw.n_total),
      "IndexNSWFlat serialize: stored counts do not match n_total");
  HYPERVEC_THROW_IF_NOT_MSG(
      nsw.QuantizerModel().Dimension() == nsw.d &&
          nsw.QuantizerModel().Metric() == nsw.metric_type &&
          nsw.QuantizerModel().IsTrained() == nsw.is_trained,
      "IndexNSWFlat serialize: quantizer metadata does not match the index");
  const GraphValidationReport report =
      ValidateGraph(nsw.Graph(), nsw.EntryPoint());
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid(),
      "IndexNSWFlat serialize: graph structure is invalid");
}

void WriteNSWFlatPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& nsw = static_cast<const IndexNSWFlat&>(index);
  write_index_header(nsw, f);
  const NSWIndexOptions& options = nsw.Options();
  WRITE1(options.max_degree);
  WRITE1(options.ef_construction);
  WRITE1(options.ef_search);
  const uint8_t check_relative_distance = options.check_relative_distance;
  const uint8_t fill_to_max_degree = options.fill_to_max_degree;
  WRITE1(check_relative_distance);
  WRITE1(fill_to_max_degree);
  const GraphId entry_point = nsw.EntryPoint();
  WRITE1(entry_point);
  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(nsw.n_total),
                      nsw.CodeStore().CodeSize(), "IndexNSWFlat codes");
  WRITE1(code_bytes);
  WRITEANDCHECK(nsw.CodeStore().Data(), code_bytes);
  const CsrGraph graph(nsw.Graph());
  WRITEVECTOR(graph.Offsets());
  WRITEVECTOR(graph.Edges());
}

void ValidateNSGFlatForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& nsg = static_cast<const IndexNSGFlat&>(index);
  HYPERVEC_THROW_IF_NOT_MSG(
      nsg.CodeStore().Size() == nsg.n_total &&
          nsg.Graph().NodeCount() == static_cast<size_t>(nsg.n_total),
      "IndexNSGFlat serialize: stored counts do not match n_total");
  HYPERVEC_THROW_IF_NOT_MSG(
      nsg.QuantizerModel().Dimension() == nsg.d &&
          nsg.QuantizerModel().Metric() == nsg.metric_type &&
          nsg.QuantizerModel().IsTrained() == nsg.is_trained,
      "IndexNSGFlat serialize: quantizer metadata does not match the index");
  const GraphValidationReport report =
      ValidateGraph(nsg.Graph(), nsg.EntryPoint());
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(nsg.n_total),
      "IndexNSGFlat serialize: graph is invalid or unreachable");
}

void WriteNSGFlatPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& nsg = static_cast<const IndexNSGFlat&>(index);
  write_index_header(nsg, f);
  const NSGIndexOptions& options = nsg.Options();
  WRITE1(options.knn_degree);
  WRITE1(options.nn_descent_iterations);
  WRITE1(options.nn_descent_convergence_threshold);
  WRITE1(options.random_seed);
  WRITE1(options.max_degree);
  WRITE1(options.build_search_width);
  WRITE1(options.candidate_pool_size);
  WRITE1(options.ef_search);
  const uint8_t check_relative_distance = options.check_relative_distance;
  WRITE1(check_relative_distance);
  const GraphId entry_point = nsg.EntryPoint();
  WRITE1(entry_point);
  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(nsg.n_total),
                      nsg.CodeStore().CodeSize(), "IndexNSGFlat codes");
  WRITE1(code_bytes);
  WRITEANDCHECK(nsg.CodeStore().Data(), code_bytes);
  const CsrGraph graph(nsg.Graph());
  WRITEVECTOR(graph.Offsets());
  WRITEVECTOR(graph.Edges());
}

void ValidateDiskANNFlatForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& diskann = static_cast<const IndexDiskANNFlat&>(index);
  const bool empty = diskann.n_total == 0;
  const DiskAnnNodeLayout* layout = diskann.Layout();
  const GraphStorage* graph = diskann.Graph();
  const RandomAccessReader* reader = diskann.NodeReader();
  HYPERVEC_THROW_IF_NOT_MSG(
      diskann.n_total >= 0 && diskann.is_trained &&
          diskann.metric_type == kMetricL2 &&
          diskann.QuantizerModel().TypeName() == "flat" &&
          diskann.QuantizerModel().Dimension() == diskann.d &&
          diskann.QuantizerModel().Metric() == diskann.metric_type &&
          diskann.CodeStore().CodeSize() ==
              diskann.QuantizerModel().CodeSize() &&
          diskann.CodeStore().Size() == diskann.n_total,
      "IndexDiskANNFlat serialize: index metadata is inconsistent");
  HYPERVEC_THROW_IF_NOT_MSG(
      empty ? layout == nullptr && graph == nullptr && reader == nullptr &&
                  diskann.EntryPoint() == kInvalidGraphId
            : layout != nullptr && graph != nullptr && reader != nullptr &&
                  layout->NodeCount() == static_cast<size_t>(diskann.n_total) &&
                  layout->Dimension() == static_cast<size_t>(diskann.d) &&
                  layout->MaxDegree() == diskann.Options().max_degree &&
                  layout->PageSize() == diskann.Options().page_size &&
                  graph->NodeCount() == layout->NodeCount() &&
                  reader->Size() == layout->StorageSize(),
      "IndexDiskANNFlat serialize: paged storage is inconsistent");
  if (!empty) {
    const GraphValidationReport report =
        ValidateGraph(*graph, diskann.EntryPoint());
    HYPERVEC_THROW_IF_NOT_MSG(
        report.IsStructurallyValid() &&
            report.reachable_nodes == static_cast<size_t>(diskann.n_total),
        "IndexDiskANNFlat serialize: graph is invalid or unreachable");
  }
}

void WriteDiskANNFlatPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& diskann = static_cast<const IndexDiskANNFlat&>(index);
  write_index_header(diskann, f);
  const DiskAnnIndexOptions& options = diskann.Options();
  WRITE1(options.max_degree);
  WRITE1(options.build_search_width);
  WRITE1(options.candidate_pool_size);
  WRITE1(options.alpha);
  WRITE1(options.build_passes);
  WRITE1(options.random_seed);
  WRITE1(options.search_width);
  const uint8_t check_relative_distance = options.check_relative_distance;
  WRITE1(check_relative_distance);
  WRITE1(options.page_size);
  WRITE1(options.cache_capacity_pages);
  const GraphId entry_point = diskann.EntryPoint();
  WRITE1(entry_point);

  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(diskann.n_total),
                      diskann.CodeStore().CodeSize(), "IndexDiskANNFlat codes");
  WRITE1(code_bytes);
  WRITEANDCHECK(diskann.CodeStore().Data(), code_bytes);
  if (diskann.n_total == 0) {
    const size_t node_bytes = 0;
    WRITE1(node_bytes);
  } else {
    write_random_access_payload(*diskann.NodeReader(), f);
  }
}

namespace {

void ValidateHNSWGraphForWrite(const IndexHNSW& index) {
  HYPERVEC_THROW_IF_NOT_MSG(
      index.n_total >= 0 &&
          index.n_total <= std::numeric_limits<HNSW::storage_idx_t>::max(),
      "IndexHNSW serialize: n_total exceeds graph ID capacity");
  const HNSW& hnsw = index.hnsw;
  const size_t total = static_cast<size_t>(index.n_total);
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.ef_construction > 0 && hnsw.cum_nneighbor_per_level.size() >= 2 &&
          hnsw.cum_nneighbor_per_level.front() == 0,
      "IndexHNSW serialize: graph configuration is invalid");
  for (size_t level = 1; level < hnsw.cum_nneighbor_per_level.size(); ++level) {
    HYPERVEC_THROW_IF_NOT_MSG(
        hnsw.cum_nneighbor_per_level[level] >
            hnsw.cum_nneighbor_per_level[level - 1],
        "IndexHNSW serialize: neighbor capacities must increase");
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.levels.size() == total && hnsw.offsets.size() == total + 1 &&
          !hnsw.offsets.empty() && hnsw.offsets.front() == 0,
      "IndexHNSW serialize: graph arrays do not match n_total");

  if (total == 0) {
    HYPERVEC_THROW_IF_NOT_MSG(
        hnsw.max_level == -1 && hnsw.entry_point == -1 &&
            hnsw.offsets.back() == 0 && hnsw.neighbors.size() == 0,
        "IndexHNSW serialize: empty graph metadata is invalid");
    return;
  }

  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.max_level >= 0 && hnsw.entry_point >= 0 &&
          static_cast<size_t>(hnsw.entry_point) < total,
      "IndexHNSW serialize: graph entry point is invalid");
  int observed_levels = 0;
  for (size_t node = 0; node < total; ++node) {
    const int node_levels = hnsw.levels[node];
    HYPERVEC_THROW_IF_NOT_MSG(
        node_levels > 0 && static_cast<size_t>(node_levels) <
                               hnsw.cum_nneighbor_per_level.size(),
        "IndexHNSW serialize: node level is invalid");
    observed_levels = std::max(observed_levels, node_levels);
    const size_t expected_span = static_cast<size_t>(
        hnsw.cum_nneighbor_per_level[static_cast<size_t>(node_levels)]);
    HYPERVEC_THROW_IF_NOT_MSG(
        hnsw.offsets[node + 1] >= hnsw.offsets[node] &&
            hnsw.offsets[node + 1] - hnsw.offsets[node] == expected_span,
        "IndexHNSW serialize: node offsets do not match its levels");
  }
  HYPERVEC_THROW_IF_NOT_MSG(
      hnsw.max_level == observed_levels - 1 &&
          hnsw.levels[static_cast<size_t>(hnsw.entry_point)] ==
              observed_levels &&
          hnsw.offsets.back() == hnsw.neighbors.size(),
      "IndexHNSW serialize: graph metadata is inconsistent");

  for (int level = 0; level <= hnsw.max_level; ++level) {
    const HNSWGraphStorage graph(hnsw, level);
    for (size_t node = 0; node < total; ++node) {
      if (hnsw.levels[node] > level) {
        (void)graph.Neighbors(static_cast<GraphId>(node));
      }
    }
  }
}

void ValidateHNSWStorageForWrite(const IndexHNSW& index) {
  HYPERVEC_THROW_IF_NOT_MSG(index.storage != nullptr,
                            "IndexHNSW serialize: storage is missing");
  HYPERVEC_THROW_IF_NOT_MSG(
      index.storage->d == index.d && index.storage->n_total == index.n_total &&
          index.storage->metric_type == index.metric_type &&
          index.storage->metric_arg == index.metric_arg &&
          index.storage->is_trained == index.is_trained,
      "IndexHNSW serialize: storage metadata does not match the graph");
  ValidateHNSWGraphForWrite(index);
}

void WriteHNSWPayload(const IndexHNSW& index, IOWriter* f) {
  write_index_header(index, f);
  write_HNSW(index.hnsw, f);
  WriteIndex(index.storage, f, 0);
}

}  // namespace

void ValidateHNSWFlatForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& hnsw = static_cast<const IndexHNSWFlat&>(index);
  ValidateHNSWStorageForWrite(hnsw);
  HYPERVEC_THROW_IF_NOT_MSG(
      typeid(*hnsw.storage) == typeid(IndexFlatL2) ||
          typeid(*hnsw.storage) == typeid(IndexFlatIP),
      "IndexHNSWFlat serialize: inner storage has no registered flat codec");
}

void WriteHNSWFlatPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  WriteHNSWPayload(static_cast<const IndexHNSWFlat&>(index), f);
}

void ValidateHNSWPQForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& hnsw = static_cast<const IndexHNSWPQ&>(index);
  ValidateHNSWStorageForWrite(hnsw);
  HYPERVEC_THROW_IF_NOT_MSG(
      typeid(*hnsw.storage) == typeid(IndexPQ),
      "IndexHNSWPQ serialize: inner storage is not an IndexPQ");
}

void WriteHNSWPQPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  WriteHNSWPayload(static_cast<const IndexHNSWPQ&>(index), f);
}

void ValidateHNSWLVQForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& hnsw = static_cast<const IndexHNSWLVQ&>(index);
  ValidateHNSWStorageForWrite(hnsw);
  HYPERVEC_THROW_IF_NOT_MSG(
      typeid(*hnsw.storage) == typeid(IndexLVQ),
      "IndexHNSWLVQ serialize: inner storage is not an IndexLVQ");
}

void WriteHNSWLVQPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  WriteHNSWPayload(static_cast<const IndexHNSWLVQ&>(index), f);
}

void ValidateVamanaFlatForWrite(const Index& index, int io_flags) {
  (void)io_flags;
  const auto& vamana = static_cast<const IndexVamanaFlat&>(index);
  HYPERVEC_THROW_IF_NOT_MSG(
      vamana.n_total >= 0 && vamana.is_trained &&
          vamana.metric_type == kMetricL2 &&
          vamana.QuantizerModel().TypeName() == "flat" &&
          vamana.QuantizerModel().Dimension() == vamana.d &&
          vamana.QuantizerModel().Metric() == vamana.metric_type &&
          vamana.CodeStore().CodeSize() == vamana.QuantizerModel().CodeSize() &&
          vamana.CodeStore().Size() == vamana.n_total &&
          vamana.Graph().NodeCount() == static_cast<size_t>(vamana.n_total) &&
          vamana.Graph().MaxDegree() == vamana.Options().max_degree,
      "IndexVamanaFlat serialize: index metadata is inconsistent");
  const GraphValidationReport report =
      ValidateGraph(vamana.Graph(), vamana.EntryPoint());
  HYPERVEC_THROW_IF_NOT_MSG(
      report.IsStructurallyValid() &&
          report.reachable_nodes == static_cast<size_t>(vamana.n_total),
      "IndexVamanaFlat serialize: graph is invalid or unreachable");
}

void WriteVamanaFlatPayload(const Index& index, IOWriter* f, int io_flags) {
  (void)io_flags;
  const auto& vamana = static_cast<const IndexVamanaFlat&>(index);
  write_index_header(vamana, f);
  const VamanaIndexOptions& options = vamana.Options();
  WRITE1(options.max_degree);
  WRITE1(options.build_search_width);
  WRITE1(options.candidate_pool_size);
  WRITE1(options.alpha);
  WRITE1(options.build_passes);
  WRITE1(options.random_seed);
  WRITE1(options.search_width);
  const uint8_t check_relative_distance = options.check_relative_distance;
  WRITE1(check_relative_distance);
  const GraphId entry_point = vamana.EntryPoint();
  WRITE1(entry_point);
  const size_t code_bytes =
      mul_no_overflow(static_cast<size_t>(vamana.n_total),
                      vamana.CodeStore().CodeSize(), "IndexVamanaFlat codes");
  WRITE1(code_bytes);
  WRITEANDCHECK(vamana.CodeStore().Data(), code_bytes);
  const CsrGraph graph(vamana.Graph());
  WRITEVECTOR(graph.Offsets());
  WRITEVECTOR(graph.Edges());
}

}  // namespace persistence_internal

void WriteIndex(const Index* index, IOWriter* f, int io_flags) {
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "WriteIndex: index must not be null");
  IndexIORegistry& registry = GetIndexIORegistry();
  if (registry.Contains(std::type_index(typeid(*index)))) {
    registry.Write(*index, f, io_flags);
    return;
  }

  HYPERVEC_THROW_MSG("unsupported index type for writing");
}

void write_ProductQuantizer(const ProductQuantizer* pq, IOWriter* f) {
  uint32_t h = fourcc("PqPq");
  WRITE1(h);
  write_pq(*pq, f);
}

void write_ProductQuantizer(const ProductQuantizer* pq, const char* fname) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  write_ProductQuantizer(pq, f.get());
}

void write_LocalVectorQuantizer(const LocalVectorQuantizer* lvq, IOWriter* f) {
  uint32_t h = fourcc("LvQq");
  WRITE1(h);
  write_lvq(*lvq, f);
}

void write_LocalVectorQuantizer(const LocalVectorQuantizer* lvq,
                                const char* fname) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  write_LocalVectorQuantizer(lvq, f.get());
}

void WriteIndex(const Index* index, FILE* f, int io_flags) {
  FileIOWriter writer(f);
  WriteIndex(index, &writer, io_flags);
}

void WriteIndex(const Index* index, const char* fname, int io_flags) {
  std::unique_ptr<IOWriter> f(new FileIOWriter(fname));
  WriteIndex(index, f.get(), io_flags);
}

}  // namespace hypervec
