/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/diskann/index_diskann.h>
#include <index/flat/index_flat.h>
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
#include <persistence/index_io_builtins.h>
#include <persistence/index_io_registry.h>
#include <persistence/io.h>
#include <quantization/lvq/index_ivflvq.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_ivfpq.h>
#include <quantization/pq/index_pq.h>
#include <quantization/rabitq/index_ivf_rabitq.h>

#include <typeindex>

namespace hypervec {
namespace persistence_internal {

namespace {

class BuiltinIndexIOCodecs {
 public:
  BuiltinIndexIOCodecs() {
    registry.Register({"flat", fourcc("IFlx"), {}},
                      std::type_index(typeid(IndexFlat)), WriteFlatPayload,
                      ReadFlatPayload);
    registry.Register({"flat_l2", fourcc("IFlm"), {fourcc("IFll")}},
                      std::type_index(typeid(IndexFlatL2)), WriteFlatL2Payload,
                      ReadFlatL2Payload);
    registry.Register({"flat_ip", fourcc("IFlp"), {}},
                      std::type_index(typeid(IndexFlatIP)), WriteFlatIPPayload,
                      ReadFlatIPPayload);
    registry.Register({"pq", fourcc("IPQ8"), {}},
                      std::type_index(typeid(IndexPQ)), WritePQPayload,
                      ReadPQPayload);
    registry.Register({"lvq", fourcc("ILVQ"), {}},
                      std::type_index(typeid(IndexLVQ)), WriteLVQPayload,
                      ReadLVQPayload);
    registry.Register({"ivf_flat", fourcc("IVFf"), {}},
                      std::type_index(typeid(IndexIVFFlat)),
                      WriteIVFFlatPayload, ReadIVFFlatPayload);
    registry.Register({"ivf_pq", fourcc("IVPQ"), {}},
                      std::type_index(typeid(IndexIVFPQ)), WriteIVFPQPayload,
                      ReadIVFPQPayload);
    registry.Register({"ivf_lvq", fourcc("IVLQ"), {}},
                      std::type_index(typeid(IndexIVFLVQ)), WriteIVFLVQPayload,
                      ReadIVFLVQPayload);
    registry.Register({"ivf_rabitq", fourcc("IVRQ"), {}},
                      std::type_index(typeid(IndexIVFRaBitQ)),
                      WriteIVFRaBitQPayload, ReadIVFRaBitQPayload,
                      ValidateIVFRaBitQForWrite);
    registry.Register({"id_map", fourcc("IxMp"), {}},
                      std::type_index(typeid(IndexIDMap)), WriteIDMapPayload,
                      ReadIDMapPayload, ValidateIDMapForWrite);
    registry.Register({"pre_transform", fourcc("IPTr"), {}},
                      std::type_index(typeid(IndexPreTransform)),
                      WritePreTransformPayload, ReadPreTransformPayload,
                      ValidatePreTransformForWrite);
    registry.Register({"lsh", fourcc("ILSh"), {}},
                      std::type_index(typeid(IndexLSH)), WriteLSHPayload,
                      ReadLSHPayload, ValidateLSHForWrite);
    registry.Register(
        {"nsw_flat", fourcc("INSf"), {}}, std::type_index(typeid(IndexNSWFlat)),
        WriteNSWFlatPayload, ReadNSWFlatPayload, ValidateNSWFlatForWrite);
    registry.Register(
        {"nsg_flat", fourcc("INGf"), {}}, std::type_index(typeid(IndexNSGFlat)),
        WriteNSGFlatPayload, ReadNSGFlatPayload, ValidateNSGFlatForWrite);
    registry.Register({"diskann_flat", fourcc("IDAf"), {}},
                      std::type_index(typeid(IndexDiskANNFlat)),
                      WriteDiskANNFlatPayload, ReadDiskANNFlatPayload,
                      ValidateDiskANNFlatForWrite);
    registry.Register({"hnsw_flat", fourcc("IHNf"), {}},
                      std::type_index(typeid(IndexHNSWFlat)),
                      WriteHNSWFlatPayload, ReadHNSWFlatPayload,
                      ValidateHNSWFlatForWrite);
    registry.Register({"hnsw_pq", fourcc("IHNp"), {}},
                      std::type_index(typeid(IndexHNSWPQ)), WriteHNSWPQPayload,
                      ReadHNSWPQPayload, ValidateHNSWPQForWrite);
    registry.Register(
        {"hnsw_lvq", fourcc("IHNl"), {}}, std::type_index(typeid(IndexHNSWLVQ)),
        WriteHNSWLVQPayload, ReadHNSWLVQPayload, ValidateHNSWLVQForWrite);
    registry.Register({"vamana_flat", fourcc("IVAf"), {}},
                      std::type_index(typeid(IndexVamanaFlat)),
                      WriteVamanaFlatPayload, ReadVamanaFlatPayload,
                      ValidateVamanaFlatForWrite);
  }

  IndexIORegistry registry;
};

}  // namespace

IndexIORegistry& GetBuiltinIndexIORegistry() {
  static BuiltinIndexIOCodecs codecs;
  return codecs.registry;
}

}  // namespace persistence_internal

IndexIORegistry& GetIndexIORegistry() {
  return persistence_internal::GetBuiltinIndexIORegistry();
}

}  // namespace hypervec
