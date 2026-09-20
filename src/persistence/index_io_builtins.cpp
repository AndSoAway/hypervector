/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/flat/index_flat.h>
#include <persistence/index_io_builtins.h>
#include <persistence/index_io_registry.h>
#include <persistence/io.h>
#include <quantization/lvq/index_lvq.h>
#include <quantization/pq/index_pq.h>

#include <typeindex>

namespace hypervec {
namespace persistence_internal {

namespace {

class BuiltinIndexIOCodecs {
 public:
  BuiltinIndexIOCodecs() {
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
  }

  IndexIORegistry registry;
};

}  // namespace

IndexIORegistry& GetBuiltinIndexIORegistry() {
  static BuiltinIndexIOCodecs codecs;
  return codecs.registry;
}

}  // namespace persistence_internal
}  // namespace hypervec
