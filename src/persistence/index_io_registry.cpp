/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <persistence/index_io_registry.h>
#include <persistence/io.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hypervec {

struct IndexIORegistry::Impl {
  struct Entry {
    IndexIODescriptor descriptor;
    std::type_index index_type;
    IndexPayloadWriter writer;
    IndexPayloadReader reader;
  };

  mutable std::shared_mutex mutex;
  std::map<std::string, Entry> entries;
  std::unordered_map<std::type_index, std::string> types;
  std::unordered_map<uint32_t, std::string> tags;
};

IndexIORegistry::IndexIORegistry() : impl_(std::make_unique<Impl>()) {}

IndexIORegistry::~IndexIORegistry() = default;

void IndexIORegistry::Register(IndexIODescriptor descriptor,
                               std::type_index index_type,
                               IndexPayloadWriter writer,
                               IndexPayloadReader reader) {
  HYPERVEC_THROW_IF_NOT_MSG(!descriptor.name.empty(),
                            "registered index codec name must not be empty");
  HYPERVEC_THROW_IF_NOT_MSG(descriptor.write_tag != 0,
                            "registered index codec tag must not be zero");
  HYPERVEC_THROW_IF_NOT_MSG(static_cast<bool>(writer),
                            "registered index writer must not be empty");
  HYPERVEC_THROW_IF_NOT_MSG(static_cast<bool>(reader),
                            "registered index reader must not be empty");

  std::vector<uint32_t> normalized_tags = {descriptor.write_tag};
  normalized_tags.insert(normalized_tags.end(), descriptor.read_tags.begin(),
                         descriptor.read_tags.end());
  std::sort(normalized_tags.begin(), normalized_tags.end());
  normalized_tags.erase(
      std::unique(normalized_tags.begin(), normalized_tags.end()),
      normalized_tags.end());
  HYPERVEC_THROW_IF_NOT_MSG(
      std::none_of(normalized_tags.begin(), normalized_tags.end(),
                   [](uint32_t tag) { return tag == 0; }),
      "registered index read tags must not contain zero");
  descriptor.read_tags = normalized_tags;

  std::unique_lock lock(impl_->mutex);
  HYPERVEC_THROW_IF_NOT_FMT(
      impl_->entries.find(descriptor.name) == impl_->entries.end(),
      "index codec name '%s' is already registered", descriptor.name.c_str());
  HYPERVEC_THROW_IF_NOT_MSG(
      impl_->types.find(index_type) == impl_->types.end(),
      "C++ index type already has a registered persistence codec");
  for (uint32_t tag : descriptor.read_tags) {
    HYPERVEC_THROW_IF_NOT_FMT(
        impl_->tags.find(tag) == impl_->tags.end(),
        "index persistence tag '%s' is already registered",
        fourcc_inv_printable(tag).c_str());
  }

  const std::string name = descriptor.name;
  impl_->entries.emplace(
      name, Impl::Entry{std::move(descriptor), index_type, std::move(writer),
                        std::move(reader)});
  impl_->types.emplace(index_type, name);
  for (uint32_t tag : impl_->entries.at(name).descriptor.read_tags) {
    impl_->tags.emplace(tag, name);
  }
}

bool IndexIORegistry::Contains(std::type_index index_type) const {
  std::shared_lock lock(impl_->mutex);
  return impl_->types.find(index_type) != impl_->types.end();
}

bool IndexIORegistry::Contains(uint32_t read_tag) const {
  std::shared_lock lock(impl_->mutex);
  return impl_->tags.find(read_tag) != impl_->tags.end();
}

std::vector<IndexIODescriptor> IndexIORegistry::List() const {
  std::shared_lock lock(impl_->mutex);
  std::vector<IndexIODescriptor> descriptors;
  descriptors.reserve(impl_->entries.size());
  for (const auto& [name, entry] : impl_->entries) {
    (void)name;
    descriptors.push_back(entry.descriptor);
  }
  return descriptors;
}

void IndexIORegistry::Write(const Index& index, IOWriter* writer,
                            int io_flags) const {
  HYPERVEC_THROW_IF_NOT_MSG(writer != nullptr,
                            "index persistence writer must not be null");

  uint32_t tag = 0;
  IndexPayloadWriter payload_writer;
  {
    std::shared_lock lock(impl_->mutex);
    const auto type = impl_->types.find(std::type_index(typeid(index)));
    HYPERVEC_THROW_IF_NOT_MSG(
        type != impl_->types.end(),
        "no persistence codec is registered for this C++ index type");
    const Impl::Entry& entry = impl_->entries.at(type->second);
    tag = entry.descriptor.write_tag;
    payload_writer = entry.writer;
  }

  HYPERVEC_THROW_IF_NOT_FMT((*writer)(&tag, sizeof(tag), 1) == 1,
                            "write error in %s", writer->name.c_str());
  payload_writer(index, writer, io_flags);
}

std::unique_ptr<Index> IndexIORegistry::Read(IOReader* reader,
                                             int io_flags) const {
  HYPERVEC_THROW_IF_NOT_MSG(reader != nullptr,
                            "index persistence reader must not be null");
  uint32_t tag = 0;
  HYPERVEC_THROW_IF_NOT_FMT((*reader)(&tag, sizeof(tag), 1) == 1,
                            "read error in %s", reader->name.c_str());

  return ReadPayload(tag, reader, io_flags);
}

std::unique_ptr<Index> IndexIORegistry::ReadPayload(uint32_t read_tag,
                                                    IOReader* reader,
                                                    int io_flags) const {
  HYPERVEC_THROW_IF_NOT_MSG(reader != nullptr,
                            "index persistence reader must not be null");

  IndexPayloadReader payload_reader;
  std::type_index expected_type(typeid(void));
  {
    std::shared_lock lock(impl_->mutex);
    const auto registered_tag = impl_->tags.find(read_tag);
    HYPERVEC_THROW_IF_NOT_FMT(registered_tag != impl_->tags.end(),
                              "unknown index persistence tag '%s'",
                              fourcc_inv_printable(read_tag).c_str());
    const Impl::Entry& entry = impl_->entries.at(registered_tag->second);
    payload_reader = entry.reader;
    expected_type = entry.index_type;
  }

  std::unique_ptr<Index> index = payload_reader(reader, io_flags);
  HYPERVEC_THROW_IF_NOT_MSG(index != nullptr,
                            "registered index reader returned null");
  const Index* loaded_index = index.get();
  HYPERVEC_THROW_IF_NOT_MSG(
      std::type_index(typeid(*loaded_index)) == expected_type,
      "registered index reader returned the wrong C++ index type");
  return index;
}

}  // namespace hypervec
