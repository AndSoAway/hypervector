/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/vector_dataset.h>
#include <utils/log/assert.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace hypervec {

namespace {

uint32_t DecodeUint32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

void ReadExact(std::ifstream* input, uint8_t* output, size_t size,
               const std::string& filename) {
  HYPERVEC_THROW_IF_NOT_FMT(
      size <=
          static_cast<size_t>((std::numeric_limits<std::streamsize>::max)()),
      "evaluation dataset '%s' contains an oversized row", filename.c_str());
  input->read(reinterpret_cast<char*>(output),
              static_cast<std::streamsize>(size));
  HYPERVEC_THROW_IF_NOT_FMT(
      input->gcount() == static_cast<std::streamsize>(size),
      "evaluation dataset '%s' is truncated", filename.c_str());
}

uint64_t FileSize(std::ifstream* input, const std::string& filename) {
  input->seekg(0, std::ios::end);
  const std::streamoff end = input->tellg();
  HYPERVEC_THROW_IF_NOT_FMT(end >= 0,
                            "cannot determine evaluation dataset size: '%s'",
                            filename.c_str());
  input->seekg(0, std::ios::beg);
  return static_cast<uint64_t>(end);
}

template <typename Value, typename Decode>
VectorDataset<Value> ReadVectorFile(const std::string& filename,
                                    Decode decode) {
  HYPERVEC_THROW_IF_NOT_MSG(!filename.empty(),
                            "evaluation dataset filename must not be empty");
  std::ifstream input(filename, std::ios::binary);
  HYPERVEC_THROW_IF_NOT_FMT(
      input.is_open(), "cannot open evaluation dataset '%s'", filename.c_str());
  const uint64_t file_size = FileSize(&input, filename);
  HYPERVEC_THROW_IF_NOT_FMT(file_size > 0, "evaluation dataset '%s' is empty",
                            filename.c_str());

  VectorDataset<Value> dataset;
  uint64_t consumed = 0;
  std::vector<uint8_t> row_bytes;
  while (consumed < file_size) {
    HYPERVEC_THROW_IF_NOT_FMT(
        file_size - consumed >= sizeof(uint32_t),
        "evaluation dataset '%s' has a truncated dimension header",
        filename.c_str());
    uint8_t dimension_bytes[sizeof(uint32_t)];
    ReadExact(&input, dimension_bytes, sizeof(dimension_bytes), filename);
    consumed += sizeof(dimension_bytes);
    const int32_t dimension =
        std::bit_cast<int32_t>(DecodeUint32(dimension_bytes));
    HYPERVEC_THROW_IF_NOT_FMT(
        dimension > 0, "evaluation dataset '%s' has a non-positive dimension",
        filename.c_str());
    HYPERVEC_THROW_IF_NOT_FMT(
        dataset.dimension == 0 || dataset.dimension == dimension,
        "evaluation dataset '%s' contains inconsistent dimensions",
        filename.c_str());

    const uint64_t payload_size =
        static_cast<uint64_t>(dimension) * sizeof(uint32_t);
    HYPERVEC_THROW_IF_NOT_FMT(
        payload_size <= file_size - consumed,
        "evaluation dataset '%s' has a truncated vector payload",
        filename.c_str());
    HYPERVEC_THROW_IF_NOT_MSG(
        static_cast<uint64_t>(dimension) <=
            (std::numeric_limits<size_t>::max)() - dataset.values.size(),
        "evaluation dataset exceeds addressable memory");

    row_bytes.resize(static_cast<size_t>(payload_size));
    ReadExact(&input, row_bytes.data(), row_bytes.size(), filename);
    consumed += payload_size;
    const size_t previous_size = dataset.values.size();
    dataset.values.resize(previous_size + static_cast<size_t>(dimension));
    for (int32_t column = 0; column < dimension; ++column) {
      dataset.values[previous_size + static_cast<size_t>(column)] = decode(
          row_bytes.data() + static_cast<size_t>(column) * sizeof(uint32_t));
    }

    HYPERVEC_THROW_IF_NOT_MSG(
        dataset.vector_count < (std::numeric_limits<idx_t>::max)(),
        "evaluation dataset contains too many vectors");
    ++dataset.vector_count;
    dataset.dimension = dimension;
  }
  return dataset;
}

}  // namespace

FloatVectorDataset ReadFvecsFile(const std::string& filename) {
  return ReadVectorFile<float>(filename, [](const uint8_t* bytes) {
    return std::bit_cast<float>(DecodeUint32(bytes));
  });
}

IntegerVectorDataset ReadIvecsFile(const std::string& filename) {
  return ReadVectorFile<idx_t>(filename, [](const uint8_t* bytes) {
    return static_cast<idx_t>(std::bit_cast<int32_t>(DecodeUint32(bytes)));
  });
}

}  // namespace hypervec
