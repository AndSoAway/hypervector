/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/vector_dataset.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
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

void EncodeUint32(uint32_t value, uint8_t* bytes) {
  bytes[0] = static_cast<uint8_t>(value);
  bytes[1] = static_cast<uint8_t>(value >> 8U);
  bytes[2] = static_cast<uint8_t>(value >> 16U);
  bytes[3] = static_cast<uint8_t>(value >> 24U);
}

size_t DatasetElementCount(const FloatVectorDataset& dataset) {
  HYPERVEC_THROW_IF_NOT_MSG(dataset.vector_count > 0,
                            "evaluation dataset must contain vectors");
  HYPERVEC_THROW_IF_NOT_MSG(dataset.dimension > 0,
                            "evaluation dataset dimension must be positive");
  const uint64_t rows = static_cast<uint64_t>(dataset.vector_count);
  const uint64_t dimension = static_cast<uint64_t>(dataset.dimension);
  HYPERVEC_THROW_IF_NOT_MSG(
      rows <= (std::numeric_limits<size_t>::max)() / dimension,
      "evaluation dataset size exceeds addressable memory");
  return static_cast<size_t>(rows * dimension);
}

void WriteExact(std::ofstream* output, const uint8_t* data, size_t size,
                const std::string& filename) {
  HYPERVEC_THROW_IF_NOT_MSG(
      size <=
          static_cast<size_t>((std::numeric_limits<std::streamsize>::max)()),
      "evaluation dataset output row is too large");
  output->write(reinterpret_cast<const char*>(data),
                static_cast<std::streamsize>(size));
  HYPERVEC_THROW_IF_NOT_FMT(
      output->good(), "cannot write evaluation dataset '%s'", filename.c_str());
}

void WriteFvecsRows(const std::string& filename,
                    const FloatVectorDataset& dataset,
                    const std::vector<idx_t>* rows) {
  ValidateFloatVectorDataset(dataset);
  HYPERVEC_THROW_IF_NOT_MSG(!filename.empty(),
                            "evaluation dataset filename must not be empty");
  const size_t output_rows = rows == nullptr
                                 ? static_cast<size_t>(dataset.vector_count)
                                 : rows->size();
  HYPERVEC_THROW_IF_NOT_MSG(output_rows > 0,
                            "evaluation dataset output must contain rows");

  std::vector<uint8_t> seen;
  if (rows != nullptr) {
    seen.resize(static_cast<size_t>(dataset.vector_count));
    for (idx_t row : *rows) {
      HYPERVEC_THROW_IF_NOT_MSG(
          row >= 0 && row < dataset.vector_count,
          "evaluation dataset output row is out of range");
      HYPERVEC_THROW_IF_NOT_MSG(
          seen[static_cast<size_t>(row)] == 0,
          "evaluation dataset output rows must be unique");
      seen[static_cast<size_t>(row)] = 1;
    }
  }

  const size_t dimension = static_cast<size_t>(dataset.dimension);
  HYPERVEC_THROW_IF_NOT_MSG(
      dimension <= static_cast<size_t>((std::numeric_limits<int32_t>::max)()) &&
          dimension <=
              (std::numeric_limits<size_t>::max)() / sizeof(uint32_t) - 1,
      "evaluation dataset output row is too large");
  const size_t row_size = sizeof(uint32_t) * (dimension + 1);
  std::vector<uint8_t> row_bytes(row_size);
  EncodeUint32(static_cast<uint32_t>(dataset.dimension), row_bytes.data());

  std::ofstream output(filename, std::ios::binary | std::ios::trunc);
  HYPERVEC_THROW_IF_NOT_FMT(output.is_open(),
                            "cannot open evaluation dataset output '%s'",
                            filename.c_str());
  for (size_t output_row = 0; output_row < output_rows; ++output_row) {
    const size_t source_row =
        rows == nullptr ? output_row : static_cast<size_t>((*rows)[output_row]);
    const size_t offset = source_row * dimension;
    for (size_t column = 0; column < dimension; ++column) {
      EncodeUint32(std::bit_cast<uint32_t>(dataset.values[offset + column]),
                   row_bytes.data() + sizeof(uint32_t) * (column + 1));
    }
    WriteExact(&output, row_bytes.data(), row_bytes.size(), filename);
  }
  output.close();
  HYPERVEC_THROW_IF_NOT_FMT(!output.fail(),
                            "cannot finalize evaluation dataset '%s'",
                            filename.c_str());
}

uint64_t BoundedRandom(std::mt19937_64* random, uint64_t bound) {
  const uint64_t threshold = static_cast<uint64_t>(-bound) % bound;
  uint64_t value;
  do {
    value = (*random)();
  } while (value < threshold);
  return value % bound;
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

void ValidateFloatVectorDataset(const FloatVectorDataset& dataset) {
  const size_t expected_size = DatasetElementCount(dataset);
  HYPERVEC_THROW_IF_NOT_MSG(dataset.values.size() == expected_size,
                            "evaluation dataset values do not match its shape");
  for (size_t offset = 0; offset < dataset.values.size(); ++offset) {
    HYPERVEC_THROW_IF_NOT_FMT(
        std::isfinite(dataset.values[offset]),
        "evaluation dataset contains a non-finite value at offset %zu", offset);
  }
}

void NormalizeL2(FloatVectorDataset* dataset) {
  HYPERVEC_THROW_IF_NOT_MSG(dataset != nullptr,
                            "evaluation dataset must not be null");
  ValidateFloatVectorDataset(*dataset);
  const size_t dimension = static_cast<size_t>(dataset->dimension);
  std::vector<double> inverse_norms(static_cast<size_t>(dataset->vector_count));
  for (idx_t row = 0; row < dataset->vector_count; ++row) {
    const size_t offset = static_cast<size_t>(row) * dimension;
    double squared_norm = 0.0;
    for (size_t column = 0; column < dimension; ++column) {
      const double value = dataset->values[offset + column];
      squared_norm += value * value;
    }
    HYPERVEC_THROW_IF_NOT_FMT(squared_norm > 0.0 && std::isfinite(squared_norm),
                              "evaluation dataset row %" PRId64
                              " has an invalid L2 norm",
                              static_cast<int64_t>(row));
    inverse_norms[static_cast<size_t>(row)] = 1.0 / std::sqrt(squared_norm);
  }
  for (idx_t row = 0; row < dataset->vector_count; ++row) {
    const size_t offset = static_cast<size_t>(row) * dimension;
    const double inverse_norm = inverse_norms[static_cast<size_t>(row)];
    for (size_t column = 0; column < dimension; ++column) {
      dataset->values[offset + column] = static_cast<float>(
          static_cast<double>(dataset->values[offset + column]) * inverse_norm);
    }
  }
}

DatasetRowSplit MakeDatasetRowSplit(idx_t vector_count, idx_t query_count,
                                    uint64_t seed) {
  HYPERVEC_THROW_IF_NOT_MSG(vector_count > 1,
                            "dataset split requires at least two vectors");
  HYPERVEC_THROW_IF_NOT_MSG(
      query_count > 0 && query_count < vector_count,
      "dataset query count must be positive and smaller than the vector count");
  HYPERVEC_THROW_IF_NOT_MSG(static_cast<uint64_t>(vector_count) <=
                                (std::numeric_limits<size_t>::max)(),
                            "dataset split size exceeds addressable memory");

  const size_t row_count = static_cast<size_t>(vector_count);
  const size_t selected_count = static_cast<size_t>(query_count);
  std::vector<idx_t> candidates(row_count);
  std::iota(candidates.begin(), candidates.end(), idx_t{0});
  std::mt19937_64 random(seed);
  for (size_t position = 0; position < selected_count; ++position) {
    const uint64_t remaining = static_cast<uint64_t>(row_count - position);
    const size_t selected =
        position + static_cast<size_t>(BoundedRandom(&random, remaining));
    std::swap(candidates[position], candidates[selected]);
  }

  DatasetRowSplit split;
  split.query_rows.assign(candidates.begin(),
                          candidates.begin() + selected_count);
  std::sort(split.query_rows.begin(), split.query_rows.end());
  std::vector<uint8_t> is_query(row_count);
  for (idx_t row : split.query_rows) {
    is_query[static_cast<size_t>(row)] = 1;
  }
  split.base_rows.reserve(row_count - selected_count);
  for (size_t row = 0; row < row_count; ++row) {
    if (is_query[row] == 0) {
      split.base_rows.push_back(static_cast<idx_t>(row));
    }
  }
  return split;
}

void WriteFvecsFile(const std::string& filename,
                    const FloatVectorDataset& dataset) {
  WriteFvecsRows(filename, dataset, nullptr);
}

void WriteFvecsRowsFile(const std::string& filename,
                        const FloatVectorDataset& dataset,
                        const std::vector<idx_t>& rows) {
  WriteFvecsRows(filename, dataset, &rows);
}

void WriteIvecsFile(const std::string& filename,
                    const IntegerVectorDataset& dataset) {
  HYPERVEC_THROW_IF_NOT_MSG(!filename.empty(),
                            "evaluation dataset filename must not be empty");
  HYPERVEC_THROW_IF_NOT_MSG(dataset.vector_count > 0,
                            "evaluation dataset must contain vectors");
  HYPERVEC_THROW_IF_NOT_MSG(dataset.dimension > 0,
                            "evaluation dataset dimension must be positive");
  const uint64_t rows = static_cast<uint64_t>(dataset.vector_count);
  const uint64_t dimension = static_cast<uint64_t>(dataset.dimension);
  HYPERVEC_THROW_IF_NOT_MSG(
      rows <= (std::numeric_limits<size_t>::max)() / dimension,
      "evaluation dataset size exceeds addressable memory");
  const size_t value_count = static_cast<size_t>(rows * dimension);
  HYPERVEC_THROW_IF_NOT_MSG(dataset.values.size() == value_count,
                            "evaluation dataset values do not match its shape");
  for (idx_t value : dataset.values) {
    HYPERVEC_THROW_IF_NOT_MSG(
        value >= (std::numeric_limits<int32_t>::min)() &&
            value <= (std::numeric_limits<int32_t>::max)(),
        "ivecs value exceeds the signed 32-bit format range");
  }

  const size_t dimension_size = static_cast<size_t>(dataset.dimension);
  HYPERVEC_THROW_IF_NOT_MSG(
      dimension_size <=
          (std::numeric_limits<size_t>::max)() / sizeof(uint32_t) - 1,
      "evaluation dataset output row is too large");
  const size_t row_size = sizeof(uint32_t) * (dimension_size + 1);
  std::vector<uint8_t> row_bytes(row_size);
  EncodeUint32(static_cast<uint32_t>(dataset.dimension), row_bytes.data());
  std::ofstream output(filename, std::ios::binary | std::ios::trunc);
  HYPERVEC_THROW_IF_NOT_FMT(output.is_open(),
                            "cannot open evaluation dataset output '%s'",
                            filename.c_str());
  for (idx_t row = 0; row < dataset.vector_count; ++row) {
    const size_t offset =
        static_cast<size_t>(row) * static_cast<size_t>(dataset.dimension);
    for (int32_t column = 0; column < dataset.dimension; ++column) {
      const int32_t value =
          static_cast<int32_t>(dataset.values[offset + column]);
      EncodeUint32(std::bit_cast<uint32_t>(value),
                   row_bytes.data() +
                       sizeof(uint32_t) * (static_cast<size_t>(column) + 1));
    }
    WriteExact(&output, row_bytes.data(), row_bytes.size(), filename);
  }
  output.close();
  HYPERVEC_THROW_IF_NOT_FMT(!output.fail(),
                            "cannot finalize evaluation dataset '%s'",
                            filename.c_str());
}

}  // namespace hypervec
