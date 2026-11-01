/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <eval/vector_dataset.h>
#include <gtest/gtest.h>
#include <utils/log/exception.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

class TemporaryDatasetFile {
 public:
  explicit TemporaryDatasetFile(std::string extension) {
    static std::atomic<uint64_t> sequence = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("hypervec-dataset-" + std::to_string(sequence.fetch_add(1)) +
             std::move(extension));
  }

  ~TemporaryDatasetFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  TemporaryDatasetFile(const TemporaryDatasetFile&) = delete;
  TemporaryDatasetFile& operator=(const TemporaryDatasetFile&) = delete;

  const std::filesystem::path& Path() const { return path_; }

  void Write(const std::vector<uint8_t>& bytes) const {
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.is_open());
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output.good());
  }

 private:
  std::filesystem::path path_;
};

void AppendUint32(uint32_t value, std::vector<uint8_t>* bytes) {
  bytes->push_back(static_cast<uint8_t>(value));
  bytes->push_back(static_cast<uint8_t>(value >> 8U));
  bytes->push_back(static_cast<uint8_t>(value >> 16U));
  bytes->push_back(static_cast<uint8_t>(value >> 24U));
}

void AppendFvec(const std::vector<float>& values, std::vector<uint8_t>* bytes) {
  AppendUint32(static_cast<uint32_t>(values.size()), bytes);
  for (float value : values) {
    AppendUint32(std::bit_cast<uint32_t>(value), bytes);
  }
}

void AppendIvec(const std::vector<int32_t>& values,
                std::vector<uint8_t>* bytes) {
  AppendUint32(static_cast<uint32_t>(values.size()), bytes);
  for (int32_t value : values) {
    AppendUint32(std::bit_cast<uint32_t>(value), bytes);
  }
}

}  // namespace

TEST(VectorDataset, ReadsFvecsAndIvecsRows) {
  TemporaryDatasetFile fvecs(".fvecs");
  std::vector<uint8_t> float_bytes;
  AppendFvec({1.0F, -2.5F, 3.25F}, &float_bytes);
  AppendFvec({4.0F, 5.5F, -6.0F}, &float_bytes);
  fvecs.Write(float_bytes);

  const auto floats = hypervec::ReadFvecsFile(fvecs.Path().string());
  EXPECT_EQ(floats.vector_count, 2);
  EXPECT_EQ(floats.dimension, 3);
  EXPECT_EQ(floats.values,
            (std::vector<float>{1.0F, -2.5F, 3.25F, 4.0F, 5.5F, -6.0F}));

  TemporaryDatasetFile ivecs(".ivecs");
  std::vector<uint8_t> integer_bytes;
  AppendIvec({7, 11}, &integer_bytes);
  AppendIvec({13, -1}, &integer_bytes);
  ivecs.Write(integer_bytes);

  const auto integers = hypervec::ReadIvecsFile(ivecs.Path().string());
  EXPECT_EQ(integers.vector_count, 2);
  EXPECT_EQ(integers.dimension, 2);
  EXPECT_EQ(integers.values, (std::vector<hypervec::idx_t>{7, 11, 13, -1}));
}

TEST(VectorDataset, RejectsMissingEmptyAndTruncatedFiles) {
  EXPECT_THROW(hypervec::ReadFvecsFile("/path/that/does/not/exist.fvecs"),
               hypervec::HypervecException);

  TemporaryDatasetFile empty(".fvecs");
  empty.Write({});
  EXPECT_THROW(hypervec::ReadFvecsFile(empty.Path().string()),
               hypervec::HypervecException);

  TemporaryDatasetFile truncated_header(".fvecs");
  truncated_header.Write({2, 0});
  EXPECT_THROW(hypervec::ReadFvecsFile(truncated_header.Path().string()),
               hypervec::HypervecException);

  TemporaryDatasetFile truncated_payload(".fvecs");
  std::vector<uint8_t> bytes;
  AppendUint32(2, &bytes);
  AppendUint32(std::bit_cast<uint32_t>(1.0F), &bytes);
  truncated_payload.Write(bytes);
  EXPECT_THROW(hypervec::ReadFvecsFile(truncated_payload.Path().string()),
               hypervec::HypervecException);
}

TEST(VectorDataset, RejectsNonPositiveAndInconsistentDimensions) {
  TemporaryDatasetFile zero_dimension(".ivecs");
  std::vector<uint8_t> zero_bytes;
  AppendUint32(0, &zero_bytes);
  zero_dimension.Write(zero_bytes);
  EXPECT_THROW(hypervec::ReadIvecsFile(zero_dimension.Path().string()),
               hypervec::HypervecException);

  TemporaryDatasetFile inconsistent(".ivecs");
  std::vector<uint8_t> inconsistent_bytes;
  AppendIvec({1, 2}, &inconsistent_bytes);
  AppendIvec({3, 4, 5}, &inconsistent_bytes);
  inconsistent.Write(inconsistent_bytes);
  EXPECT_THROW(hypervec::ReadIvecsFile(inconsistent.Path().string()),
               hypervec::HypervecException);
}
