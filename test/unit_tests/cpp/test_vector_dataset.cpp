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

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
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

TEST(VectorDataset, WritesCompleteAndSelectedFvecRows) {
  const hypervec::FloatVectorDataset source{
      3, 2, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};

  TemporaryDatasetFile complete(".fvecs");
  hypervec::WriteFvecsFile(complete.Path().string(), source);
  const auto complete_result =
      hypervec::ReadFvecsFile(complete.Path().string());
  EXPECT_EQ(complete_result.vector_count, 3);
  EXPECT_EQ(complete_result.dimension, 2);
  EXPECT_EQ(complete_result.values, source.values);

  TemporaryDatasetFile selected(".fvecs");
  hypervec::WriteFvecsRowsFile(selected.Path().string(), source, {2, 0});
  const auto selected_result =
      hypervec::ReadFvecsFile(selected.Path().string());
  EXPECT_EQ(selected_result.vector_count, 2);
  EXPECT_EQ(selected_result.dimension, 2);
  EXPECT_EQ(selected_result.values,
            (std::vector<float>{5.0F, 6.0F, 1.0F, 2.0F}));

  EXPECT_THROW(
      hypervec::WriteFvecsRowsFile(selected.Path().string(), source, {0, 0}),
      hypervec::HypervecException);
  EXPECT_THROW(
      hypervec::WriteFvecsRowsFile(selected.Path().string(), source, {3}),
      hypervec::HypervecException);
  EXPECT_THROW(
      hypervec::WriteFvecsRowsFile(selected.Path().string(), source, {}),
      hypervec::HypervecException);
}

TEST(VectorDataset, ValidatesAndNormalizesFiniteNonzeroRowsTransactionally) {
  hypervec::FloatVectorDataset dataset{2, 2, {3.0F, 4.0F, 0.0F, -2.0F}};
  hypervec::NormalizeL2(&dataset);
  EXPECT_FLOAT_EQ(dataset.values[0], 0.6F);
  EXPECT_FLOAT_EQ(dataset.values[1], 0.8F);
  EXPECT_FLOAT_EQ(dataset.values[2], 0.0F);
  EXPECT_FLOAT_EQ(dataset.values[3], -1.0F);

  hypervec::FloatVectorDataset subnormal{
      1, 1, {(std::numeric_limits<float>::denorm_min)()}};
  hypervec::NormalizeL2(&subnormal);
  EXPECT_FLOAT_EQ(subnormal.values[0], 1.0F);

  hypervec::FloatVectorDataset zero_row{2, 2, {3.0F, 4.0F, 0.0F, 0.0F}};
  const std::vector<float> before = zero_row.values;
  EXPECT_THROW(hypervec::NormalizeL2(&zero_row), hypervec::HypervecException);
  EXPECT_EQ(zero_row.values, before);

  hypervec::FloatVectorDataset non_finite{
      1, 1, {(std::numeric_limits<float>::infinity)()}};
  EXPECT_THROW(hypervec::ValidateFloatVectorDataset(non_finite),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::NormalizeL2(nullptr), hypervec::HypervecException);
}

TEST(VectorDataset, CreatesDeterministicDisjointDatasetSplits) {
  const hypervec::DatasetRowSplit first =
      hypervec::MakeDatasetRowSplit(10, 3, 42);
  const hypervec::DatasetRowSplit repeated =
      hypervec::MakeDatasetRowSplit(10, 3, 42);
  const hypervec::DatasetRowSplit different =
      hypervec::MakeDatasetRowSplit(10, 3, 43);

  EXPECT_EQ(first.base_rows, repeated.base_rows);
  EXPECT_EQ(first.query_rows, repeated.query_rows);
  EXPECT_NE(first.query_rows, different.query_rows);
  EXPECT_TRUE(std::is_sorted(first.base_rows.begin(), first.base_rows.end()));
  EXPECT_TRUE(std::is_sorted(first.query_rows.begin(), first.query_rows.end()));

  std::unordered_set<hypervec::idx_t> rows(first.base_rows.begin(),
                                           first.base_rows.end());
  for (hypervec::idx_t row : first.query_rows) {
    EXPECT_TRUE(rows.insert(row).second);
  }
  EXPECT_EQ(rows.size(), 10U);
  for (hypervec::idx_t row = 0; row < 10; ++row) {
    EXPECT_TRUE(rows.contains(row));
  }

  EXPECT_THROW(hypervec::MakeDatasetRowSplit(1, 1, 42),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::MakeDatasetRowSplit(10, 0, 42),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::MakeDatasetRowSplit(10, 10, 42),
               hypervec::HypervecException);
}
