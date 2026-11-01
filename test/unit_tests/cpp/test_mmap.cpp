/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>
#include <index/flat/index_flat.h>
#include <index/ivf/index_ivf_flat.h>
#include <invlists/inverted_lists.h>
#include <persistence/index_io.h>
#include <persistence/io.h>
#include <persistence/mapped_io.h>
#include <utils/log/assert.h>
#include <utils/log/exception.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17): the project requires C++20.
#include <fstream>
#include <limits>
#include <list>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

class TempFile {
 public:
  explicit TempFile(const std::vector<uint8_t>& data) : size_(data.size()) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t stamp = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    path_ = std::filesystem::temp_directory_path() /
            ("hypervec-mmap-" + std::to_string(stamp) + "-" +
             std::to_string(sequence.fetch_add(1)));
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
    HYPERVEC_THROW_IF_NOT_MSG(output.good(),
                              "TempFile: failed to write test data");
  }

  ~TempFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  void Overwrite(const std::vector<uint8_t>& data) const {
    HYPERVEC_THROW_IF_NOT_MSG(
        data.size() == size_,
        "TempFile: replacement must preserve the mapped file size");
    std::fstream output(path_, std::ios::binary | std::ios::in | std::ios::out);
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
    output.flush();
    HYPERVEC_THROW_IF_NOT_MSG(output.good(),
                              "TempFile: failed to replace test data");
  }

  const std::filesystem::path& Path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
  size_t size_ = 0;
};

std::vector<float> MakeData(size_t count, size_t dimension, uint64_t seed) {
  std::vector<float> data(count * dimension);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> distribution;
  for (float& value : data) {
    value = distribution(rng);
  }
  return data;
}

void ExpectSameSearch(const hypervec::Index& expected,
                      const hypervec::Index& actual,
                      const std::vector<float>& queries, hypervec::idx_t k) {
  const hypervec::idx_t count =
      static_cast<hypervec::idx_t>(queries.size()) / expected.d;
  std::vector<float> expected_distances(static_cast<size_t>(count * k));
  std::vector<float> actual_distances(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> expected_labels(static_cast<size_t>(count * k));
  std::vector<hypervec::idx_t> actual_labels(static_cast<size_t>(count * k));
  expected.Search(count, queries.data(), k, expected_distances.data(),
                  expected_labels.data());
  actual.Search(count, queries.data(), k, actual_distances.data(),
                actual_labels.data());
  EXPECT_EQ(actual_labels, expected_labels);
  EXPECT_EQ(actual_distances, expected_distances);
}

}  // namespace

TEST(TestMmap, FlatCodesRemainMappedAfterReaderDestruction) {
#if !defined(__linux__) && !defined(__FreeBSD__) && !defined(_WIN32)
  GTEST_SKIP() << "Memory-mapped persistence is not supported on this OS.";
#endif
  constexpr size_t kDatabaseCount = 100;
  constexpr size_t kQueryCount = 10;
  constexpr size_t kDimension = 16;
  constexpr hypervec::idx_t kNeighbors = 10;
  const std::vector<float> first_data =
      MakeData(kDatabaseCount, kDimension, 123);
  const std::vector<float> second_data =
      MakeData(kDatabaseCount, kDimension, 456);
  const std::vector<float> queries = MakeData(kQueryCount, kDimension, 789);

  hypervec::IndexFlatL2 first(kDimension);
  hypervec::IndexFlatL2 second(kDimension);
  first.Add(kDatabaseCount, first_data.data());
  second.Add(kDatabaseCount, second_data.data());

  hypervec::VectorIOWriter first_writer;
  hypervec::VectorIOWriter second_writer;
  hypervec::WriteIndex(&first, &first_writer);
  hypervec::WriteIndex(&second, &second_writer);
  ASSERT_EQ(first_writer.data.size(), second_writer.data.size());

  const TempFile file(first_writer.data);
  const std::string path = file.Path().string();
  std::unique_ptr<hypervec::Index> mapped =
      hypervec::ReadIndexUp(path.c_str(), hypervec::IO_FLAG_MMAP_IFC);
  auto* mapped_flat = dynamic_cast<hypervec::IndexFlatL2*>(mapped.get());
  ASSERT_NE(mapped_flat, nullptr);
  EXPECT_FALSE(mapped_flat->codes.is_owned);
  EXPECT_NE(mapped_flat->codes.owner, nullptr);
  EXPECT_FALSE(mapped_flat->IsDataAligned());
  EXPECT_THROW(mapped_flat->GetXb(), hypervec::HypervecException);
  ExpectSameSearch(first, *mapped, queries, kNeighbors);

  const std::array<hypervec::idx_t, 2> subset_labels = {0, -1};
  std::array<float, 2> subset_distances{};
  mapped_flat->ComputeDistanceSubset(
      1, queries.data(), 2, subset_distances.data(), subset_labels.data());
  EXPECT_TRUE(std::isfinite(subset_distances[0]));
  EXPECT_TRUE(std::isinf(subset_distances[1]));
  mapped_flat->SyncL2Norms();
  EXPECT_EQ(mapped_flat->cached_l2norms.size(), kDatabaseCount);

  file.Overwrite(second_writer.data);
  ExpectSameSearch(second, *mapped, queries, kNeighbors);

  file.Overwrite(first_writer.data);
  ExpectSameSearch(first, *mapped, queries, kNeighbors);
}

TEST(TestMmap, FileHandleOverloadPreservesCurrentPosition) {
#if !defined(__linux__) && !defined(__FreeBSD__) && !defined(_WIN32)
  GTEST_SKIP() << "Memory-mapped persistence is not supported on this OS.";
#endif
  hypervec::IndexFlatL2 source(2);
  const std::vector<float> vectors = {0.0F, 0.0F, 2.0F, 2.0F};
  source.Add(2, vectors.data());
  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);

  constexpr int kPrefixSize = 7;
  std::vector<uint8_t> prefixed(kPrefixSize, 0xA5U);
  prefixed.insert(prefixed.end(), writer.data.begin(), writer.data.end());
  const TempFile file(prefixed);
  const std::string path = file.Path().string();
  FILE* handle = std::fopen(path.c_str(), "rb");
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(std::fseek(handle, kPrefixSize, SEEK_SET), 0);
  std::unique_ptr<hypervec::Index> mapped =
      hypervec::ReadIndexUp(handle, hypervec::IO_FLAG_MMAP_IFC);
  ASSERT_EQ(std::fclose(handle), 0);

  auto* mapped_flat = dynamic_cast<hypervec::IndexFlatL2*>(mapped.get());
  ASSERT_NE(mapped_flat, nullptr);
  EXPECT_FALSE(mapped_flat->codes.is_owned);
  const std::vector<float> query = {2.1F, 1.9F};
  float distance = 0.0F;
  hypervec::idx_t label = -1;
  mapped->Search(1, query.data(), 1, &distance, &label);
  EXPECT_EQ(label, 1);
}

TEST(TestMmap, InvertedListCodesUseMappedViews) {
#if !defined(__linux__) && !defined(__FreeBSD__) && !defined(_WIN32)
  GTEST_SKIP() << "Memory-mapped persistence is not supported on this OS.";
#endif
  constexpr size_t kDimension = 2;
  const std::vector<float> vectors = {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F,
                                      1.0F, 1.0F, 8.0F, 8.0F, 8.0F, 9.0F,
                                      9.0F, 8.0F, 9.0F, 9.0F};
  hypervec::IndexIVFFlat source(kDimension, 2, hypervec::kMetricL2);
  source.Train(8, vectors.data());
  source.Add(8, vectors.data());
  source.nprobe = 2;

  hypervec::VectorIOWriter writer;
  hypervec::WriteIndex(&source, &writer);
  const TempFile file(writer.data);
  const std::string path = file.Path().string();
  std::unique_ptr<hypervec::Index> mapped =
      hypervec::ReadIndexUp(path.c_str(), hypervec::IO_FLAG_MMAP_IFC);
  auto* mapped_ivf = dynamic_cast<hypervec::IndexIVFFlat*>(mapped.get());
  ASSERT_NE(mapped_ivf, nullptr);
  auto* lists =
      dynamic_cast<hypervec::ArrayInvertedLists*>(mapped_ivf->invlists);
  ASSERT_NE(lists, nullptr);
  bool found_mapped_codes = false;
  for (size_t list = 0; list < lists->nlist; ++list) {
    if (lists->list_size(list) > 0) {
      found_mapped_codes = found_mapped_codes || !lists->codes[list].is_owned;
    }
  }
  EXPECT_TRUE(found_mapped_codes);

  const std::vector<float> queries = {0.2F, 0.1F, 8.8F, 8.9F};
  ExpectSameSearch(source, *mapped, queries, 3);
}

TEST(TestMmap, ReaderRejectsTruncationWithoutOverflowingPosition) {
#if !defined(__linux__) && !defined(__FreeBSD__) && !defined(_WIN32)
  GTEST_SKIP() << "Memory-mapped persistence is not supported on this OS.";
#endif
  const TempFile file({0, 1, 2, 3, 4, 5, 6, 7});
  auto owner =
      std::make_shared<hypervec::MmappedFileMappingOwner>(file.Path().string());
  hypervec::MappedFileIOReader reader(owner);
  std::array<uint32_t, 3> values{};
  EXPECT_EQ(reader(values.data(), sizeof(uint32_t), values.size()), 2U);
  EXPECT_EQ(reader.pos, 8U);
  EXPECT_EQ(reader(values.data(), sizeof(uint32_t), 1), 0U);
  EXPECT_EQ(reader.pos, 8U);

  hypervec::MappedFileIOReader oversized_reader(owner);
  void* address = nullptr;
  EXPECT_EQ(
      oversized_reader.mmap(&address, (std::numeric_limits<size_t>::max)(), 2),
      0U);
  EXPECT_EQ(oversized_reader.pos, 0U);
}

TEST(TestMmap, RejectsEmptyFilesAndNonFileReaders) {
#if !defined(__linux__) && !defined(__FreeBSD__) && !defined(_WIN32)
  GTEST_SKIP() << "Memory-mapped persistence is not supported on this OS.";
#endif
  const TempFile empty({});
  EXPECT_THROW(hypervec::MmappedFileMappingOwner(empty.Path().string()),
               hypervec::HypervecException);

  hypervec::VectorIOReader reader;
  EXPECT_THROW(hypervec::ReadIndex(&reader, hypervec::IO_FLAG_MMAP_IFC),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::MappedFileIOReader(nullptr),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::MmappedFileMappingOwner(static_cast<FILE*>(nullptr)),
               hypervec::HypervecException);
}

// IndexBinaryFlat is not included in the current CMake build.
TEST(TestMmap, MmapBinaryFlatcodes) {
  GTEST_SKIP()
      << "IndexBinaryFlat is not available in the current CMake build.";
}
