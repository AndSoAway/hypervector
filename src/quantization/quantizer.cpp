/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/quantizer.h>
#include <utils/distances/extra_distances.h>
#include <utils/log/assert.h>

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hypervec {
namespace {

void ValidateCountAndPointer(idx_t count, const void* pointer,
                             const char* operation, const char* pointer_name) {
  HYPERVEC_THROW_IF_NOT_FMT(count >= 0,
                            "%s: count must be non-negative, got %" PRId64,
                            operation, static_cast<int64_t>(count));
  HYPERVEC_THROW_IF_NOT_FMT(count == 0 || pointer != nullptr,
                            "%s: %s must not be null when count is positive",
                            operation, pointer_name);
}

size_t FlatCodeSize(idx_t dimension) {
  HYPERVEC_THROW_IF_NOT_FMT(
      dimension > 0, "FlatQuantizer: dimension must be positive, got %" PRId64,
      static_cast<int64_t>(dimension));
  return mul_no_overflow(static_cast<size_t>(dimension), sizeof(float),
                         "FlatQuantizer code size");
}

class BoundedDistanceComputer final : public DistanceComputer {
 public:
  BoundedDistanceComputer(std::unique_ptr<DistanceComputer> inner, idx_t count)
      : inner_(std::move(inner)), count_(count) {
    HYPERVEC_THROW_IF_NOT_MSG(inner_ != nullptr,
                              "quantizer returned a null DistanceComputer");
  }

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                              "DistanceComputer: query must not be null");
    inner_->SetQuery(query);
  }

  float operator()(idx_t index) override {
    CheckIndex(index);
    return (*inner_)(index);
  }

  float symmetric_dis(idx_t lhs, idx_t rhs) override {
    CheckIndex(lhs);
    CheckIndex(rhs);
    return inner_->symmetric_dis(lhs, rhs);
  }

 private:
  void CheckIndex(idx_t index) const {
    HYPERVEC_THROW_IF_NOT_FMT(
        index >= 0 && index < count_,
        "DistanceComputer: index %" PRId64 " out of range [0, %" PRId64 ")",
        static_cast<int64_t>(index), static_cast<int64_t>(count_));
  }

  std::unique_ptr<DistanceComputer> inner_;
  idx_t count_;
};

template <typename VectorDistanceType>
class FlatQuantizerDistanceComputer final : public DistanceComputer {
 public:
  FlatQuantizerDistanceComputer(EncodedVectorView store, idx_t dimension,
                                VectorDistanceType distance)
      : store_(store),
        distance_(std::move(distance)),
        decode_buffer_a_(static_cast<size_t>(dimension)),
        decode_buffer_b_(static_cast<size_t>(dimension)) {}

  void SetQuery(const float* query) override {
    HYPERVEC_THROW_IF_NOT_MSG(query != nullptr,
                              "FlatQuantizer: query must not be null");
    query_ = query;
  }

  float operator()(idx_t index) override {
    HYPERVEC_THROW_IF_NOT_MSG(
        query_ != nullptr,
        "FlatQuantizer: SetQuery must be called before distance evaluation");
    Decode(store_.Code(index), decode_buffer_a_.data());
    return distance_(query_, decode_buffer_a_.data());
  }

  float symmetric_dis(idx_t lhs, idx_t rhs) override {
    Decode(store_.Code(lhs), decode_buffer_a_.data());
    Decode(store_.Code(rhs), decode_buffer_b_.data());
    return distance_(decode_buffer_a_.data(), decode_buffer_b_.data());
  }

 private:
  void Decode(const uint8_t* code, float* vector) const {
    std::memcpy(vector, code, store_.CodeSize());
  }

  EncodedVectorView store_;
  VectorDistanceType distance_;
  const float* query_ = nullptr;
  std::vector<float> decode_buffer_a_;
  std::vector<float> decode_buffer_b_;
};

}  // namespace

EncodedVectorView::EncodedVectorView(const uint8_t* data, idx_t count,
                                     size_t code_size)
    : data_(data), count_(count), code_size_(code_size) {
  HYPERVEC_THROW_IF_NOT_FMT(
      count >= 0, "EncodedVectorView: count must be non-negative, got %" PRId64,
      static_cast<int64_t>(count));
  HYPERVEC_THROW_IF_NOT_MSG(code_size > 0,
                            "EncodedVectorView: code_size must be positive");
  HYPERVEC_THROW_IF_NOT_MSG(
      count == 0 || data != nullptr,
      "EncodedVectorView: data must not be null when count is positive");
  mul_no_overflow(static_cast<size_t>(count), code_size,
                  "EncodedVectorView byte size");
}

const uint8_t* EncodedVectorView::Code(idx_t index) const {
  HYPERVEC_THROW_IF_NOT_FMT(
      index >= 0 && index < count_,
      "EncodedVectorView: index %" PRId64 " out of range [0, %" PRId64 ")",
      static_cast<int64_t>(index), static_cast<int64_t>(count_));
  return data_ + static_cast<size_t>(index) * code_size_;
}

InMemoryCodeStore::InMemoryCodeStore(size_t code_size) : code_size_(code_size) {
  HYPERVEC_THROW_IF_NOT_MSG(code_size > 0,
                            "InMemoryCodeStore: code_size must be positive");
}

idx_t InMemoryCodeStore::Size() const noexcept {
  return static_cast<idx_t>(bytes_.size() / code_size_);
}

const uint8_t* InMemoryCodeStore::Code(idx_t index) const {
  return View().Code(index);
}

uint8_t* InMemoryCodeStore::Code(idx_t index) {
  const size_t offset = static_cast<size_t>(View().Code(index) - bytes_.data());
  return bytes_.data() + offset;
}

EncodedVectorView InMemoryCodeStore::View() const {
  return EncodedVectorView(bytes_.data(), Size(), code_size_);
}

void InMemoryCodeStore::Append(idx_t count, const uint8_t* codes) {
  ValidateCountAndPointer(count, codes, "InMemoryCodeStore::Append", "codes");
  if (count == 0) {
    return;
  }

  const size_t additional_bytes =
      mul_no_overflow(static_cast<size_t>(count), code_size_,
                      "InMemoryCodeStore::Append byte size");
  const size_t new_size = add_no_overflow(
      bytes_.size(), additional_bytes, "InMemoryCodeStore::Append total size");
  HYPERVEC_THROW_IF_NOT_MSG(
      new_size / code_size_ <=
          static_cast<size_t>((std::numeric_limits<idx_t>::max)()),
      "InMemoryCodeStore::Append exceeds the supported vector count");

  std::vector<uint8_t> source_copy;
  if (!bytes_.empty()) {
    const uintptr_t source_address = reinterpret_cast<uintptr_t>(codes);
    const uintptr_t begin_address = reinterpret_cast<uintptr_t>(bytes_.data());
    const uintptr_t end_address = begin_address + bytes_.size();
    if (source_address >= begin_address && source_address < end_address) {
      const size_t source_offset = source_address - begin_address;
      HYPERVEC_THROW_IF_NOT_MSG(
          additional_bytes <= bytes_.size() - source_offset,
          "InMemoryCodeStore::Append source exceeds the existing code store");
      source_copy.assign(codes, codes + additional_bytes);
      codes = source_copy.data();
    }
  }

  const size_t old_size = bytes_.size();
  bytes_.resize(new_size);
  std::memcpy(bytes_.data() + old_size, codes, additional_bytes);
}

void Quantizer::Train(idx_t count, const float* vectors) {
  ValidateCountAndPointer(count, vectors, "Quantizer::Train", "vectors");
  TrainImpl(count, vectors);
}

void Quantizer::Encode(idx_t count, const float* vectors,
                       uint8_t* codes) const {
  HYPERVEC_THROW_IF_NOT_MSG(IsTrained(),
                            "Quantizer::Encode requires a trained quantizer");
  ValidateCountAndPointer(count, vectors, "Quantizer::Encode", "vectors");
  ValidateCountAndPointer(count, codes, "Quantizer::Encode", "codes");
  const size_t vector_size =
      mul_no_overflow(static_cast<size_t>(Dimension()), sizeof(float),
                      "Quantizer::Encode vector size");
  mul_no_overflow(static_cast<size_t>(count), vector_size,
                  "Quantizer::Encode input size");
  mul_no_overflow(static_cast<size_t>(count), CodeSize(),
                  "Quantizer::Encode output size");
  EncodeImpl(count, vectors, codes);
}

void Quantizer::Decode(idx_t count, const uint8_t* codes,
                       float* vectors) const {
  HYPERVEC_THROW_IF_NOT_MSG(IsTrained(),
                            "Quantizer::Decode requires a trained quantizer");
  ValidateCountAndPointer(count, codes, "Quantizer::Decode", "codes");
  ValidateCountAndPointer(count, vectors, "Quantizer::Decode", "vectors");
  const size_t vector_size =
      mul_no_overflow(static_cast<size_t>(Dimension()), sizeof(float),
                      "Quantizer::Decode vector size");
  mul_no_overflow(static_cast<size_t>(count), CodeSize(),
                  "Quantizer::Decode input size");
  mul_no_overflow(static_cast<size_t>(count), vector_size,
                  "Quantizer::Decode output size");
  DecodeImpl(count, codes, vectors);
}

std::unique_ptr<DistanceComputer> Quantizer::CreateDistanceComputer(
    EncodedVectorView store) const {
  HYPERVEC_THROW_IF_NOT_MSG(
      IsTrained(),
      "Quantizer::CreateDistanceComputer requires a trained quantizer");
  HYPERVEC_THROW_IF_NOT_FMT(
      store.CodeSize() == CodeSize(),
      "Quantizer::CreateDistanceComputer: code size mismatch, expected %zu, "
      "got %zu",
      CodeSize(), store.CodeSize());
  return std::make_unique<BoundedDistanceComputer>(
      CreateDistanceComputerImpl(store), store.Size());
}

FlatQuantizer::FlatQuantizer(idx_t dimension, MetricType metric,
                             float metric_arg)
    : dimension_(dimension),
      metric_(MetricTypeFromInt(static_cast<int>(metric))),
      metric_arg_(metric_arg),
      code_size_(FlatCodeSize(dimension)) {
  HYPERVEC_THROW_IF_NOT_MSG(
      metric_ != kMetricLp || (std::isfinite(metric_arg_) && metric_arg_ > 0),
      "FlatQuantizer: kMetricLp requires a finite, positive metric_arg");
}

void FlatQuantizer::TrainImpl(idx_t /*count*/, const float* /*vectors*/) {}

void FlatQuantizer::EncodeImpl(idx_t count, const float* vectors,
                               uint8_t* codes) const {
  const size_t bytes = mul_no_overflow(static_cast<size_t>(count), code_size_,
                                       "FlatQuantizer::Encode byte size");
  if (bytes > 0) {
    std::memcpy(codes, vectors, bytes);
  }
}

void FlatQuantizer::DecodeImpl(idx_t count, const uint8_t* codes,
                               float* vectors) const {
  const size_t bytes = mul_no_overflow(static_cast<size_t>(count), code_size_,
                                       "FlatQuantizer::Decode byte size");
  if (bytes > 0) {
    std::memcpy(vectors, codes, bytes);
  }
}

std::unique_ptr<DistanceComputer> FlatQuantizer::CreateDistanceComputerImpl(
    EncodedVectorView store) const {
  return with_VectorDistance(
      static_cast<size_t>(dimension_), metric_, metric_arg_,
      [store, this](auto distance) -> std::unique_ptr<DistanceComputer> {
        using DistanceType = decltype(distance);
        return std::make_unique<FlatQuantizerDistanceComputer<DistanceType>>(
            store, dimension_, std::move(distance));
      });
}

}  // namespace hypervec
