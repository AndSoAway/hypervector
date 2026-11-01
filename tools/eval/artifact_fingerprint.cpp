/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include "artifact_fingerprint.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "json_util.h"

namespace hypervec::eval_cli {

namespace {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

uint32_t LoadBigEndian(const uint8_t* input) {
  return (static_cast<uint32_t>(input[0]) << 24U) |
         (static_cast<uint32_t>(input[1]) << 16U) |
         (static_cast<uint32_t>(input[2]) << 8U) |
         static_cast<uint32_t>(input[3]);
}

class Sha256 {
 public:
  void Update(const uint8_t* input, size_t size) {
    if (size > (std::numeric_limits<uint64_t>::max)() - total_size_) {
      throw std::runtime_error("artifact is too large to fingerprint");
    }
    total_size_ += size;

    if (buffer_size_ != 0) {
      const size_t copied = std::min(size, buffer_.size() - buffer_size_);
      std::copy_n(input, copied, buffer_.begin() + buffer_size_);
      buffer_size_ += copied;
      input += copied;
      size -= copied;
      if (buffer_size_ == buffer_.size()) {
        Transform(buffer_.data());
        buffer_size_ = 0;
      }
    }
    while (size >= buffer_.size()) {
      Transform(input);
      input += buffer_.size();
      size -= buffer_.size();
    }
    std::copy_n(input, size, buffer_.begin());
    buffer_size_ = size;
  }

  std::string Finish() {
    if (total_size_ > (std::numeric_limits<uint64_t>::max)() / 8U) {
      throw std::runtime_error("artifact is too large to fingerprint");
    }
    const uint64_t bit_count = total_size_ * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56U) {
      std::fill(buffer_.begin() + buffer_size_, buffer_.end(), 0U);
      Transform(buffer_.data());
      buffer_size_ = 0;
    }
    std::fill(buffer_.begin() + buffer_size_, buffer_.begin() + 56U, 0U);
    for (size_t offset = 0; offset < 8; ++offset) {
      buffer_[63U - offset] = static_cast<uint8_t>(bit_count >> (offset * 8U));
    }
    Transform(buffer_.data());

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint32_t word : state_) {
      output << std::setw(8) << word;
    }
    return output.str();
  }

 private:
  void Transform(const uint8_t* block) {
    std::array<uint32_t, 64> words{};
    for (size_t offset = 0; offset < 16; ++offset) {
      words[offset] = LoadBigEndian(block + offset * 4U);
    }
    for (size_t offset = 16; offset < words.size(); ++offset) {
      const uint32_t before_15 = words[offset - 15U];
      const uint32_t before_2 = words[offset - 2U];
      const uint32_t sigma0 = std::rotr(before_15, 7) ^
                              std::rotr(before_15, 18) ^ (before_15 >> 3U);
      const uint32_t sigma1 =
          std::rotr(before_2, 17) ^ std::rotr(before_2, 19) ^ (before_2 >> 10U);
      words[offset] =
          words[offset - 16U] + sigma0 + words[offset - 7U] + sigma1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (size_t offset = 0; offset < words.size(); ++offset) {
      const uint32_t sum1 =
          std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t temporary1 =
          h + sum1 + choice + kRoundConstants[offset] + words[offset];
      const uint32_t sum0 =
          std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                    0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                    0x1f83d9abU, 0x5be0cd19U};
  std::array<uint8_t, 64> buffer_{};
  size_t buffer_size_ = 0;
  uint64_t total_size_ = 0;
};

}  // namespace

ArtifactFingerprint FingerprintFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open artifact for fingerprinting: " +
                             path.string());
  }
  Sha256 sha256;
  uint64_t size_bytes = 0;
  std::array<char, 64U * 1024U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      const uint64_t unsigned_count = static_cast<uint64_t>(count);
      if (unsigned_count >
          (std::numeric_limits<uint64_t>::max)() - size_bytes) {
        throw std::runtime_error("artifact size exceeds uint64_t: " +
                                 path.string());
      }
      sha256.Update(reinterpret_cast<const uint8_t*>(buffer.data()),
                    static_cast<size_t>(count));
      size_bytes += unsigned_count;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("cannot read artifact for fingerprinting: " +
                             path.string());
  }
  return {size_bytes, sha256.Finish()};
}

void WriteArtifactJson(std::ostream& output, const std::filesystem::path& path,
                       const ArtifactFingerprint& fingerprint,
                       std::string_view indentation) {
  const std::string normalized =
      std::filesystem::absolute(path).lexically_normal().generic_string();
  output << indentation << "{\n"
         << indentation << "  \"path\": \"" << JsonEscape(normalized) << "\",\n"
         << indentation << "  \"size_bytes\": " << fingerprint.size_bytes
         << ",\n"
         << indentation << "  \"sha256\": \"" << fingerprint.sha256 << "\"\n"
         << indentation << "}";
}

}  // namespace hypervec::eval_cli
