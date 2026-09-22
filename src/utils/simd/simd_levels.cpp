/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/log/assert.h>
#include <utils/simd/simd_dispatch.h>
#include <utils/simd/simd_levels.h>

#include <cstdlib>
#include <string>

namespace hypervec {
namespace {

uint64_t DetectSupportedLevels() {
  uint64_t mask = uint64_t{1} << static_cast<int>(SIMDLevel::NONE);
#if defined(HYPERVEC_ENABLE_DD) && defined(COMPILE_SIMD_AVX2)
  // Compiler builtins include OSXSAVE/XCR0 checks, not just CPU feature bits.
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    mask |= uint64_t{1} << static_cast<int>(SIMDLevel::AVX2);
  }
#endif
  return mask;
}

uint64_t SupportedLevels() {
  // Also safe when a caller queries SIMD from another TU's static constructor.
  static const uint64_t mask = DetectSupportedLevels();
  return mask;
}

template <SIMDLevel Level>
SIMDLevel DispatchedLevel() {
  return Level;
}

}  // namespace

std::atomic<SIMDLevel> SIMDConfig::level{SIMDLevel::NONE};
const uint64_t SIMDConfig::supported_simd_levels = SupportedLevels();

SIMDConfig::SIMDConfig(const char** override_env) {
  SIMDLevel selected = auto_detect_simd_level();
#ifdef HYPERVEC_ENABLE_DD
  const char* env =
      override_env ? *override_env : std::getenv("HYPERVEC_SIMD_LEVEL");
  if (env != nullptr) {
    selected = to_simd_level(env);
    HYPERVEC_THROW_IF_NOT_MSG(is_simd_level_available(selected),
                              "HYPERVEC_SIMD_LEVEL requests an unavailable "
                              "CPU/OS or uncompiled level");
  }
#else
  (void)override_env;
#endif
  level.store(selected, std::memory_order_relaxed);
}

SIMDLevel SIMDConfig::get_level() {
  // Lazy, thread-safe initialization: a bad environment setting throws from
  // a normal API call, never from a process-wide static constructor.
  static const SIMDConfig initialized;
  (void)initialized;
  return level.load(std::memory_order_relaxed);
}

void SIMDConfig::set_level(SIMDLevel selected) {
  (void)get_level();
  HYPERVEC_THROW_IF_NOT_MSG(is_simd_level_available(selected),
                            "SIMDConfig::set_level: unavailable SIMD level");
  level.store(selected, std::memory_order_relaxed);
}

std::string SIMDConfig::get_level_name() { return to_string(get_level()); }

bool SIMDConfig::is_simd_level_available(SIMDLevel selected) {
  const int value = static_cast<int>(selected);
  return value >= 0 && value < static_cast<int>(SIMDLevel::COUNT) &&
         (SupportedLevels() & (uint64_t{1} << value)) != 0;
}

SIMDLevel SIMDConfig::auto_detect_simd_level() {
  return is_simd_level_available(SIMDLevel::AVX2) ? SIMDLevel::AVX2
                                                  : SIMDLevel::NONE;
}

SIMDLevel SIMDConfig::get_dispatched_level() {
  DISPATCH_SIMDLevel(DispatchedLevel);
}

std::string to_string(SIMDLevel level) {
  switch (level) {
    case SIMDLevel::NONE:
      return "NONE";
    case SIMDLevel::AVX2:
      return "AVX2";
    case SIMDLevel::AVX512:
      return "AVX512";
    case SIMDLevel::AVX512_SPR:
      return "AVX512_SPR";
    case SIMDLevel::ARM_NEON:
      return "ARM_NEON";
    case SIMDLevel::ARM_SVE:
      return "ARM_SVE";
    case SIMDLevel::COUNT:
    default:
      throw HypervecException("Invalid SIMDLevel");
  }
}

SIMDLevel to_simd_level(const std::string& level_str) {
  if (level_str == "NONE") {
    return SIMDLevel::NONE;
  }
  if (level_str == "AVX2") {
    return SIMDLevel::AVX2;
  }
  if (level_str == "AVX512") {
    return SIMDLevel::AVX512;
  }
  if (level_str == "AVX512_SPR") {
    return SIMDLevel::AVX512_SPR;
  }
  if (level_str == "ARM_NEON") {
    return SIMDLevel::ARM_NEON;
  }
  if (level_str == "ARM_SVE") {
    return SIMDLevel::ARM_SVE;
  }

  throw HypervecException("Invalid SIMD level string: " + level_str);
}

}  // namespace hypervec
