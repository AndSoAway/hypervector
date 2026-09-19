# @lint-ignore-every LICENSELINT
# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

function(link_to_hypervec_lib target)
  if(NOT HYPERVEC_OPT_LEVEL STREQUAL "avx2" AND NOT HYPERVEC_OPT_LEVEL STREQUAL "avx512" AND NOT HYPERVEC_OPT_LEVEL STREQUAL "avx512_spr" AND NOT HYPERVEC_OPT_LEVEL STREQUAL "sve" AND NOT HYPERVEC_OPT_LEVEL STREQUAL "dd")
    target_link_libraries(${target} PRIVATE hypervec)
  endif()

  if(HYPERVEC_OPT_LEVEL STREQUAL "avx2")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
    target_link_libraries(${target} PRIVATE hypervec_avx2)
  endif()

  if(HYPERVEC_OPT_LEVEL STREQUAL "avx512")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${target} PRIVATE hypervec_avx512)
  endif()

  if(HYPERVEC_OPT_LEVEL STREQUAL "avx512_spr")
    if(NOT WIN32)
      # Architecture mode to support AVX512 extensions available since Intel (R) Sapphire Rapids.
      # Ref: https://networkbuilders.intel.com/solutionslibrary/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=sapphirerapids -mtune=sapphirerapids>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${target} PRIVATE hypervec_avx512_spr)
  endif()

  if(HYPERVEC_OPT_LEVEL STREQUAL "sve")
    if(NOT WIN32)
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\.[1-9]+)?-[^+ ](\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
      endif()
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\.[1-9]+)?-[^+ ](\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
      endif()
    else()
      # TODO: support Windows
    endif()
    target_link_libraries(${target} PRIVATE hypervec_sve)
  endif()

  if(HYPERVEC_OPT_LEVEL STREQUAL "dd")
    # DD mode: link to main hypervec library with DD-specific definitions
    # When HYPERVEC_OPT_LEVEL=dd, the main hypervec library is built with DD enabled,
    # so we link to hypervec (not a separate hypervec_dd).
    # HYPERVEC_ENABLE_DD exposes SIMDConfig class to consuming code (e.g., tests)
    # COMPILE_SIMD_* flags enable DD code paths in headers (architecture-specific)
    # Note: No SIMD compile flags here - DD handles dispatch internally.
    # Special tests (like test_simd_levels.cpp) that use raw intrinsics
    # should get their own SIMD flags via set_source_files_properties.
    #
    # Architecture-specific definitions mirror simd_dispatch.bzl dispatch_config:
    # - x86_64: AVX2 + AVX512 enabled
    # - aarch64: ARM_NEON enabled
    target_compile_definitions(${target} PRIVATE HYPERVEC_ENABLE_DD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
      target_compile_definitions(${target} PRIVATE COMPILE_SIMD_AVX2 COMPILE_SIMD_AVX512)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
      target_compile_definitions(${target} PRIVATE COMPILE_SIMD_ARM_NEON)
    endif()
    target_link_libraries(${target} PRIVATE hypervec)
  endif()
endfunction()