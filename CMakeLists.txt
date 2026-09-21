# @lint-ignore-every LICENSELINT
# Copyright (c) 2024 HyperVec Authors. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)
if(CMAKE_VERSION VERSION_LESS 3.24.0)
  message(WARNING "CMake version ${CMAKE_VERSION} is less than 3.24.0. "
                  "Consider upgrading CMake to avoid potential issues.")
endif()

set(HYPERVEC_LANGUAGES CXX)

project(HYPERVEC
  VERSION 1.14.1
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL https://github.com/QY-Graph/hypervec
  LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(HYPERVEC_OPT_LEVEL "generic" CACHE STRING
    "CPU optimization level (currently supported: generic)")
set_property(CACHE HYPERVEC_OPT_LEVEL PROPERTY STRINGS generic)

option(BUILD_TESTING "Build the HyperVec test suite" OFF)
option(HYPERVEC_UNIT_TESTS_ONLY "Skip performance benchmarks" OFF)
option(HYPERVEC_FETCH_DEPS
       "Allow CMake to download missing test dependencies" OFF)
option(HYPERVEC_ENABLE_MKL "Enable MKL." OFF)
option(HYPERVEC_ENABLE_PYTHON "Build Python extension." OFF)
option(HYPERVEC_ENABLE_C_API "Build C API." OFF)
option(HYPERVEC_ENABLE_EXTRAS "Build C++ examples" OFF)
option(HYPERVEC_USE_LTO "Enable Link-Time optimization" OFF)

if(NOT HYPERVEC_OPT_LEVEL STREQUAL "generic")
  message(FATAL_ERROR
    "HYPERVEC_OPT_LEVEL='${HYPERVEC_OPT_LEVEL}' is unavailable. The current "
    "CMake build only provides the portable 'generic' library target.")
endif()

if(HYPERVEC_ENABLE_MKL)
  message(FATAL_ERROR
    "HYPERVEC_ENABLE_MKL is not implemented. Leave it OFF and let CMake "
    "select an available BLAS/LAPACK provider.")
endif()

if(HYPERVEC_ENABLE_C_API)
  message(FATAL_ERROR
    "HYPERVEC_ENABLE_C_API is not implemented by the current CMake build.")
endif()

if(HYPERVEC_USE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT HYPERVEC_LTO_SUPPORTED
                      OUTPUT HYPERVEC_LTO_ERROR
                      LANGUAGES CXX)
  if(NOT HYPERVEC_LTO_SUPPORTED)
    message(FATAL_ERROR
      "HYPERVEC_USE_LTO=ON was requested, but IPO/LTO is unavailable: "
      "${HYPERVEC_LTO_ERROR}")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

message(STATUS "HyperVec configuration:")
message(STATUS "  optimization level : ${HYPERVEC_OPT_LEVEL}")
message(STATUS "  build tests         : ${BUILD_TESTING}")
message(STATUS "  fetch dependencies  : ${HYPERVEC_FETCH_DEPS}")
message(STATUS "  build Python        : ${HYPERVEC_ENABLE_PYTHON}")
message(STATUS "  build examples      : ${HYPERVEC_ENABLE_EXTRAS}")
message(STATUS "  enable LTO          : ${HYPERVEC_USE_LTO}")

# CTest is included before test subdirectories so BUILD_TESTING has one clear,
# project-wide meaning. It is OFF by default to keep a core build offline.
include(CTest)

add_subdirectory(src)

if(HYPERVEC_ENABLE_PYTHON)
  add_subdirectory(src/python)
endif()

if(HYPERVEC_ENABLE_EXTRAS)
  add_subdirectory(test/examples/cpp)
endif()

if(BUILD_TESTING)
  add_subdirectory(test/unit_tests)
  if(NOT HYPERVEC_UNIT_TESTS_ONLY)
    add_subdirectory(test/benchmarks)
  endif()
endif()
