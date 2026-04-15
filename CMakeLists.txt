# @lint-ignore-every LICENSELINT
# Copyright (c) 2024 HyperVec Authors. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

set(HYPERVEC_LANGUAGES CXX)

project(HYPERVEC
  VERSION 1.14.1
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL https://github.com/QY-Graph/hypervec
  LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Valid values are generic, avx2, avx512, avx512_spr, sve.
option(HYPERVEC_OPT_LEVEL "generic" "")
option(HYPERVEC_ENABLE_MKL "Enable MKL." ON)
option(HYPERVEC_ENABLE_PYTHON "Build Python extension." ON)
option(HYPERVEC_ENABLE_C_API "Build C API." OFF)
option(HYPERVEC_ENABLE_EXTRAS "Build extras like benchmarks and demos" ON)
option(HYPERVEC_USE_LTO "Enable Link-Time optimization" OFF)

add_subdirectory(src)

if(HYPERVEC_ENABLE_PYTHON)
  add_subdirectory(src/python)
endif()

if(HYPERVEC_ENABLE_EXTRAS)
  add_subdirectory(test/examples/cpp)
endif()

# CTest must be included in the top level to enable make test target.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test/unit_tests)
  if(NOT HYPERVEC_UNIT_TESTS_ONLY)
    add_subdirectory(test/benchmarks)
  endif()
endif()