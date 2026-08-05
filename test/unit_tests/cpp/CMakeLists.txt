# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

# HNSW-only test configuration

set(HYPERVEC_TEST_SRC
  test_heap.cpp
  test_hnsw.cpp
  test_callback.cpp
  test_utils.cpp
  test_mmap.cpp
  test_factory_tools.cpp
  test_distances_simd.cpp
  test_omp_threads.cpp
  test_partitioning.cpp
  test_simd_levels.cpp
  test_simdlib.cpp
  test_kmeans.cpp
  test_pq.cpp
  test_index_pq.cpp
  test_index_ivfpq.cpp
  test_index_hnsw_pq.cpp
  test_lvq.cpp
  test_rabitq.cpp
  test_index_hnsw_rabitq.cpp
)

add_executable(hypervec_test ${HYPERVEC_TEST_SRC})

include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

link_to_hypervec_lib(hypervec_test)

# Prefer system-installed Google Test; fall back to downloading from GitHub.
find_package(GTest QUIET CONFIG)
if(NOT GTest_FOUND)
  find_package(GTest QUIET MODULE)
endif()

if(NOT GTest_FOUND)
  # Download gtest via FetchContent
  include(FetchContent)
  set(GOOGLETEST_TAG release-1.12.1)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850
    OVERRIDE_FIND_PACKAGE)
  set(BUILD_GMOCK CACHE BOOL OFF)
  set(INSTALL_GTEST CACHE BOOL OFF)

  # Disable automatic git update to avoid clock-skew issues in WSL
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

  FetchContent_MakeAvailable(googletest)
  if(NOT TARGET GTest::GTest)
    add_library(GTest::GTest ALIAS gtest)
  endif()
  if(NOT TARGET GTest::Main)
    add_library(GTest::Main ALIAS gtest_main)
  endif()
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(hypervec_test PRIVATE
  OpenMP::OpenMP_CXX
  GTest::gtest_main
)

if(WIN32)
  target_compile_definitions(hypervec_test PRIVATE HYPERVEC_MAIN_LIB)
endif()

include(GoogleTest)
gtest_discover_tests(hypervec_test)