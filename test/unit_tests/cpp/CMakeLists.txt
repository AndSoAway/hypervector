# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

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
  test_index_ivf_flat.cpp
  test_index_ivfpq.cpp
  test_index_hnsw_pq.cpp
  test_lvq.cpp
)

add_executable(hypervec_test ${HYPERVEC_TEST_SRC})

include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

link_to_hypervec_lib(hypervec_test)

find_package(GTest CONFIG QUIET)
if(NOT TARGET GTest::gtest_main AND HYPERVEC_FETCH_DEPS)
  message(STATUS "GoogleTest was not found locally; fetching pinned release-1.12.1")
  include(FetchContent)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850)
  FetchContent_MakeAvailable(googletest)
endif()

if(NOT TARGET GTest::gtest_main)
  message(FATAL_ERROR
    "GoogleTest was not found. Install a GTest CMake package or explicitly "
    "allow the pinned dependency download with -DHYPERVEC_FETCH_DEPS=ON.")
endif()

find_package(OpenMP REQUIRED)

set(OMP_LIB "")
if(MINGW)
  set(OMP_LIB "gomp")
endif()

target_link_libraries(hypervec_test PRIVATE
  OpenMP::OpenMP_CXX
  ${OMP_LIB}
  GTest::gtest_main
)

if(WIN32)
  target_compile_definitions(hypervec_test PRIVATE HYPERVEC_MAIN_LIB)
endif()

include(GoogleTest)
# Unit tests must be reproducible independently of the machine's default
# OpenMP thread count. Parallel behaviour belongs in dedicated concurrency and
# performance tests rather than changing numerical fixtures implicitly.
gtest_discover_tests(hypervec_test
  PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
