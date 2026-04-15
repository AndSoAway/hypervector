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
)

add_executable(hypervec_test ${HYPERVEC_TEST_SRC})

include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

link_to_hypervec_lib(hypervec_test)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850 # release-1.12.1
  OVERRIDE_FIND_PACKAGE)
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

if(NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake
   AND NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/GTestConfig.cmake)
  file(
    WRITE ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake
    [=[
include(CMakeFindDependencyMacro)
find_dependency(googletest)
if(NOT TARGET GTest::GTest)
  add_library(GTest::GTest INTERFACE IMPORTED)
  target_link_libraries(GTest::GTest INTERFACE GTest::gtest)
endif()
if(NOT TARGET GTest::Main)
  add_library(GTest::Main INTERFACE IMPORTED)
  target_link_libraries(GTest::Main INTERFACE GTest::gtest_main)
endif()
]=])
endif()

find_package(OpenMP REQUIRED)
find_package(GTest CONFIG REQUIRED)

set(OMP_LIB "gomp")

target_link_libraries(hypervec_test PRIVATE
  OpenMP::OpenMP_CXX
  ${OMP_LIB}
  GTest::gtest_main
)

if(WIN32)
  target_compile_definitions(hypervec_test PRIVATE HYPERVEC_MAIN_LIB)
endif()

include(GoogleTest)
gtest_discover_tests(hypervec_test)
