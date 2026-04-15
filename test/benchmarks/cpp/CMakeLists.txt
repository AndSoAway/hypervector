# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every LINEWRAP
project(hypervec_perf_tests)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_INSTALL_DOCS OFF)

include(FetchContent)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark
FetchContent_MakeAvailable(
  googlebenchmark)


find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(gflags REQUIRED)

add_library(hypervec_perf_tests_utils
  utils.cpp
)
# `#include <hypervec/perf_tests/utils.h>` or any other headers
target_include_directories(hypervec_perf_tests_utils PRIVATE
   ${CMAKE_SOURCE_DIR})

include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

link_to_hypervec_lib(hypervec_perf_tests_utils)

set(HYPERVEC_PERF_TEST_SRC
  bench_no_multithreading_rcq_search.cpp
  bench_scalar_quantizer_accuracy.cpp
  bench_scalar_quantizer_decode.cpp
  bench_scalar_quantizer_distance.cpp
  bench_scalar_quantizer_encode.cpp
)
foreach(bench ${HYPERVEC_PERF_TEST_SRC})
  get_filename_component(bench_exec ${bench} NAME_WE)
  add_executable(${bench_exec} ${bench})
  link_to_hypervec_lib(${bench_exec})
  target_link_libraries(${bench_exec} PRIVATE hypervec_perf_tests_utils OpenMP::OpenMP_CXX benchmark::benchmark gflags)
  # `#include <hypervec/perf_tests/utils.h>` or any other headers
  target_include_directories(${bench_exec} PRIVATE
   ${CMAKE_SOURCE_DIR})

endforeach()
