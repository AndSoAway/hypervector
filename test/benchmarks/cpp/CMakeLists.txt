# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every LINEWRAP
project(hypervec_perf_tests)

# TODO: re-add benchmarks once concrete targets exist (PQ / SQ / RCQ).
#       Each new bench .cpp goes into HYPERVEC_PERF_TEST_SRC and gets one
#       executable via the foreach loop below.
set(HYPERVEC_PERF_TEST_SRC
)

if(HYPERVEC_PERF_TEST_SRC)
  set(BENCHMARK_ENABLE_TESTING OFF)
  set(BENCHMARK_INSTALL_DOCS OFF)

  include(FetchContent)
  FetchContent_Declare(googlebenchmark
          GIT_REPOSITORY https://github.com/google/benchmark.git
          GIT_TAG main)
  FetchContent_MakeAvailable(googlebenchmark)

  find_package(Threads REQUIRED)
  find_package(OpenMP REQUIRED)
  find_package(gflags REQUIRED)

  include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

  foreach(bench ${HYPERVEC_PERF_TEST_SRC})
    get_filename_component(bench_exec ${bench} NAME_WE)
    add_executable(${bench_exec} ${bench})
    link_to_hypervec_lib(${bench_exec})
    target_link_libraries(${bench_exec} PRIVATE
      OpenMP::OpenMP_CXX
      benchmark::benchmark
      gflags)
    target_include_directories(${bench_exec} PRIVATE ${CMAKE_SOURCE_DIR})
  endforeach()
endif()
