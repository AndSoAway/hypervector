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
  find_package(benchmark CONFIG QUIET)
  if(NOT TARGET benchmark::benchmark AND HYPERVEC_FETCH_DEPS)
    message(STATUS "Google Benchmark was not found locally; fetching v1.8.3")
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG 344117638c8ff7e239044fd0fa7085839fc03021) # v1.8.3
    FetchContent_MakeAvailable(googlebenchmark)
  endif()

  if(NOT TARGET benchmark::benchmark)
    message(FATAL_ERROR
      "Google Benchmark was not found. Install its CMake package or "
      "explicitly allow the pinned download with -DHYPERVEC_FETCH_DEPS=ON.")
  endif()

  find_package(Threads REQUIRED)

  include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

  foreach(bench ${HYPERVEC_PERF_TEST_SRC})
    get_filename_component(bench_exec ${bench} NAME_WE)
    add_executable(${bench_exec} ${bench})
    link_to_hypervec_lib(${bench_exec})
    target_link_libraries(${bench_exec} PRIVATE
      benchmark::benchmark)
    target_include_directories(${bench_exec} PRIVATE ${CMAKE_SOURCE_DIR})
  endforeach()
endif()
