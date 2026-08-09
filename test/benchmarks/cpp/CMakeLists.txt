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
  bench_rabitq.cpp
  vector_dataset_utils.cpp
)

# Build benchmarks regardless of HYPERVEC_PERF_TEST_SRC content
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_INSTALL_DOCS OFF)

find_package(OpenMP REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

foreach(bench_src ${HYPERVEC_PERF_TEST_SRC})
  get_filename_component(bench_exec ${bench_src} NAME_WE)
  # vector_dataset_utils.cpp is a utility file, not a standalone executable
  if(bench_exec STREQUAL "vector_dataset_utils")
    continue()
  endif()
  add_executable(${bench_exec} ${bench_src} vector_dataset_utils.cpp)
  set_target_properties(${bench_exec} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  link_to_hypervec_lib(${bench_exec})
  target_link_libraries(${bench_exec} PRIVATE
    OpenMP::OpenMP_CXX)
  target_include_directories(${bench_exec} PRIVATE ${CMAKE_SOURCE_DIR})
endforeach()
