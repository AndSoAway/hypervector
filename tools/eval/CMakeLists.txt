# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2
# (the "License") found in the LICENSE file in the root directory of this
# source tree.

include(FetchContent)

if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json QUIET)
endif()
if(NOT TARGET nlohmann_json::nlohmann_json AND HYPERVEC_FETCH_DEPS)
  set(JSON_BuildTests OFF CACHE BOOL "Disable nlohmann-json tests" FORCE)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH MD5=d603041cbc6051edbaa02ebb82cf0aa9
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

if(NOT TARGET tabulate::tabulate)
  find_package(tabulate QUIET)
endif()
if(NOT TARGET tabulate::tabulate AND HYPERVEC_FETCH_DEPS)
  FetchContent_Declare(
    tabulate
    URL https://github.com/p-ranav/tabulate/archive/3a58301067bbc03da89ae5a51b3e05b7da719d38.tar.gz
    URL_HASH MD5=9d396f30fcc513abbb970773c8ddf8ff
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(tabulate)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
  message(FATAL_ERROR
    "nlohmann-json is required for hypervec_eval. Install it or set "
    "HYPERVEC_FETCH_DEPS=ON.")
endif()
if(NOT TARGET tabulate::tabulate)
  message(FATAL_ERROR
    "tabulate is required for hypervec_eval. Install it or set "
    "HYPERVEC_FETCH_DEPS=ON.")
endif()

set(HYPERVEC_EVAL_SOURCES
  case/eval_case.cpp
  eval_config.cpp
  eval_dataset.cpp
  exporter/exporter.cpp
  exporter/formatter.cpp
  monitor/duration_monitor.cpp
  monitor/latency_monitor.cpp
  monitor/monitor.cpp
  monitor/recall_monitor.cpp
  platform_memory.cpp)

add_library(hypervec_eval_lib STATIC ${HYPERVEC_EVAL_SOURCES})
target_include_directories(hypervec_eval_lib
  PUBLIC
    ${CMAKE_SOURCE_DIR}/tools
    ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(hypervec_eval_lib
  PUBLIC
    nlohmann_json::nlohmann_json
    tabulate::tabulate)
include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)
link_to_hypervec_lib(hypervec_eval_lib)

add_executable(hypervec_eval main.cpp)
target_link_libraries(hypervec_eval PRIVATE hypervec_eval_lib)

if(BUILD_TESTING)
  enable_testing()
  add_executable(hypervec_eval_test tests/eval_test.cpp)
  target_link_libraries(hypervec_eval_test PRIVATE hypervec_eval_lib)
  add_test(NAME hypervec_eval_test COMMAND hypervec_eval_test)
endif()
