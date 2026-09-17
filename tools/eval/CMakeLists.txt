# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2
# (the "License") found in the LICENSE file in the root directory of this
# source tree.

include(FetchContent)

find_package(HDF5 QUIET COMPONENTS CXX)
if(NOT HDF5_FOUND AND HYPERVEC_FETCH_DEPS)
  foreach(output_kind ARCHIVE LIBRARY RUNTIME)
    set(output_variable CMAKE_${output_kind}_OUTPUT_DIRECTORY)
    if(DEFINED ${output_variable})
      set(HYPERVEC_EVAL_SAVED_${output_kind}_DEFINED TRUE)
      set(HYPERVEC_EVAL_SAVED_${output_kind} "${${output_variable}}")
    endif()
  endforeach()
  set(HDF5_BUILD_CPP_LIB ON CACHE BOOL "Build HDF5 C++ library" FORCE)
  set(HDF5_BUILD_TOOLS OFF CACHE BOOL "Disable HDF5 tools" FORCE)
  set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "Disable HDF5 examples" FORCE)
  set(HDF5_BUILD_TESTING OFF CACHE BOOL "Disable HDF5 tests" FORCE)
  find_package(ZLIB QUIET)
  if(ZLIB_FOUND)
    set(HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "Enable HDF5 zlib support" FORCE)
  else()
    set(HDF5_ENABLE_Z_LIB_SUPPORT OFF CACHE BOOL "Disable HDF5 zlib support" FORCE)
  endif()
  set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE BOOL "Disable HDF5 szip support" FORCE)
  set(HYPERVEC_EVAL_PARENT_BUILD_TESTING ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "Disable fetched HDF5 tests" FORCE)
  FetchContent_Declare(
    hdf5
    URL https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.4.tar.gz
    URL_HASH MD5=fdea52afcce07ed6c3e2a36e7fa11f21
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(hdf5)
  set(BUILD_TESTING ${HYPERVEC_EVAL_PARENT_BUILD_TESTING}
      CACHE BOOL "Build project tests" FORCE)
  foreach(output_kind ARCHIVE LIBRARY RUNTIME)
    set(output_variable CMAKE_${output_kind}_OUTPUT_DIRECTORY)
    if(HYPERVEC_EVAL_SAVED_${output_kind}_DEFINED)
      set(${output_variable} "${HYPERVEC_EVAL_SAVED_${output_kind}}"
          CACHE PATH "Project output directory" FORCE)
    else()
      unset(${output_variable})
      unset(${output_variable} CACHE)
    endif()
  endforeach()
endif()

if(NOT TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
  find_package(yaml-cpp QUIET)
endif()
if(NOT TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp AND
   HYPERVEC_FETCH_DEPS)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
  FetchContent_Declare(
    yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz
    URL_HASH MD5=7d17de1b2a4b1d2776181f67c940bcdf
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(yaml-cpp)
endif()

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

if(TARGET hdf5::hdf5_cpp)
  set(HYPERVEC_EVAL_HDF5_TARGET hdf5::hdf5_cpp)
elseif(TARGET hdf5_cpp-static)
  set(HYPERVEC_EVAL_HDF5_TARGET hdf5_cpp-static)
elseif(TARGET hdf5_cpp-shared)
  set(HYPERVEC_EVAL_HDF5_TARGET hdf5_cpp-shared)
elseif(HDF5_FOUND)
  set(HYPERVEC_EVAL_HDF5_TARGET ${HDF5_CXX_LIBRARIES})
  set(HYPERVEC_EVAL_HDF5_INCLUDES ${HDF5_INCLUDE_DIRS})
else()
  message(FATAL_ERROR
    "HDF5 C++ support is required for hypervec_eval. Install it or set "
    "HYPERVEC_FETCH_DEPS=ON.")
endif()

if(TARGET yaml-cpp::yaml-cpp)
  set(HYPERVEC_EVAL_YAML_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(HYPERVEC_EVAL_YAML_TARGET yaml-cpp)
else()
  message(FATAL_ERROR
    "yaml-cpp is required for hypervec_eval. Install it or set "
    "HYPERVEC_FETCH_DEPS=ON.")
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
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/test
    ${CMAKE_SOURCE_DIR}/src/include
  PRIVATE ${HYPERVEC_EVAL_HDF5_INCLUDES})
target_link_libraries(hypervec_eval_lib
  PUBLIC
    nlohmann_json::nlohmann_json
    ${HYPERVEC_EVAL_YAML_TARGET}
    tabulate::tabulate
  PRIVATE
    ${HYPERVEC_EVAL_HDF5_TARGET})
include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)
link_to_hypervec_lib(hypervec_eval_lib)

add_executable(hypervec_eval main.cpp)
target_link_libraries(hypervec_eval PRIVATE hypervec_eval_lib)

if(BUILD_TESTING)
  enable_testing()
  add_executable(hypervec_eval_test tests/eval_test.cpp)
  target_include_directories(hypervec_eval_test PRIVATE
    ${HYPERVEC_EVAL_HDF5_INCLUDES})
  target_link_libraries(hypervec_eval_test PRIVATE
    hypervec_eval_lib ${HYPERVEC_EVAL_HDF5_TARGET})
  add_test(NAME hypervec_eval_test COMMAND hypervec_eval_test)
endif()
