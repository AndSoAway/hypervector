# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pyhypervec
  DESCRIPTION "Python bindings for hypervec."
  HOMEPAGE_URL "https://github.com/QY-Graph/hypervec"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

macro(configure_swighypervec source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()

  if(HYPERVEC_OPT_LEVEL STREQUAL "dd")
    set_property(SOURCE ${source} APPEND PROPERTY
      SWIG_FLAGS -DHYPERVEC_ENABLE_DD
    )
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
configure_file(swighypervec.swig ${CMAKE_CURRENT_SOURCE_DIR}/swighypervec_avx2.swig COPYONLY)
configure_file(swighypervec.swig ${CMAKE_CURRENT_SOURCE_DIR}/swighypervec_avx512.swig COPYONLY)
configure_file(swighypervec.swig ${CMAKE_CURRENT_SOURCE_DIR}/swighypervec_avx512_spr.swig COPYONLY)
configure_file(swighypervec.swig ${CMAKE_CURRENT_SOURCE_DIR}/swighypervec_sve.swig COPYONLY)

configure_swighypervec(swighypervec.swig)
configure_swighypervec(swighypervec_avx2.swig)
configure_swighypervec(swighypervec_avx512.swig)
configure_swighypervec(swighypervec_avx512_spr.swig)
configure_swighypervec(swighypervec_sve.swig)
configure_swighypervec(hypervec_example_external_module.swig)

if(TARGET hypervec)
  # Manually add headers as extra dependencies of swighypervec.
  set(SWIG_MODULE_swighypervec_EXTRA_DEPS)
  foreach(h ${HYPERVEC_HEADERS})
    list(APPEND SWIG_MODULE_swighypervec_EXTRA_DEPS
      "${hypervec_SOURCE_DIR}/${h}")
    list(APPEND SWIG_MODULE_swighypervec_avx2_EXTRA_DEPS
      "${hypervec_SOURCE_DIR}/${h}")
    list(APPEND SWIG_MODULE_swighypervec_avx512_EXTRA_DEPS
      "${hypervec_SOURCE_DIR}/${h}")
    list(APPEND SWIG_MODULE_swighypervec_avx512_spr_EXTRA_DEPS
      "${hypervec_SOURCE_DIR}/${h}")
    list(APPEND SWIG_MODULE_swighypervec_sve_EXTRA_DEPS
      "${hypervec_SOURCE_DIR}/${h}")
    list(APPEND SWIG_MODULE_hypervec_example_external_module_EXTRA_DEPS
      "${hypervec_SOURCE_DIR}/${h}")
  endforeach()
else()
  find_package(hypervec REQUIRED)
endif()

swig_add_library(swighypervec
  TYPE SHARED
  LANGUAGE python
  SOURCES swighypervec.swig
)

if(TARGET swighypervec AND NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set_target_properties(swighypervec PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
endif()

set_property(TARGET swighypervec PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

set_property(SOURCE swighypervec_avx2.swig
  PROPERTY SWIG_MODULE_NAME swighypervec_avx2)
swig_add_library(swighypervec_avx2
  TYPE SHARED
  LANGUAGE python
  SOURCES swighypervec_avx2.swig
)
set_property(TARGET swighypervec_avx2 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT HYPERVEC_OPT_LEVEL STREQUAL "avx2" AND NOT HYPERVEC_OPT_LEVEL STREQUAL "avx512" AND NOT HYPERVEC_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(swighypervec_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swighypervec_avx512.swig
  PROPERTY SWIG_MODULE_NAME swighypervec_avx512)
swig_add_library(swighypervec_avx512
  TYPE SHARED
  LANGUAGE python
  SOURCES swighypervec_avx512.swig
)
set_property(TARGET swighypervec_avx512 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT HYPERVEC_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(swighypervec_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swighypervec_avx512_spr.swig
  PROPERTY SWIG_MODULE_NAME swighypervec_avx512_spr)
swig_add_library(swighypervec_avx512_spr
  TYPE SHARED
  LANGUAGE python
  SOURCES swighypervec_avx512_spr.swig
)
set_property(TARGET swighypervec_avx512_spr PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT HYPERVEC_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(swighypervec_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swighypervec_sve.swig
  PROPERTY SWIG_MODULE_NAME swighypervec_sve)
swig_add_library(swighypervec_sve
  TYPE SHARED
  LANGUAGE python
  SOURCES swighypervec_sve.swig
)
set_property(TARGET swighypervec_sve PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT HYPERVEC_OPT_LEVEL STREQUAL "sve")
  set_target_properties(swighypervec_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE hypervec_example_external_module.swig
  PROPERTY SWIG_MODULE_NAME hypervec_example_external_module)
swig_add_library(hypervec_example_external_module
  TYPE SHARED
  LANGUAGE python
  SOURCES hypervec_example_external_module.swig
)
set_property(TARGET hypervec_example_external_module PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

if(AIX)
  set_target_properties(swighypervec PROPERTIES SUFFIX .a)
  set_target_properties(hypervec_example_external_module PROPERTIES SUFFIX .a)
elseif(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swighypervec PROPERTIES SUFFIX .so)
  set_target_properties(swighypervec_avx2 PROPERTIES SUFFIX .so)
  set_target_properties(swighypervec_avx512 PROPERTIES SUFFIX .so)
  set_target_properties(swighypervec_avx512_spr PROPERTIES SUFFIX .so)
  set_target_properties(swighypervec_sve PROPERTIES SUFFIX .so)
  set_target_properties(hypervec_example_external_module PROPERTIES SUFFIX .so)
else()
  # we need bigobj for the swig wrapper
  target_compile_options(swighypervec PRIVATE /bigobj)
  target_compile_options(swighypervec_avx2 PRIVATE /bigobj)
  target_compile_options(swighypervec_avx512 PRIVATE /bigobj)
  target_compile_options(swighypervec_avx512_spr PRIVATE /bigobj)
  target_compile_options(swighypervec_sve PRIVATE /bigobj)
  target_compile_options(hypervec_example_external_module PRIVATE /bigobj)
endif()



# Enable DD support in Python bindings when building with Dynamic Dispatch
if(HYPERVEC_OPT_LEVEL STREQUAL "dd")
  # Add compile definition for the generated C++ wrapper
  target_compile_definitions(swighypervec PRIVATE HYPERVEC_ENABLE_DD)
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swighypervec PRIVATE
  hypervec
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swighypervec_avx2 PRIVATE
  hypervec_avx2
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swighypervec_avx512 PRIVATE
  hypervec_avx512
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swighypervec_avx512_spr PRIVATE
  hypervec_avx512_spr
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swighypervec_sve PRIVATE
  hypervec_sve
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(hypervec_example_external_module PRIVATE
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
  swighypervec
  hypervec
)

# Hack so that python_callbacks.h can be included as
# `#include <hypervec/python/python_callbacks.h>`.
target_include_directories(swighypervec PRIVATE ${PROJECT_SOURCE_DIR}/..)
target_include_directories(swighypervec_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/..)
target_include_directories(swighypervec_avx512 PRIVATE ${PROJECT_SOURCE_DIR}/..)
target_include_directories(swighypervec_avx512_spr PRIVATE ${PROJECT_SOURCE_DIR}/..)
target_include_directories(swighypervec_sve PRIVATE ${PROJECT_SOURCE_DIR}/..)
target_include_directories(hypervec_example_external_module PRIVATE ${PROJECT_SOURCE_DIR}/..)

find_package(Python REQUIRED
  COMPONENTS Development.Module NumPy
)

add_library(hypervec_python_callbacks EXCLUDE_FROM_ALL
  python_callbacks.cpp
)
set_property(TARGET hypervec_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

if ("${CMAKE_SYSTEM_NAME}" MATCHES "AIX")
target_link_libraries(hypervec_python_callbacks PRIVATE hypervec)
endif()

# Hack so that python_callbacks.h can be included as
# `#include <hypervec/python/python_callbacks.h>`.
target_include_directories(hypervec_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/..)
target_include_directories(hypervec_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(swighypervec PRIVATE hypervec_python_callbacks)
target_link_libraries(swighypervec_avx2 PRIVATE hypervec_python_callbacks)
target_link_libraries(swighypervec_avx512 PRIVATE hypervec_python_callbacks)
target_link_libraries(swighypervec_avx512_spr PRIVATE hypervec_python_callbacks)
target_link_libraries(swighypervec_sve PRIVATE hypervec_python_callbacks)
target_link_libraries(hypervec_example_external_module PRIVATE hypervec_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)
configure_file(class_wrappers.py class_wrappers.py COPYONLY)
configure_file(extra_wrappers.py extra_wrappers.py COPYONLY)
configure_file(array_conversions.py array_conversions.py COPYONLY)

# file(GLOB files "${CMAKE_SOURCE_DIR}/contrib/*.py")
file(COPY ${CMAKE_SOURCE_DIR}/contrib  DESTINATION .)
