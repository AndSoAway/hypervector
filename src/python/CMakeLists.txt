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
find_package(Python REQUIRED
  COMPONENTS Interpreter Development.Module NumPy
)

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

function(configure_swig_doxygen target)
  if(SWIG_VERSION VERSION_GREATER_EQUAL 4.0)
    set_property(TARGET ${target} PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
  endif()
endfunction()

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

endmacro()

configure_swighypervec(swighypervec.swig)
configure_swighypervec(hypervec_example_external_module.swig)

if(TARGET hypervec)
  # Manually add headers as extra dependencies of swighypervec.
  set(SWIG_MODULE_swighypervec_EXTRA_DEPS)
  foreach(h ${HYPERVEC_HEADERS})
    list(APPEND SWIG_MODULE_swighypervec_EXTRA_DEPS
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

configure_swig_doxygen(swighypervec)

set_property(SOURCE hypervec_example_external_module.swig
  PROPERTY SWIG_MODULE_NAME hypervec_example_external_module)
swig_add_library(hypervec_example_external_module
  TYPE SHARED
  LANGUAGE python
  SOURCES hypervec_example_external_module.swig
)
configure_swig_doxygen(hypervec_example_external_module)

if(AIX)
  set_target_properties(swighypervec PROPERTIES SUFFIX .a)
  set_target_properties(hypervec_example_external_module PROPERTIES SUFFIX .a)
elseif(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swighypervec PROPERTIES SUFFIX .so)
  set_target_properties(hypervec_example_external_module PROPERTIES SUFFIX .so)
elseif(MSVC)
  # we need bigobj for the swig wrapper
  target_compile_options(swighypervec PRIVATE /bigobj)
  target_compile_options(hypervec_example_external_module PRIVATE /bigobj)
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swighypervec PRIVATE
  hypervec
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

if(MINGW)
  target_link_libraries(swighypervec PRIVATE gomp)
  target_link_libraries(hypervec_example_external_module PRIVATE gomp)
endif()

# Hack so that python_callbacks.h can be included as
# `#include <hypervec/python/python_callbacks.h>`.
target_include_directories(swighypervec PRIVATE ${PROJECT_SOURCE_DIR}/..)
target_include_directories(hypervec_example_external_module PRIVATE ${PROJECT_SOURCE_DIR}/..)

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
target_include_directories(hypervec_python_callbacks PRIVATE ${hypervec_SOURCE_DIR}/include)
target_include_directories(hypervec_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(swighypervec PRIVATE hypervec_python_callbacks)
target_link_libraries(hypervec_example_external_module PRIVATE hypervec_python_callbacks)

# Python source modules — single source of truth is the src/python directory
# itself, not a hand-maintained per-file list (Issue #41). GLOB + COPYONLY
# mirrors every *.py into the build dir so `pip wheel .` (run from
# build-docker/src/python) always sees exactly the file set that setup.py
# packages; newly added modules can no longer drift out of the wheel.
# CONFIGURE_DEPENDS re-runs the glob when files are added or removed.
# Note: supersedes the per-file hypervec_bundle.py addition from #39 — the
# glob already covers it (verified: 17 modules land in the wheel on both x86
# and ARM).
file(GLOB _python_src_modules CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(_py_module IN LISTS _python_src_modules)
  get_filename_component(_py_name "${_py_module}" NAME)
  configure_file("${_py_module}" "${_py_name}" COPYONLY)
endforeach()
unset(_python_src_modules)
unset(_py_module)
unset(_py_name)

# Non-.py packaging metadata copied explicitly (setup.py consumes them from
# the build dir when it assembles the hypervec/ sub-package).
configure_file(__init__.pyi __init__.pyi COPYONLY)
configure_file(py.typed py.typed COPYONLY)
configure_file(hypervec.proto hypervec.proto COPYONLY)

# file(GLOB files "${CMAKE_SOURCE_DIR}/contrib/*.py")
if(EXISTS ${CMAKE_SOURCE_DIR}/contrib)
  file(COPY ${CMAKE_SOURCE_DIR}/contrib DESTINATION .)
endif()
