# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2
# (the "License") found in the LICENSE file in the root directory of this
# source tree.

# Fetches cppjieba (header-only Chinese segmentation library) and its bundled
# limonp dependency via CMake FetchContent.  After this file is included,
# cppjieba_SOURCE_DIR is available and the include paths below are exposed as
# an INTERFACE target so any target that links cppjieba_iface picks them up.

include(FetchContent)

FetchContent_Declare(
  cppjieba
  GIT_REPOSITORY https://github.com/yanyiwu/cppjieba.git
  GIT_TAG        v5.4.0
  GIT_SUBMODULES "deps/limonp"
)
FetchContent_MakeAvailable(cppjieba)

# cppjieba ships no CMakeLists.txt that creates a proper target, so we create a
# thin INTERFACE library so targets can simply link against it.
if(NOT TARGET cppjieba_iface)
  add_library(cppjieba_iface INTERFACE)
  target_include_directories(cppjieba_iface INTERFACE
    ${cppjieba_SOURCE_DIR}/include
    ${cppjieba_SOURCE_DIR}/deps/limonp/include
  )
endif()

# Expose the dict directory path as a CMake cache variable so test code can
# reference it without hard-coding the FetchContent build path.
set(CPPJIEBA_DICT_DIR "${cppjieba_SOURCE_DIR}/dict"
  CACHE PATH "Path to cppjieba dict files" FORCE)
