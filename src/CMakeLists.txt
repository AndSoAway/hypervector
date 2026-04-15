# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.10)

project(hypervec VERSION 1.0.0)

# =============================================================================
# Minimal HNSW-only HyperVec Build Configuration
# =============================================================================

set(HYPERVEC_SRC
  core/index.cpp
  core/hypervec_exception.cpp
  core/id_selector.cpp
  index/flat/index_flat.cpp
  index/flat/index_flat_codes.cpp
  index/hnsw/index_hnsw.cpp
  index/hnsw/hnsw.cpp
  index/hnsw/visited_table.cpp
  index/idmap/index_id_map.cpp
  factory/clone_index.cpp
  search/aux_index_structures.cpp
  io/index_read.cpp
  io/index_write.cpp
  io/io.cpp
  io/mapped_io.cpp
  utils/structures/heap.cpp
  utils/structures/random.cpp
  utils/structures/sorting.cpp
  utils/structures/partitioning.cpp
  utils/utils.cpp
  utils/distances/distances.cpp
  utils/distances/distances_simd.cpp
  utils/distances/extra_distances.cpp
  utils/simd/simd_levels.cpp
)

set(HYPERVEC_HEADERS
  include/core/index.h
  include/core/metric_type.h
  include/core/hypervec_assert.h
  include/core/hypervec_exception.h
  include/core/platform_macros.h
  include/core/distance_computer.h
  include/core/id_selector.h
  include/core/maybe_owned_vector.h
  include/index/flat/index_flat.h
  include/index/flat/index_flat_codes.h
  include/index/hnsw/index_hnsw.h
  include/index/hnsw/hnsw.h
  include/index/hnsw/visited_table.h
  include/index/idmap/index_id_map.h
  include/factory/clone_index.h
  include/search/aux_index_structures.h
  include/io/io.h
  include/io/io_macros.h
  include/io/index_read_utils.h
  include/io/mapped_io.h
  include/utils/utils.h
  include/utils/structures/heap.h
  include/utils/structures/random.h
  include/utils/structures/sorting.h
  include/utils/distances/distances.h
  include/utils/simd/simd_levels.h
)

# Export HYPERVEC_HEADERS variable to parent scope.
set(HYPERVEC_HEADERS ${HYPERVEC_HEADERS} PARENT_SCOPE)

add_library(hypervec ${HYPERVEC_SRC})

# Handle new include path: #include <core/Index.h> etc.
target_include_directories(hypervec PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

set_target_properties(hypervec PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(hypervec PRIVATE HYPERVEC_MAIN_LIB)
endif()

if(WIN32)
  set_target_properties(hypervec PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
endif()

string(FIND "${CMAKE_CXX_FLAGS}" "FINTEGER" finteger_idx)
if (${finteger_idx} EQUAL -1)
  target_compile_definitions(hypervec PRIVATE FINTEGER=int)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(hypervec PUBLIC OpenMP::OpenMP_CXX)

find_package(BLAS REQUIRED)
target_link_libraries(hypervec PRIVATE ${BLAS_LIBRARIES})

find_package(LAPACK REQUIRED)
target_link_libraries(hypervec PRIVATE ${LAPACK_LIBRARIES})

install(TARGETS hypervec
  EXPORT hypervec-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

foreach(header ${HYPERVEC_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${dir}
  )
endforeach()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/hypervec-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${CMAKE_SOURCE_DIR}/cmake/hypervec-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/hypervec-config.cmake
  COPYONLY)
install(FILES ${PROJECT_BINARY_DIR}/cmake/hypervec-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/hypervec-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/hypervec
)

install(EXPORT hypervec-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/hypervec
)
