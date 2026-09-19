# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

# CPU core sources: indexes, quantizers, persistence, and shared utilities.

set(HYPERVEC_SRC
  index/index.cpp
  index/index_factory.cpp
  index/ivf/index_ivf.cpp
  index/ivf/index_ivf_flat.cpp
  index/ivf/inverted_list_scanner.cpp
  index/flat/index_flat.cpp
  index/flat/index_flat_codes.cpp
  index/graph/graph_searcher.cpp
  index/graph/graph_storage.cpp
  index/graph/graph_validation.cpp
  index/graph/neighbor_pruner.cpp
  index/graph/nsw_builder.cpp
  index/graph/visited_table.cpp
  index/hnsw/index_hnsw.cpp
  index/hnsw/index_hnsw_pq.cpp
  index/hnsw/index_hnsw_lvq.cpp
  index/hnsw/hnsw.cpp
  index/idmap/index_id_map.cpp
  index/nsw/index_nsw.cpp
  invlists/inverted_lists.cpp
  persistence/index_clone.cpp
  utils/common/interrupt_callback.cpp
  utils/common/buffer_list.cpp
  utils/common/range_search_result.cpp
  utils/log/exception.cpp
  persistence/index_read.cpp
  persistence/index_write.cpp
  persistence/io.cpp
  persistence/mapped_io.cpp
  utils/structures/heap.cpp
  utils/structures/random.cpp
  utils/structures/sorting.cpp
  utils/structures/partitioning.cpp
  utils/structures/sparse_row.cpp
  utils/structures/term_dictionary.cpp
  utils/algo/kmeans/kmeans.cpp
  quantization/pq/pq.cpp
  quantization/pq/pq_distance_computer.cpp
  quantization/pq/pq_quantizer_adapter.cpp
  quantization/pq/index_pq.cpp
  quantization/pq/index_ivfpq.cpp
  quantization/lvq/lvq.cpp
  quantization/lvq/lvq_distance_computer.cpp
  quantization/lvq/lvq_quantizer_adapter.cpp
  quantization/lvq/index_lvq.cpp
  quantization/lvq/index_ivflvq.cpp
  quantization/quantizer.cpp
  utils/utils.cpp
  utils/distances/distances.cpp
  utils/distances/distances_fused.cpp
  utils/distances/distances_simd.cpp
  utils/distances/extra_distances.cpp
  utils/selector/id_selector.cpp
  utils/simd/simd_levels.cpp
)

file(GLOB_RECURSE HYPERVEC_PUBLIC_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)
set(HYPERVEC_PUBLIC_HEADERS "${HYPERVEC_PUBLIC_HEADERS}" PARENT_SCOPE)

add_library(hypervec ${HYPERVEC_SRC})
add_library(hypervec::hypervec ALIAS hypervec)

target_include_directories(hypervec
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

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
if(NOT TARGET BLAS::BLAS)
  add_library(BLAS::BLAS INTERFACE IMPORTED)
  set_property(TARGET BLAS::BLAS PROPERTY
    INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
endif()
target_link_libraries(hypervec PRIVATE BLAS::BLAS)

find_package(LAPACK REQUIRED)
if(NOT TARGET LAPACK::LAPACK)
  add_library(LAPACK::LAPACK INTERFACE IMPORTED)
  set_property(TARGET LAPACK::LAPACK PROPERTY
    INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}")
endif()
target_link_libraries(hypervec PRIVATE LAPACK::LAPACK)

install(TARGETS hypervec
  EXPORT hypervec-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN ".gitkeep" EXCLUDE
)

include(CMakePackageConfigHelpers)
set(HYPERVEC_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/hypervec")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/hypervec-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hypervec-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hypervec-config.cmake"
  INSTALL_DESTINATION ${HYPERVEC_INSTALL_CMAKEDIR}
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/hypervec-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/hypervec-config-version.cmake"
  DESTINATION ${HYPERVEC_INSTALL_CMAKEDIR}
)

install(EXPORT hypervec-targets
  FILE hypervec-targets.cmake
  NAMESPACE hypervec::
  DESTINATION ${HYPERVEC_INSTALL_CMAKEDIR}
)
