# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

set(HYPERVEC_TEST_SRC
  test_exact_ground_truth.cpp
  test_heap.cpp
  test_diskann_searcher.cpp
  test_diskann_storage.cpp
  test_index_diskann.cpp
  test_graph_pruning_validation.cpp
  test_graph_searcher.cpp
  test_graph_storage.cpp
  test_nn_descent_builder.cpp
  test_nsg_builder.cpp
  test_index_nsg.cpp
  test_nsw_builder.cpp
  test_paged_graph_storage.cpp
  test_vamana_builder.cpp
  test_index_vamana.cpp
  test_index_nsw.cpp
  test_index_rerank.cpp
  test_hnsw.cpp
  test_hnsw_graph_storage.cpp
  test_inverted_list_scanner.cpp
  test_index_build.cpp
  test_index_capabilities.cpp
  test_index_clone.cpp
  test_index_io_registry.cpp
  test_index_factory.cpp
  test_index_flat_correctness.cpp
  test_search_parameters_factory.cpp
  test_search_evaluator.cpp
  test_vector_dataset.cpp
  test_index_lsh.cpp
  test_index_pre_transform.cpp
  test_index_hnsw_correctness.cpp
  test_index_id_map_correctness.cpp
  test_persistence_validation.cpp
  test_callback.cpp
  test_utils.cpp
  test_mmap.cpp
  test_page_cache.cpp
  test_random_access_io.cpp
  test_factory_tools.cpp
  test_distances_simd.cpp
  test_omp_threads.cpp
  test_partitioning.cpp
  test_simd_levels.cpp
  test_simdlib.cpp
  test_kmeans.cpp
  test_pq.cpp
  test_quantizer_protocol.cpp
  test_rabitq_quantizer.cpp
  test_index_ivf_rabitq.cpp
  test_opq_matrix.cpp
  test_vector_transform.cpp
  test_index_pq.cpp
  test_index_ivf_flat.cpp
  test_index_ivf_correctness.cpp
  test_index_ivfpq.cpp
  test_index_hnsw_pq.cpp
  test_lvq.cpp
  test_sparse_row.cpp
  test_term_dictionary.cpp
  test_sparse_inverted_index.cpp
)

add_executable(hypervec_test ${HYPERVEC_TEST_SRC})

include(${CMAKE_SOURCE_DIR}/cmake/link_to_hypervec_lib.cmake)

link_to_hypervec_lib(hypervec_test)

find_package(GTest CONFIG QUIET)
if(NOT TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
  find_package(GTest QUIET)
endif()

if(NOT TARGET GTest::gtest_main AND
   NOT TARGET GTest::Main AND
   HYPERVEC_FETCH_DEPS)
  message(STATUS "GoogleTest was not found locally; fetching release-1.12.1")
  include(FetchContent)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850)
  FetchContent_MakeAvailable(googletest)
endif()

if(TARGET GTest::gtest_main)
  set(HYPERVEC_GTEST_MAIN_TARGET GTest::gtest_main)
elseif(TARGET GTest::Main)
  set(HYPERVEC_GTEST_MAIN_TARGET GTest::Main)
else()
  message(FATAL_ERROR
    "GoogleTest was not found. Install its CMake package or explicitly allow "
    "the pinned download with -DHYPERVEC_FETCH_DEPS=ON.")
endif()

set(OMP_LIB "")
if(MINGW)
  set(OMP_LIB "gomp")
endif()

target_link_libraries(hypervec_test PRIVATE
  ${OMP_LIB}
  ${HYPERVEC_GTEST_MAIN_TARGET}
)

if(WIN32)
  target_compile_definitions(hypervec_test PRIVATE HYPERVEC_MAIN_LIB)
endif()

include(GoogleTest)
gtest_discover_tests(hypervec_test
  PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
