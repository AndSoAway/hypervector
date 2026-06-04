# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

# HNSW-only demos

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(OMP_LIB "")
if(NOT MSVC)
  set(OMP_LIB "gomp")
endif()

add_executable(demo_hnsw_indexing EXCLUDE_FROM_ALL demo_hnsw_indexing.cpp)
target_link_libraries(demo_hnsw_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_hnsw_graph_search EXCLUDE_FROM_ALL demo_hnsw_graph_search.cpp)
target_link_libraries(demo_hnsw_graph_search PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_ivf_indexing EXCLUDE_FROM_ALL demo_ivf_indexing.cpp)
target_link_libraries(demo_ivf_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
target_link_libraries(demo_ivfpq_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_hnsw_pq_indexing EXCLUDE_FROM_ALL demo_hnsw_pq_indexing.cpp)
target_link_libraries(demo_hnsw_pq_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_lvq_indexing EXCLUDE_FROM_ALL demo_lvq_indexing.cpp)
target_link_libraries(demo_lvq_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_ivflvq_indexing EXCLUDE_FROM_ALL demo_ivflvq_indexing.cpp)
target_link_libraries(demo_ivflvq_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_hnsw_lvq_indexing EXCLUDE_FROM_ALL demo_hnsw_lvq_indexing.cpp)
target_link_libraries(demo_hnsw_lvq_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})
