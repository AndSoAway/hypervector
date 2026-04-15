# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

# HNSW-only demos

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(OMP_LIB "gomp")

add_executable(demo_hnsw_indexing EXCLUDE_FROM_ALL demo_hnsw_indexing.cpp)
target_link_libraries(demo_hnsw_indexing PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})

add_executable(demo_hnsw_graph_search EXCLUDE_FROM_ALL demo_hnsw_graph_search.cpp)
target_link_libraries(demo_hnsw_graph_search PRIVATE hypervec OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OMP_LIB})
