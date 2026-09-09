# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

set(HYPERVEC_EXAMPLE_SRC
  demo_hnsw_indexing.cpp
  demo_hnsw_graph_search.cpp
  demo_ivf_indexing.cpp
  demo_ivfpq_indexing.cpp
  demo_hnsw_pq_indexing.cpp
  demo_lvq_indexing.cpp
  demo_ivflvq_indexing.cpp
  demo_hnsw_lvq_indexing.cpp
)

foreach(example_source IN LISTS HYPERVEC_EXAMPLE_SRC)
  get_filename_component(example_target ${example_source} NAME_WE)
  add_executable(${example_target} ${example_source})
  target_link_libraries(${example_target} PRIVATE hypervec)
endforeach()
