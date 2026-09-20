# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

add_executable(hypervec_eval main.cpp)
target_link_libraries(hypervec_eval PRIVATE hypervec)

add_executable(hypervec_dataset dataset_main.cpp)
target_link_libraries(hypervec_dataset PRIVATE hypervec)

install(TARGETS hypervec_eval hypervec_dataset
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TESTING)
  add_test(NAME HypervecEval.Help COMMAND hypervec_eval --help)
  add_test(NAME HypervecDataset.Help COMMAND hypervec_dataset --help)
endif()
