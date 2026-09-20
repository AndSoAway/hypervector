# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

add_executable(hypervec_eval main.cpp)
target_link_libraries(hypervec_eval PRIVATE hypervec)

add_executable(hypervec_dataset dataset_main.cpp)
target_link_libraries(hypervec_dataset PRIVATE hypervec)

add_executable(hypervec_ground_truth ground_truth_main.cpp)
target_link_libraries(hypervec_ground_truth PRIVATE hypervec)

add_executable(hypervec_build index_build_main.cpp)
target_link_libraries(hypervec_build PRIVATE hypervec)

install(PROGRAMS convert_hdf5.py
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME hypervec_convert_hdf5
)

install(TARGETS
  hypervec_eval
  hypervec_dataset
  hypervec_ground_truth
  hypervec_build
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TESTING)
  add_test(NAME HypervecEval.Help COMMAND hypervec_eval --help)
  add_test(NAME HypervecDataset.Help COMMAND hypervec_dataset --help)
  add_test(NAME HypervecGroundTruth.Help COMMAND hypervec_ground_truth --help)
  add_test(NAME HypervecBuild.Help COMMAND hypervec_build --help)
  add_test(NAME HypervecBuild.ListIndexes COMMAND hypervec_build --list-indexes)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME HypervecEval.Pipeline
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/test_eval_pipeline.py
              $<TARGET_FILE:hypervec_ground_truth>
              $<TARGET_FILE:hypervec_build>
              $<TARGET_FILE:hypervec_eval>)
    add_test(NAME HypervecHdf5Converter
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_hdf5.py
              ${CMAKE_CURRENT_SOURCE_DIR}/convert_hdf5.py)
  endif()
endif()
