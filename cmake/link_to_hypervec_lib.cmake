# @lint-ignore-every LICENSELINT
# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

function(link_to_hypervec_lib target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Cannot link unknown target '${target}' to HyperVec")
  endif()

  if(NOT TARGET hypervec)
    message(FATAL_ERROR "The HyperVec library target 'hypervec' is unavailable")
  endif()

  target_link_libraries(${target} PRIVATE hypervec)
endfunction()
