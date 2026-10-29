# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

# cppjieba is vendored as header-only sources.  The upstream CMakeLists.txt is
# deliberately NOT executed: it force-overrides CMAKE_INSTALL_PREFIX, which
# would leak into the parent project's install layout.
add_library(cppjieba_iface INTERFACE)

# Only build-tree include paths: the installed package never needs to compile
# against cppjieba, because hypervec consumes it privately (pImpl).
target_include_directories(cppjieba_iface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/limonp/include>
)

# hypervec is part of the hypervec-targets export set, so its private link
# dependency must be exported too.  Give it a project-prefixed export name to
# avoid colliding with a consumer's own target (the package has no namespace).
set_target_properties(cppjieba_iface PROPERTIES EXPORT_NAME hypervec_cppjieba)
install(TARGETS cppjieba_iface EXPORT hypervec-targets)

# Dict files are runtime data, not build inputs: ship them next to the library
# so an installed build can segment Chinese text with default parameters.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dict/
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/hypervec/dict
)
