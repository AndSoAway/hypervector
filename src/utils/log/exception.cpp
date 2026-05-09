/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/log/exception.h>

#include <sstream>

namespace hypervec {

void handleExceptions(
  std::vector<std::pair<int, std::exception_ptr>>& exceptions) {
  if (exceptions.size() == 1) {
    std::rethrow_exception(exceptions.front().second);

  } else if (exceptions.size() > 1) {
    std::stringstream ss;

    for (auto& p : exceptions) {
      try {
        std::rethrow_exception(p.second);
      } catch (std::exception& ex) {
        if (ex.what()) {
          ss << "Exception thrown from index " << p.first << ": " << ex.what()
             << "\n";
        } else {
          ss << "Unknown exception thrown from index " << p.first << "\n";
        }
      } catch (...) {
        ss << "Unknown exception thrown from index " << p.first << "\n";
      }
    }

    throw HypervecException(ss.str());
  }
}

}  // namespace hypervec
