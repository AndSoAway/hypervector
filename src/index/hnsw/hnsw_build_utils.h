/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <omp.h>

#include <cstddef>
#include <vector>

namespace hypervec {

class OmpLockArray {
 public:
  explicit OmpLockArray(size_t size) : locks_(size) {
    for (auto& lock : locks_) {
      omp_init_lock(&lock);
    }
  }

  OmpLockArray(const OmpLockArray&) = delete;
  OmpLockArray& operator=(const OmpLockArray&) = delete;

  ~OmpLockArray() {
    for (auto& lock : locks_) {
      omp_destroy_lock(&lock);
    }
  }

  std::vector<omp_lock_t>& Get() { return locks_; }

 private:
  std::vector<omp_lock_t> locks_;
};

}  // namespace hypervec
