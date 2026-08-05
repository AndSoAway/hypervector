/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#pragma once

#include <utils/common/platform_macros.h>

#include <chrono>
#include <memory>
#include <mutex>

namespace hypervec {

/***********************************************************
 * Interrupt callback
 ***********************************************************/

struct HYPERVEC_API InterruptCallback {
  virtual bool WantInterrupt() = 0;
  virtual ~InterruptCallback() {}

  // lock that protects concurrent calls to IsInterrupted
  static std::mutex lock;

  static std::unique_ptr<InterruptCallback> instance;

  static void ClearInstance();

  /** check if:
   * - an interrupt callback is set
   * - the callback returns true
   * if this is the case, then throw an exception. Should not be called
   * from multiple threads.
   */
  static void check();

  /// same as check() but return true if is interrupted instead of
  /// throwing. Can be called from multiple threads.
  static bool IsInterrupted();

  /** assuming each iteration takes a certain number of flops, what
   * is a reasonable interval to check for interrupts?
   */
  static size_t GetPeriodHint(size_t flops);
};

struct TimeoutCallback : InterruptCallback {
  std::chrono::time_point<std::chrono::steady_clock> start;
  double timeout;
  bool WantInterrupt() override;
  void SetTimeout(double timeout_in_seconds);
  static void Reset(double timeout_in_seconds);
};

}  // namespace hypervec