/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */



#include <utils/common/interrupt_callback.h>

#include <algorithm>

namespace hypervec {

std::unique_ptr<InterruptCallback> InterruptCallback::instance;

std::mutex InterruptCallback::lock;

void InterruptCallback::ClearInstance() {
  delete instance.release();
}

void InterruptCallback::check() {
  if (!instance.get()) {
    return;
  }
  if (instance->WantInterrupt()) {
    HYPERVEC_THROW_MSG("computation interrupted");
  }
}

bool InterruptCallback::IsInterrupted() {
  if (!instance.get()) {
    return false;
  }
  std::lock_guard<std::mutex> guard(lock);
  return instance->WantInterrupt();
}

size_t InterruptCallback::GetPeriodHint(size_t flops) {
  if (!instance.get()) {
    return (size_t)1 << 30;  // never check
  }
  // for 10M flops, it is reasonable to check once every 10 iterations
  return std::max((size_t)10 * 10 * 1000 * 1000 / (flops + 1), (size_t)1);
}

void TimeoutCallback::SetTimeout(double timeout_in_seconds) {
  timeout = timeout_in_seconds;
  start = std::chrono::steady_clock::now();
}

bool TimeoutCallback::WantInterrupt() {
  if (timeout == 0) {
    return false;
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float, std::milli> duration = end - start;
  float elapsed_in_seconds = duration.count() / 1000.0;
  if (elapsed_in_seconds > timeout) {
    timeout = 0;
    return true;
  }
  return false;
}

void TimeoutCallback::Reset(double timeout_in_seconds) {
  auto tc(new hypervec::TimeoutCallback());
  hypervec::InterruptCallback::instance.reset(tc);
  tc->SetTimeout(timeout_in_seconds);
}

}  // namespace hypervec