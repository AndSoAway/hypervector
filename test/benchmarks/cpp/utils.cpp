/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"
namespace hypervec::perf_tests {
std::map<std::string, hypervec::ScalarQuantizer::QuantizerType> sq_types() {
    static std::map<std::string, hypervec::ScalarQuantizer::QuantizerType>
            sq_types = {
                    {"QT_8bit", hypervec::ScalarQuantizer::QT_8bit},
                    {"QT_4bit", hypervec::ScalarQuantizer::QT_4bit},
                    {"QT_8bit_uniform",
                     hypervec::ScalarQuantizer::QT_8bit_uniform},
                    {"QT_4bit_uniform",
                     hypervec::ScalarQuantizer::QT_4bit_uniform},
                    {"QT_fp16", hypervec::ScalarQuantizer::QT_fp16},
                    {"QT_8bit_direct", hypervec::ScalarQuantizer::QT_8bit_direct},
                    {"QT_6bit", hypervec::ScalarQuantizer::QT_6bit},
                    {"QT_bf16", hypervec::ScalarQuantizer::QT_bf16},
                    {"QT_8bit_direct_signed",
                     hypervec::ScalarQuantizer::QT_8bit_direct_signed}};
    return sq_types;
}
} // namespace hypervec::perf_tests
