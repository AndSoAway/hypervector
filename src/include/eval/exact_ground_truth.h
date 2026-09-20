/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#ifndef SRC_INCLUDE_EVAL_EXACT_GROUND_TRUTH_H_
#define SRC_INCLUDE_EVAL_EXACT_GROUND_TRUTH_H_

#include <eval/semantic_metric.h>

namespace hypervec {

/** Compute deterministic exact top-k neighbor IDs.
 *
 * Cosine evaluation expects both datasets to be L2-normalized already and
 * uses exact inner-product search. This keeps ground truth and evaluated
 * indexes on precisely the same prepared float values.
 */
IntegerVectorDataset ComputeExactGroundTruth(const FloatVectorDataset& base,
                                             const FloatVectorDataset& queries,
                                             idx_t k, GroundTruthMetric metric);

}  // namespace hypervec

#endif  // SRC_INCLUDE_EVAL_EXACT_GROUND_TRUTH_H_
