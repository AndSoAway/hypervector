/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/ivf/index_ivf_flat.h>

#include <invlists/inverted_lists.h>
#include <utils/distances/distances.h>
#include <utils/distances/metric_type.h>
#include <utils/log/assert.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <cinttypes>
#include <cstring>
#include <limits>

namespace hypervec {

IndexIVFFlat::IndexIVFFlat(idx_t d, idx_t nlist, MetricType metric)
  : IndexIVF(d, nlist, (size_t)d * sizeof(float), metric) {}

void IndexIVFFlat::EncodeVectors(idx_t n, const float* x,
                                 uint8_t* codes) const {
  if (n > 0) {
    memcpy(codes, x, (size_t)n * (size_t)d * sizeof(float));
  }
}

void IndexIVFFlat::SearchPreassigned(idx_t n, const float* x, idx_t k,
                                     const idx_t* list_ids,
                                     const float* /* centroid_dis */,
                                     float* distances, idx_t* labels,
                                     idx_t nprobe_actual,
                                     const IDSelector* sel) const {
  const bool sim = IsSimilarityMetric(metric_type);
  const size_t dim = (size_t)d;

#pragma omp parallel for schedule(dynamic, 1) if (n > 1)
  for (idx_t qi = 0; qi < n; qi++) {
    const float* xq = x + qi * dim;
    float* heap_dis = distances + qi * k;
    idx_t* heap_ids = labels + qi * k;

    if (sim) {
      heap_heapify<CMin<float, idx_t>>(k, heap_dis, heap_ids);
    } else {
      heap_heapify<CMax<float, idx_t>>(k, heap_dis, heap_ids);
    }

    for (idx_t pi = 0; pi < nprobe_actual; pi++) {
      idx_t list_no = list_ids[qi * nprobe_actual + pi];
      if (list_no < 0) {
        continue;
      }
      size_t list_sz = invlists->list_size((size_t)list_no);
      if (list_sz == 0) {
        continue;
      }

      InvertedLists::ScopedCodes codes(invlists, (size_t)list_no);
      InvertedLists::ScopedIds ids(invlists, (size_t)list_no);
      const float* vecs = (const float*)codes.get();
      const idx_t* id_ptr = ids.get();

      if (sim) {
        float threshold = heap_dis[0];
        for (size_t j = 0; j < list_sz; j++) {
          if (sel && !sel->IsMember(id_ptr[j])) {
            continue;
          }
          float dis = fvec_inner_product(xq, vecs + j * dim, dim);
          if (CMin<float, idx_t>::cmp(threshold, dis)) {
            heap_replace_top<CMin<float, idx_t>>(k, heap_dis, heap_ids, dis,
                                                 id_ptr[j]);
            threshold = heap_dis[0];
          }
        }
      } else {
        float threshold = heap_dis[0];
        for (size_t j = 0; j < list_sz; j++) {
          if (sel && !sel->IsMember(id_ptr[j])) {
            continue;
          }
          float dis = fvec_L2sqr(xq, vecs + j * dim, dim);
          if (CMax<float, idx_t>::cmp(threshold, dis)) {
            heap_replace_top<CMax<float, idx_t>>(k, heap_dis, heap_ids, dis,
                                                 id_ptr[j]);
            threshold = heap_dis[0];
          }
        }
      }
    }

    if (sim) {
      heap_reorder<CMin<float, idx_t>>(k, heap_dis, heap_ids);
    } else {
      heap_reorder<CMax<float, idx_t>>(k, heap_dis, heap_ids);
    }
  }
}

void IndexIVFFlat::Reconstruct(idx_t key, float* recons) const {
  const size_t dim = (size_t)d;
  for (size_t list_no = 0; list_no < (size_t)nlist; list_no++) {
    size_t list_sz = invlists->list_size(list_no);
    if (list_sz == 0) {
      continue;
    }
    InvertedLists::ScopedIds ids(invlists, list_no);
    const idx_t* id_ptr = ids.get();
    for (size_t j = 0; j < list_sz; j++) {
      if (id_ptr[j] == key) {
        InvertedLists::ScopedCodes codes(invlists, list_no);
        memcpy(recons, (const float*)codes.get() + j * dim,
               dim * sizeof(float));
        return;
      }
    }
  }
  HYPERVEC_THROW_FMT("IndexIVFFlat::Reconstruct: key %" PRId64 " not found",
                     key);
}

}  // namespace hypervec
