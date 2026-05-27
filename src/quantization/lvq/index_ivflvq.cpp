/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/lvq/index_ivflvq.h>

#include <invlists/inverted_lists.h>
#include <utils/log/assert.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <cinttypes>
#include <vector>

namespace hypervec {

IndexIVFLVQ::IndexIVFLVQ() : IndexIVF(0, 0, 0, kMetricL2) {}

IndexIVFLVQ::IndexIVFLVQ(idx_t d, idx_t nlist, idx_t nlocal, int nbits,
                         MetricType metric)
  : IndexIVF(d, nlist, 0, metric)
  , lvq(d, nlocal, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(
    metric == kMetricL2, "IndexIVFLVQ: supports kMetricL2 only, got metric=%d",
    static_cast<int>(metric));
  delete invlists;
  invlists = new ArrayInvertedLists(static_cast<size_t>(nlist), lvq.code_size);
  own_invlists = true;
}

void IndexIVFLVQ::Train(idx_t n, const float* x) {
  IndexIVF::Train(n, x);
  if (by_residual) {
    std::vector<float> coarse_dis(static_cast<size_t>(n));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
    FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

    std::vector<float> residuals(static_cast<size_t>(n) * d);
    for (idx_t i = 0; i < n; i++) {
      const float* c = centroids.data() + centroid_ids[i] * d;
      const float* xi = x + i * d;
      float* ri = residuals.data() + i * d;
      for (idx_t j = 0; j < d; j++) {
        ri[j] = xi[j] - c[j];
      }
    }
    lvq.Train(n, residuals.data());
  } else {
    lvq.Train(n, x);
  }
}

void IndexIVFLVQ::EncodeVectors(idx_t n, const float* x,
                                uint8_t* codes) const {
  if (!by_residual) {
    lvq.ComputeCodes(n, x, codes);
    return;
  }

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  std::vector<float> residuals(static_cast<size_t>(n) * d);
  for (idx_t i = 0; i < n; i++) {
    const float* c = centroids.data() + centroid_ids[i] * d;
    const float* xi = x + i * d;
    float* ri = residuals.data() + i * d;
    for (idx_t j = 0; j < d; j++) {
      ri[j] = xi[j] - c[j];
    }
  }
  lvq.ComputeCodes(n, residuals.data(), codes);
}

void IndexIVFLVQ::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  std::vector<uint8_t> codes(static_cast<size_t>(n) * lvq.code_size);
  if (by_residual) {
    std::vector<float> residual(static_cast<size_t>(d));
    for (idx_t i = 0; i < n; i++) {
      const float* c = centroids.data() + centroid_ids[i] * d;
      const float* xi = x + i * d;
      for (idx_t j = 0; j < d; j++) {
        residual[static_cast<size_t>(j)] = xi[j] - c[j];
      }
      lvq.ComputeCode(residual.data(),
                      codes.data() + static_cast<size_t>(i) * lvq.code_size);
    }
  } else {
    lvq.ComputeCodes(n, x, codes.data());
  }

  for (idx_t i = 0; i < n; i++) {
    const idx_t id = (xids != nullptr) ? xids[i] : n_total + i;
    invlists->add_entry(static_cast<size_t>(centroid_ids[i]), id,
                        codes.data() + static_cast<size_t>(i) * lvq.code_size);
  }
  n_total += n;
}

void IndexIVFLVQ::SearchPreassigned(idx_t n, const float* x, idx_t k,
                                    const idx_t* list_ids,
                                    const float* /*centroid_dis*/,
                                    float* distances, idx_t* labels,
                                    idx_t nprobe_actual,
                                    const IDSelector* sel) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT(k > 0);

  const size_t table_sz = static_cast<size_t>(lvq.nlocal) * lvq.ksub;
#pragma omp parallel
  {
    std::vector<float> residual_query(static_cast<size_t>(d));
    std::vector<float> dis_table(table_sz);
#pragma omp for
    for (idx_t qi = 0; qi < n; qi++) {
      const float* xq = x + qi * d;
      float* heap_dis = distances + qi * k;
      idx_t* heap_ids = labels + qi * k;
      heap_heapify<CMax<float, idx_t>>(k, heap_dis, heap_ids);

      if (!by_residual) {
        lvq.ComputeDistanceTable(xq, dis_table.data());
      }

      for (idx_t pi = 0; pi < nprobe_actual; pi++) {
        const idx_t list_no = list_ids[qi * nprobe_actual + pi];
        if (list_no < 0) {
          continue;
        }
        const size_t list_sz =
          invlists->list_size(static_cast<size_t>(list_no));
        if (list_sz == 0) {
          continue;
        }

        if (by_residual) {
          const float* c = centroids.data() + list_no * d;
          for (idx_t j = 0; j < d; j++) {
            residual_query[static_cast<size_t>(j)] = xq[j] - c[j];
          }
          lvq.ComputeDistanceTable(residual_query.data(), dis_table.data());
        }

        InvertedLists::ScopedCodes scoped_codes(invlists,
                                                static_cast<size_t>(list_no));
        InvertedLists::ScopedIds scoped_ids(invlists,
                                            static_cast<size_t>(list_no));
        const uint8_t* codes_p = scoped_codes.get();
        const idx_t* ids_p = scoped_ids.get();

        float threshold = heap_dis[0];
        for (size_t j = 0; j < list_sz; j++) {
          if (sel && !sel->IsMember(ids_p[j])) {
            continue;
          }
          const float dis = lvq.ApplyDistanceTable(
            dis_table.data(), codes_p + j * lvq.code_size);
          if (CMax<float, idx_t>::cmp(threshold, dis)) {
            heap_replace_top<CMax<float, idx_t>>(k, heap_dis, heap_ids, dis,
                                                 ids_p[j]);
            threshold = heap_dis[0];
          }
        }
      }
      heap_reorder<CMax<float, idx_t>>(k, heap_dis, heap_ids);
    }
  }
}

void IndexIVFLVQ::Reconstruct(idx_t key, float* recons) const {
  for (size_t list_no = 0; list_no < static_cast<size_t>(nlist); list_no++) {
    const size_t sz = invlists->list_size(list_no);
    if (sz == 0) {
      continue;
    }
    InvertedLists::ScopedIds ids(invlists, list_no);
    const idx_t* id_ptr = ids.get();
    for (size_t j = 0; j < sz; j++) {
      if (id_ptr[j] == key) {
        InvertedLists::ScopedCodes codes(invlists, list_no);
        lvq.Decode(codes.get() + j * lvq.code_size, recons);
        if (by_residual) {
          const float* c = centroids.data() +
                           static_cast<idx_t>(list_no) * d;
          for (idx_t l = 0; l < d; l++) {
            recons[l] += c[l];
          }
        }
        return;
      }
    }
  }
  HYPERVEC_THROW_FMT("IndexIVFLVQ::Reconstruct: key %" PRId64 " not found",
                     key);
}

}  // namespace hypervec
