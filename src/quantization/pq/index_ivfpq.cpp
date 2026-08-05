/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <quantization/pq/index_ivfpq.h>

#include <invlists/inverted_lists.h>
#include <utils/distances/distances.h>
#include <utils/log/assert.h>
#include <utils/selector/id_selector.h>
#include <utils/structures/heap.h>

#include <cinttypes>
#include <cstring>
#include <vector>

namespace hypervec {

// ===========================================================================
// Construction
// ===========================================================================

IndexIVFPQ::IndexIVFPQ() : IndexIVF(0, 0, 0, kMetricL2) {}

IndexIVFPQ::IndexIVFPQ(idx_t d, idx_t nlist, idx_t M, int nbits,
                       MetricType metric)
  : IndexIVF(d, nlist,
             /*code_size=*/(static_cast<size_t>(M) * static_cast<size_t>(nbits)
                            + 7) / 8,
             metric)
  , pq(d, M, nbits) {
  HYPERVEC_THROW_IF_NOT_FMT(
    metric == kMetricL2,
    "IndexIVFPQ: T1+T2 supports kMetricL2 only, got metric=%d",
    static_cast<int>(metric));
}

// ===========================================================================
// Training
// ===========================================================================

void IndexIVFPQ::Train(idx_t n, const float* x) {
  // 1. Train the coarse quantizer on raw x via the IVF base. This populates
  //    `centroids` and sets is_trained = true.
  IndexIVF::Train(n, x);

  if (by_residual) {
    // 2a. Compute residuals: assign each x to its nearest centroid, then
    //     subtract.
    std::vector<float> coarse_dis(static_cast<size_t>(n));
    std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
    FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

    std::vector<float> residuals(static_cast<size_t>(n) * d);
    for (idx_t i = 0; i < n; i++) {
      const float* c = centroids.data() +
                       centroid_ids[static_cast<size_t>(i)] * d;
      const float* xi = x + i * d;
      float* ri = residuals.data() + i * d;
      for (idx_t j = 0; j < d; j++) {
        ri[j] = xi[j] - c[j];
      }
    }

    pq.Train(n, residuals.data());
  } else {
    // 2b. Train PQ on raw x.
    pq.Train(n, x);
  }

  // 3. T2: build the per-(coarse cell, m, k) cache.
  if (use_precomputed_table != 0) {
    PrecomputeTable();
  }
}

// ===========================================================================
// EncodeVectors / AddWithIds
// ===========================================================================

void IndexIVFPQ::EncodeVectors(idx_t n, const float* x, uint8_t* codes) const {
  if (!by_residual) {
    pq.ComputeCodes(n, x, codes);
    return;
  }

  // by_residual=true with no list ids in scope: recompute assignments. Hot
  // adders should use AddWithIds (which we override to avoid this).
  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  std::vector<float> residuals(static_cast<size_t>(n) * d);
  for (idx_t i = 0; i < n; i++) {
    const float* c = centroids.data() +
                     centroid_ids[static_cast<size_t>(i)] * d;
    const float* xi = x + i * d;
    float* ri = residuals.data() + i * d;
    for (idx_t j = 0; j < d; j++) {
      ri[j] = xi[j] - c[j];
    }
  }

  pq.ComputeCodes(n, residuals.data(), codes);
}

void IndexIVFPQ::AddWithIds(idx_t n, const float* x, const idx_t* xids) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }

  std::vector<float> coarse_dis(static_cast<size_t>(n));
  std::vector<idx_t> centroid_ids(static_cast<size_t>(n));
  FindNearestCentroids(n, x, 1, coarse_dis.data(), centroid_ids.data());

  std::vector<uint8_t> codes(static_cast<size_t>(n) * pq.code_size);

  if (by_residual) {
    // Encode residuals; reuse a single per-vector buffer to keep allocation
    // out of the hot path. We could batch via pq.ComputeCodes after building
    // the full residual matrix, but the matrix would be n*d floats which can
    // dwarf the codes themselves; per-vector keeps memory bounded.
    std::vector<float> residual(static_cast<size_t>(d));
    for (idx_t i = 0; i < n; i++) {
      const float* c = centroids.data() +
                       centroid_ids[static_cast<size_t>(i)] * d;
      const float* xi = x + i * d;
      for (idx_t j = 0; j < d; j++) {
        residual[static_cast<size_t>(j)] = xi[j] - c[j];
      }
      pq.ComputeCode(residual.data(),
                     codes.data() + static_cast<size_t>(i) * pq.code_size);
    }
  } else {
    pq.ComputeCodes(n, x, codes.data());
  }

  for (idx_t i = 0; i < n; i++) {
    const idx_t id = (xids != nullptr) ? xids[i] : n_total + i;
    const idx_t list_no = centroid_ids[static_cast<size_t>(i)];
    invlists->add_entry(static_cast<size_t>(list_no), id,
                        codes.data() +
                          static_cast<size_t>(i) * pq.code_size);
  }

  n_total += n;
}

// ===========================================================================
// SearchPreassigned
// ===========================================================================

void IndexIVFPQ::SearchPreassigned(idx_t n, const float* x, idx_t k,
                                   const idx_t* list_ids,
                                   const float* centroid_dis,
                                   float* distances, idx_t* labels,
                                   idx_t nprobe_actual,
                                   const IDSelector* sel) const {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT(k > 0);

  const size_t d_sz = static_cast<size_t>(d);
  const size_t table_sz = static_cast<size_t>(pq.M) * pq.ksub;
  const bool precomputed = (use_precomputed_table != 0);
  const idx_t pq_M = pq.M;
  const idx_t pq_ksub = pq.ksub;

#pragma omp parallel
  {
    std::vector<float> residual_query(d_sz);
    std::vector<float> dis_table(table_sz);
    // For the precomputed-table path, the per-query inner-product table
    // <q, p_{m,k}>.
    std::vector<float> qip_table(precomputed ? table_sz : 0);

#pragma omp for
    for (idx_t qi = 0; qi < n; qi++) {
      const float* xq = x + qi * d;
      float* heap_dis = distances + qi * k;
      idx_t* heap_ids = labels + qi * k;
      heap_heapify<CMax<float, idx_t>>(k, heap_dis, heap_ids);

      // T2 fast path: precompute <q, p_{m,k}> once per query, valid for all
      // probed lists. For the basic path we leave qip_table unused.
      if (precomputed) {
        for (idx_t m = 0; m < pq_M; m++) {
          fvec_inner_products_ny(qip_table.data() + m * pq_ksub,
                                 xq + m * pq.dsub,
                                 pq.GetCentroids(m, 0),
                                 static_cast<size_t>(pq.dsub),
                                 static_cast<size_t>(pq_ksub));
        }
      }

      // For the basic path, !by_residual reuses the same dis_table for every
      // probe of this query; build it once outside the probe loop.
      if (!precomputed && !by_residual) {
        pq.ComputeDistanceTable(xq, dis_table.data());
      }

      for (idx_t pi = 0; pi < nprobe_actual; pi++) {
        const idx_t list_no =
          list_ids[static_cast<size_t>(qi) * nprobe_actual + pi];
        if (list_no < 0) {
          continue;
        }
        const size_t list_sz =
          invlists->list_size(static_cast<size_t>(list_no));
        if (list_sz == 0) {
          continue;
        }

        // ---- build the (M*ksub) ADC table for this (query, list) pair ----
        if (precomputed) {
          // table_i[m, k] = precomputed[(i*M+m)*ksub+k] - 2 * qip_table[m, k]
          const float* pre = precomputed_table.data() +
                             list_no * pq_M * pq_ksub;
          for (size_t t = 0; t < table_sz; t++) {
            dis_table[t] = pre[t] - 2.0f * qip_table[t];
          }
        } else if (by_residual) {
          const float* c = centroids.data() + list_no * d;
          for (idx_t j = 0; j < d; j++) {
            residual_query[static_cast<size_t>(j)] = xq[j] - c[j];
          }
          pq.ComputeDistanceTable(residual_query.data(), dis_table.data());
        }
        // else: !by_residual && !precomputed — dis_table already built once
        // per query above.

        // The constant offset added to every code in this list. The
        // precomputed-table identity gives:
        //   ||q - (c_i + p)||² = coarse_dis_i
        //                      + Σ_m (precomputed[i,m,k] - 2*qip[m,k])
        //                      = coarse_dis_i + Σ_m table_i[m, code[m]]
        // For the basic path the per-list dis_table already encodes the full
        // L2² so list_offset is 0.
        float list_offset = 0.0f;
        if (precomputed) {
          list_offset = centroid_dis[static_cast<size_t>(qi) * nprobe_actual +
                                     pi];
        }

        // ---- scan the inverted list ----
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
          const float dis = list_offset +
                            pq.ApplyDistanceTable(
                              dis_table.data(),
                              codes_p + j * pq.code_size);
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

// ===========================================================================
// Reconstruct
// ===========================================================================

void IndexIVFPQ::Reconstruct(idx_t key, float* recons) const {
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
        pq.Decode(codes.get() + j * pq.code_size, recons);
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
  HYPERVEC_THROW_FMT(
    "IndexIVFPQ::Reconstruct: key %" PRId64 " not found", key);
}

// ===========================================================================
// PrecomputeTable (T2)
// ===========================================================================

void IndexIVFPQ::PrecomputeTable() {
  HYPERVEC_THROW_IF_NOT(is_trained);
  HYPERVEC_THROW_IF_NOT(pq.is_trained);
  HYPERVEC_THROW_IF_NOT_MSG(
    by_residual,
    "IndexIVFPQ::PrecomputeTable assumes by_residual=true; "
    "the L2 expansion only telescopes when PQ encodes residuals");

  const idx_t pq_M = pq.M;
  const idx_t pq_ksub = pq.ksub;
  const size_t table_per_cell = static_cast<size_t>(pq_M) * pq_ksub;

  precomputed_table.assign(static_cast<size_t>(nlist) * table_per_cell, 0.0f);

  // Term 1: ||p_{m,k}||² — only depends on PQ centroids, compute once.
  std::vector<float> r_norms(table_per_cell);
  for (idx_t m = 0; m < pq_M; m++) {
    fvec_norms_L2sqr(r_norms.data() + m * pq_ksub, pq.GetCentroids(m, 0),
                     static_cast<size_t>(pq.dsub),
                     static_cast<size_t>(pq_ksub));
  }

  // Term 2: <c_i, p_{m,k}> per coarse cell i, subquantizer m, code k.
  // For each (i, m), this is fvec_inner_products_ny over ksub centroids.
  // Parallelise across coarse cells.
#pragma omp parallel for if (nlist > 1)
  for (idx_t i = 0; i < nlist; i++) {
    const float* c_i = centroids.data() + i * d;
    float* tab = precomputed_table.data() + i * table_per_cell;
    for (idx_t m = 0; m < pq_M; m++) {
      fvec_inner_products_ny(tab + m * pq_ksub, c_i + m * pq.dsub,
                             pq.GetCentroids(m, 0),
                             static_cast<size_t>(pq.dsub),
                             static_cast<size_t>(pq_ksub));
    }
    // Combine: precomputed[i, m, k] = ||p_{m,k}||² + 2 * <c_i, p_{m,k}>
    for (size_t t = 0; t < table_per_cell; t++) {
      tab[t] = r_norms[t] + 2.0f * tab[t];
    }
  }
}

}  // namespace hypervec
