/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */



#include <utils/common/range_search_result.h>

#include <algorithm>
#include <cstring>

namespace hypervec {

/***********************************************************************
 * RangeSearchResult
 ***********************************************************************/

RangeSearchResult::RangeSearchResult(size_t nq, bool alloc_lims) : nq(nq) {
  if (alloc_lims) {
    lims = new size_t[nq + 1];
    memset(lims, 0, sizeof(*lims) * (nq + 1));
  } else {
    lims = nullptr;
  }
  labels = nullptr;
  distances = nullptr;
  buffer_size = 1024 * 256;
}

/// called when lims contains the nb of elements result entries
/// for each query
void RangeSearchResult::DoAllocation() {
  // works only if all the partial results are aggregated
  // simultaneously
  HYPERVEC_THROW_IF_NOT(labels == nullptr && distances == nullptr);
  size_t ofs = 0;
  for (int i = 0; i < nq; i++) {
    size_t n = lims[i];
    lims[i] = ofs;
    ofs += n;
  }
  lims[nq] = ofs;
  labels = new idx_t[ofs];
  distances = new float[ofs];
}

RangeSearchResult::~RangeSearchResult() {
  delete[] labels;
  delete[] distances;
  delete[] lims;
}

/***********************************************************************
 * RangeSearchPartialResult
 ***********************************************************************/

void RangeQueryResult::Add(float dis, idx_t id) {
  nres++;
  pres->Add(id, dis);
}

RangeSearchPartialResult::RangeSearchPartialResult(RangeSearchResult* res_in)
  : BufferList(res_in->buffer_size), res(res_in) {}

/// begin a new result
RangeQueryResult& RangeSearchPartialResult::NewResult(idx_t qno) {
  RangeQueryResult qres = {qno, 0, this};
  queries.push_back(qres);
  return queries.back();
}

void RangeSearchPartialResult::Finalize() {
  SetLims();
#pragma omp barrier

#pragma omp single
  res->DoAllocation();

#pragma omp barrier
  CopyResult();
}

/// called by RangeSearch before DoAllocation
void RangeSearchPartialResult::SetLims() {
  for (int i = 0; i < queries.size(); i++) {
    RangeQueryResult& qres = queries[i];
    res->lims[qres.qno] = qres.nres;
  }
}

/// called by RangeSearch after DoAllocation
void RangeSearchPartialResult::CopyResult(bool incremental) {
  size_t ofs = 0;
  for (int i = 0; i < queries.size(); i++) {
    RangeQueryResult& qres = queries[i];

    CopyRange(ofs, qres.nres, res->labels + res->lims[qres.qno],
               res->distances + res->lims[qres.qno]);
    if (incremental) {
      res->lims[qres.qno] += qres.nres;
    }
    ofs += qres.nres;
  }
}

void RangeSearchPartialResult::merge(
  std::vector<RangeSearchPartialResult*>& partial_results, bool do_delete) {
  int npres = partial_results.size();
  if (npres == 0) {
    return;
  }
  RangeSearchResult* result = partial_results[0]->res;
  size_t nx = result->nq;

  // count
  for (const RangeSearchPartialResult* pres : partial_results) {
    if (!pres) {
      continue;
    }
    for (const RangeQueryResult& qres : pres->queries) {
      result->lims[qres.qno] += qres.nres;
    }
  }
  result->DoAllocation();
  for (int j = 0; j < npres; j++) {
    if (!partial_results[j]) {
      continue;
    }
    partial_results[j]->CopyResult(true);
    if (do_delete) {
      delete partial_results[j];
      partial_results[j] = nullptr;
    }
  }

  // Reset the limits
  for (size_t i = nx; i > 0; i--) {
    result->lims[i] = result->lims[i - 1];
  }
  result->lims[0] = 0;
}

}  // namespace hypervec