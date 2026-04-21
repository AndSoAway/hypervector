/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <index/flat/code_packer.h>
#include <utils/distances/distance_computer.h>
#include <utils/log/exception.h>
#include <utils/selector/id_selector.h>
#include <index/flat/index_flat_codes.h>
#include <search/aux_index_structures.h>
#include <search/result_handler.h>
#include <utils/distances/extra_distances.h>

namespace hypervec {

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
  : Index(d, metric), code_size(code_size) {}

IndexFlatCodes::IndexFlatCodes() : code_size(0) {}

void IndexFlatCodes::Add(idx_t n, const float* x) {
  HYPERVEC_THROW_IF_NOT(is_trained);
  if (n == 0) {
    return;
  }
  codes.resize((n_total + n) * code_size);
  SaEncode(n, x, codes.data() + (n_total * code_size));
  n_total += n;
}

void IndexFlatCodes::AddSaCodes(idx_t n, const uint8_t* codes_in,
                                  const idx_t* /* xids */) {
  codes.resize((n_total + n) * code_size);
  memcpy(codes.data() + (n_total * code_size), codes_in, n * code_size);
  n_total += n;
}

void IndexFlatCodes::Reset() {
  codes.clear();
  n_total = 0;
}

size_t IndexFlatCodes::SaCodeSize() const {
  return code_size;
}

size_t IndexFlatCodes::RemoveIds(const IDSelector& sel) {
  idx_t j = 0;
  for (idx_t i = 0; i < n_total; i++) {
    if (sel.IsMember(i)) {
      // should be removed
    } else {
      if (i > j) {
        memmove(&codes[code_size * j], &codes[code_size * i], code_size);
      }
      j++;
    }
  }
  size_t nremove = n_total - j;
  if (nremove > 0) {
    n_total = j;
    codes.resize(n_total * code_size);
  }
  return nremove;
}

void IndexFlatCodes::ReconstructN(idx_t i0, idx_t ni, float* recons) const {
  HYPERVEC_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= n_total));
  SaDecode(ni, codes.data() + i0 * code_size, recons);
}

void IndexFlatCodes::Reconstruct(idx_t key, float* recons) const {
  ReconstructN(key, 1, recons);
}

void IndexFlatCodes::CheckCompatibleForMerge(const Index& otherIndex) const {
  // minimal sanity checks
  const IndexFlatCodes* other =
    dynamic_cast<const IndexFlatCodes*>(&otherIndex);
  HYPERVEC_THROW_IF_NOT(other);
  HYPERVEC_THROW_IF_NOT(other->d == d);
  HYPERVEC_THROW_IF_NOT(other->code_size == code_size);
  HYPERVEC_THROW_IF_NOT_MSG(typeid(*this) == typeid(*other),
                            "can only merge indexes of the same type");
}

void IndexFlatCodes::MergeFrom(Index& otherIndex, idx_t add_id) {
  HYPERVEC_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatCodes index");
  CheckCompatibleForMerge(otherIndex);
  IndexFlatCodes* other = static_cast<IndexFlatCodes*>(&otherIndex);
  codes.resize((n_total + other->n_total) * code_size);
  memcpy(codes.data() + (n_total * code_size), other->codes.data(),
         other->n_total * code_size);
  n_total += other->n_total;
  other->Reset();
}

CodePacker* IndexFlatCodes::GetCodePacker() const {
  return new CodePackerFlat(code_size);
}

void IndexFlatCodes::PermuteEntries(const idx_t* perm) {
  MaybeOwnedVector<uint8_t> new_codes(codes.size());

  for (idx_t i = 0; i < n_total; i++) {
    memcpy(new_codes.data() + i * code_size, codes.data() + perm[i] * code_size,
           code_size);
  }
  std::swap(codes, new_codes);
}

namespace {

template <class VD>
struct GenericFlatCodesDistanceComputer : FlatCodesDistanceComputer {
  const IndexFlatCodes& codec;
  const VD vd;
  // temp buffers
  std::vector<uint8_t> code_buffer;
  std::vector<float> vec_buffer;
  const float* query = nullptr;

  GenericFlatCodesDistanceComputer(const IndexFlatCodes* codec, const VD& vd)
    : FlatCodesDistanceComputer(codec->codes.data(), codec->code_size)
    , codec(*codec)
    , vd(vd)
    , code_buffer(codec->code_size * 4)
    , vec_buffer(codec->d * 4) {}

  void SetQuery(const float* x) override {
    query = x;
  }

  float operator()(idx_t i) override {
    codec.SaDecode(1, codes + i * code_size, vec_buffer.data());
    return vd(query, vec_buffer.data());
  }

  float distance_to_code(const uint8_t* code) override {
    codec.SaDecode(1, code, vec_buffer.data());
    return vd(query, vec_buffer.data());
  }

  float symmetric_dis(idx_t i, idx_t j) override {
    codec.SaDecode(1, codes + i * code_size, vec_buffer.data());
    codec.SaDecode(1, codes + j * code_size, vec_buffer.data() + vd.d);
    return vd(vec_buffer.data(), vec_buffer.data() + vd.d);
  }

  void distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2,
                         const idx_t idx3, float& dis0, float& dis1,
                         float& dis2, float& dis3) override {
    uint8_t* cp = code_buffer.data();
    for (idx_t i : {idx0, idx1, idx2, idx3}) {
      memcpy(cp, codes + i * code_size, code_size);
      cp += code_size;
    }
    // potential benefit is if batch decoding is more efficient than 1 by 1
    // decoding
    codec.SaDecode(4, code_buffer.data(), vec_buffer.data());
    dis0 = vd(query, vec_buffer.data());
    dis1 = vd(query, vec_buffer.data() + vd.d);
    dis2 = vd(query, vec_buffer.data() + 2 * vd.d);
    dis3 = vd(query, vec_buffer.data() + 3 * vd.d);
  }
};

template <class BlockResultHandler>
struct Run_search_with_decompress {
  using T = void;

  template <class VectorDistance>
  void f(VectorDistance& vd, const IndexFlatCodes* index_ptr, const float* xq,
         BlockResultHandler& res) {
    // Note that there seems to be a clang (?) bug that "sometimes" passes
    // the const Index & parameters by value, so to be on the safe side,
    // it's better to use pointers.
    const IndexFlatCodes& index = *index_ptr;
    size_t n_total = index.n_total;
    using SingleResultHandler =
      typename BlockResultHandler::SingleResultHandler;
    using DC = GenericFlatCodesDistanceComputer<VectorDistance>;
#pragma omp parallel  // if (res.nq > 100)
    {
      std::unique_ptr<DC> dc(new DC(&index, vd));
      SingleResultHandler resi(res);
#pragma omp for
      for (int64_t q = 0; q < res.nq; q++) {
        resi.begin(q);
        dc->SetQuery(xq + vd.d * q);
        for (size_t i = 0; i < n_total; i++) {
          if (res.is_in_selection(i)) {
            float dis = (*dc)(i);
            resi.AddResult(dis, i);
          }
        }
        resi.end();
      }
    }
  }
};

struct Run_search_with_decompress_res {
  using T = void;

  template <class BlockResultHandler>
  void f(BlockResultHandler& res, const IndexFlatCodes* index,
         const float* xq) {
    with_VectorDistance(index->d, index->metric_type, index->metric_arg,
                        [&](auto vd) {
                          Run_search_with_decompress<BlockResultHandler> r;
                          r.template f<decltype(vd)>(vd, index, xq, res);
                        });
  }
};

}  // anonymous namespace

FlatCodesDistanceComputer* IndexFlatCodes::GetFlatCodesDistanceComputer()
  const {
  return with_VectorDistance(
    d, metric_type, metric_arg, [&](auto vd) -> FlatCodesDistanceComputer* {
      return new GenericFlatCodesDistanceComputer<decltype(vd)>(this, vd);
    });
}

void IndexFlatCodes::Search(idx_t n, const float* x, idx_t k, float* distances,
                            idx_t* labels,
                            const SearchParameters* params) const {
  Run_search_with_decompress_res r;
  const IDSelector* sel = params ? params->sel : nullptr;
  dispatch_knn_ResultHandler(n, distances, labels, k, metric_type, sel, r, this,
                             x);
}

void IndexFlatCodes::RangeSearch(idx_t n, const float* x, float radius,
                                  RangeSearchResult* result,
                                  const SearchParameters* params) const {
  const IDSelector* sel = params ? params->sel : nullptr;
  Run_search_with_decompress_res r;
  dispatch_range_ResultHandler(result, radius, metric_type, sel, r, this, x);
}

void IndexFlatCodes::Search1(const float* x, ResultHandler& handler,
                             SearchParameters* params) const {
  const IDSelector* sel = params ? params->sel : nullptr;
  Run_search_with_decompress_res r;
  if (sel) {
    if (IsSimilarityMetric(metric_type)) {
      SingleQueryBlockResultHandler<CMin<float, idx_t>, true> res(handler, sel);
      r.f(res, this, x);
    } else {
      SingleQueryBlockResultHandler<CMax<float, idx_t>, true> res(handler, sel);
      r.f(res, this, x);
    }
  } else {
    if (IsSimilarityMetric(metric_type)) {
      SingleQueryBlockResultHandler<CMin<float, idx_t>, false> res(handler);
      r.f(res, this, x);
    } else {
      SingleQueryBlockResultHandler<CMax<float, idx_t>, false> res(handler);
      r.f(res, this, x);
    }
  }
}

}  // namespace hypervec
