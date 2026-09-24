/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/structures/sparse_row.h>

#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <utils/algo/bm25/bm25.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

// Build an owning packed byte buffer from a sorted list of SparseElements.
MaybeOwnedVector<uint8_t> pack_elements(const std::vector<SparseElement>& elems) {
  MaybeOwnedVector<uint8_t> buf(elems.size() * sizeof(SparseElement));
  if (!elems.empty()) {
    std::memcpy(buf.data(), elems.data(), buf.byte_size());
  }
  return buf;
}

}  // namespace

SparseRow::SparseRow(std::vector<uint32_t> indices, std::vector<float> values) {
  HYPERVEC_THROW_IF_NOT_FMT(
    indices.size() == values.size(),
    "SparseRow: indices size %zd != values size %zd", indices.size(),
    values.size());
  // Pack into SparseElement[] then sort by index ascending to satisfy the
  // layout contract.
  std::vector<SparseElement> elems(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    elems[i].index = indices[i];
    elems[i].value = values[i];
  }
  std::sort(elems.begin(), elems.end(),
            [](const SparseElement& a, const SparseElement& b) {
              return a.index < b.index;
            });
  // The layout contract requires strictly ascending (hence unique) indices;
  // dot()/dim() are undefined on duplicates.  set()/read never produce them,
  // but a caller could, so reject duplicates at this entry point.
  for (size_t i = 1; i < elems.size(); ++i) {
    HYPERVEC_THROW_IF_NOT_FMT(
      elems[i - 1].index != elems[i].index,
      "SparseRow: duplicate index %u", elems[i].index);
  }
  buf_ = pack_elements(elems);
}

SparseRow SparseRow::create_view(
  void* address, size_t nnz,
  const std::shared_ptr<MaybeOwnedVectorOwner>& owner) {
  SparseRow row;
  row.buf_ = MaybeOwnedVector<uint8_t>::create_view(
    address, nnz * sizeof(SparseElement), owner);
  return row;
}

uint32_t SparseRow::dim() const {
  const size_t n = nnz();
  if (n == 0) {
    return 0;
  }
  // Entries are sorted by index ascending, so the last one holds the max.
  // Note: if the max index is UINT32_MAX this +1 wraps to 0.  That index is
  // ~4.3e9 term ids, far beyond any realistic vocabulary, so we accept the
  // theoretical wrap rather than widen the return type.
  return index_at(n - 1) + 1;
}

void SparseRow::set(uint32_t index, float value) {
  // A view is read-only regardless of whether this call overwrites an existing
  // entry (in-place mutation) or inserts a new one (would COW the shared bytes
  // into a private buffer, silently detaching from the source).  Reject both
  // up front so the read-only contract holds on every path, not just overwrite.
  HYPERVEC_ASSERT_MSG(buf_.is_owned,
                      "SparseRow::set cannot be performed on a viewed row");
  const size_t n = nnz();
  const SparseElement* elems = data();
  // Binary search for the insertion point by index.
  size_t lo = 0;
  size_t hi = n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (elems[mid].index < index) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < n && elems[lo].index == index) {
    // Overwrite existing value in place.
    SparseElement* mutable_elems =
      reinterpret_cast<SparseElement*>(buf_.data());
    mutable_elems[lo].value = value;
    return;
  }
  // Insert a new element at position `lo`, preserving order.
  std::vector<SparseElement> elems_copy(n + 1);
  for (size_t i = 0; i < lo; ++i) {
    elems_copy[i] = elems[i];
  }
  elems_copy[lo].index = index;
  elems_copy[lo].value = value;
  for (size_t i = lo; i < n; ++i) {
    elems_copy[i + 1] = elems[i];
  }
  buf_ = pack_elements(elems_copy);
}

float SparseRow::dot(const SparseRow& other, DocValueComputer computer,
                     float other_extra) const {
  const size_t n_a = nnz();
  const size_t n_b = other.nnz();
  const SparseElement* a = data();
  const SparseElement* b = other.data();
  float acc = 0.0f;
  size_t i = 0;
  size_t j = 0;
  // Two-pointer merge over the shared indices (both rows are sorted).
  while (i < n_a && j < n_b) {
    if (a[i].index < b[j].index) {
      ++i;
    } else if (a[i].index > b[j].index) {
      ++j;
    } else {
      const float other_val =
        computer ? computer(b[j].value, other_extra) : b[j].value;
      acc += a[i].value * other_val;
      ++i;
      ++j;
    }
  }
  return acc;
}

float SparseRow::dot_bm25(const SparseRow& query_idf, const BM25Params& params,
                          float doc_len) const {
  // Convenience member wrapper over the free BM25Score function (bm25.h),
  // which is the single source of truth for BM25 scoring.  `*this` is the
  // document (TF values); query_idf holds the query IDF weights.
  return BM25Score(query_idf, *this, params, doc_len);
}

void write_sparse_row(const SparseRow& row, IOWriter* f) {
  uint32_t nnz = static_cast<uint32_t>(row.nnz());
  WRITE1(nnz);
  // Interleave (index, value) exactly as Python struct.pack("<If", ...) does;
  // do NOT use WRITEVECTOR (8-byte size_t prefix + non-interleaved arrays).
  for (size_t i = 0; i < row.nnz(); ++i) {
    uint32_t idx = row.index_at(i);
    float val = row.value_at(i);
    WRITE1(idx);
    WRITE1(val);
  }
}

SparseRow read_sparse_row(IOReader* f) {
  uint32_t nnz = 0;
  READ1(nnz);
  // Guard against corrupt / hostile nnz: each entry is 8 bytes on the wire.
  HYPERVEC_THROW_IF_NOT(static_cast<size_t>(nnz) <
                        (get_deserialization_vector_byte_limit() / 8));
  std::vector<SparseElement> elems(nnz);
  for (uint32_t i = 0; i < nnz; ++i) {
    uint32_t idx = 0;
    float val = 0.0f;
    READ1(idx);
    READ1(val);
    elems[i].index = idx;
    elems[i].value = val;
    // The wire format is strictly ascending by index (write_sparse_row emits
    // sorted rows).  Enforce it on read so a corrupt / hostile blob with
    // out-of-order or duplicate indices is rejected rather than silently
    // producing a row that violates the sorted-unique layout contract.
    if (i > 0) {
      HYPERVEC_THROW_IF_NOT_FMT(
        elems[i - 1].index < elems[i].index,
        "read_sparse_row: indices not strictly ascending at %u (%u >= %u)", i,
        elems[i - 1].index, elems[i].index);
    }
  }
  SparseRow row;
  row.buf_ = pack_elements(elems);
  return row;
}

}  // namespace hypervec
