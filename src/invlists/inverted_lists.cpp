/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <invlists/inverted_lists.h>

#include <utils/log/assert.h>

#include <cstring>
#include <numeric>

namespace hypervec {

// ---------------------------------------------------------------------------
// InvertedListsIterator
// ---------------------------------------------------------------------------

InvertedListsIterator::~InvertedListsIterator() = default;

// ---------------------------------------------------------------------------
// InvertedLists
// ---------------------------------------------------------------------------

InvertedLists::InvertedLists(size_t nlist, size_t code_size)
  : nlist(nlist), code_size(code_size) {}

InvertedLists::~InvertedLists() = default;

void InvertedLists::release_codes(size_t, const uint8_t*) const {}

void InvertedLists::release_ids(size_t, const idx_t*) const {}

idx_t InvertedLists::get_single_id(size_t list_no, size_t offset) const {
  HYPERVEC_THROW_IF_NOT(offset < list_size(list_no));
  return get_ids(list_no)[offset];
}

const uint8_t* InvertedLists::get_single_code(size_t list_no,
                                               size_t offset) const {
  HYPERVEC_THROW_IF_NOT(offset < list_size(list_no));
  return get_codes(list_no) + offset * code_size;
}

void InvertedLists::prefetch_lists(const idx_t*, int) const {}

bool InvertedLists::is_empty(size_t list_no, void*) const {
  return list_size(list_no) == 0;
}

InvertedListsIterator* InvertedLists::get_iterator(size_t, void*) const {
  HYPERVEC_THROW_MSG("get_iterator not implemented for this InvertedLists");
}

size_t InvertedLists::add_entry(size_t list_no, idx_t theid,
                                const uint8_t* code, void*) {
  return add_entries(list_no, 1, &theid, code);
}

void InvertedLists::update_entry(size_t list_no, size_t offset, idx_t id,
                                 const uint8_t* code) {
  update_entries(list_no, offset, 1, &id, code);
}

void InvertedLists::Reset() {
  for (size_t i = 0; i < nlist; i++) {
    resize(i, 0);
  }
}

void InvertedLists::MergeFrom(InvertedLists* oivf, size_t add_id) {
  HYPERVEC_THROW_IF_NOT(nlist == oivf->nlist);
  HYPERVEC_THROW_IF_NOT(code_size == oivf->code_size);
  for (size_t list_no = 0; list_no < nlist; list_no++) {
    size_t sz = oivf->list_size(list_no);
    if (sz == 0) {
      continue;
    }
    const uint8_t* codes = oivf->get_codes(list_no);
    const idx_t* ids = oivf->get_ids(list_no);
    std::vector<idx_t> new_ids(sz);
    for (size_t i = 0; i < sz; i++) {
      new_ids[i] = ids[i] + (idx_t)add_id;
    }
    add_entries(list_no, sz, new_ids.data(), codes);
    oivf->release_codes(list_no, codes);
    oivf->release_ids(list_no, ids);
    oivf->resize(list_no, 0);
  }
}

size_t InvertedLists::copy_subset_to(InvertedLists& other,
                                      subset_type_t subset_type, idx_t a1,
                                      idx_t a2) const {
  size_t n_copied = 0;
  for (size_t list_no = 0; list_no < nlist; list_no++) {
    size_t sz = list_size(list_no);
    if (sz == 0) {
      continue;
    }
    const uint8_t* codes = get_codes(list_no);
    const idx_t* ids = get_ids(list_no);
    for (size_t i = 0; i < sz; i++) {
      bool keep = false;
      switch (subset_type) {
        case SUBSET_TYPE_ID_RANGE:
          keep = ids[i] >= a1 && ids[i] < a2;
          break;
        case SUBSET_TYPE_ID_MOD:
          keep = ids[i] % a1 == a2;
          break;
        default:
          HYPERVEC_THROW_MSG("copy_subset_to: unsupported subset_type");
      }
      if (keep) {
        other.add_entries(list_no, 1, ids + i, codes + i * code_size);
        n_copied++;
      }
    }
    release_codes(list_no, codes);
    release_ids(list_no, ids);
  }
  return n_copied;
}

size_t InvertedLists::compute_ntotal() const {
  size_t n = 0;
  for (size_t i = 0; i < nlist; i++) {
    n += list_size(i);
  }
  return n;
}

// ---------------------------------------------------------------------------
// ArrayInvertedLists
// ---------------------------------------------------------------------------

ArrayInvertedLists::ArrayInvertedLists(size_t nlist, size_t code_size)
  : InvertedLists(nlist, code_size)
  , codes(nlist)
  , ids(nlist) {}

ArrayInvertedLists::~ArrayInvertedLists() = default;

size_t ArrayInvertedLists::list_size(size_t list_no) const {
  return ids[list_no].size();
}

const uint8_t* ArrayInvertedLists::get_codes(size_t list_no) const {
  return codes[list_no].data();
}

const idx_t* ArrayInvertedLists::get_ids(size_t list_no) const {
  return ids[list_no].data();
}

size_t ArrayInvertedLists::add_entries(size_t list_no, size_t n_entry,
                                        const idx_t* new_ids,
                                        const uint8_t* new_codes) {
  if (n_entry == 0) {
    return 0;
  }
  size_t prev_size = ids[list_no].size();
  ids[list_no].resize(prev_size + n_entry);
  memcpy(ids[list_no].data() + prev_size, new_ids, n_entry * sizeof(idx_t));
  codes[list_no].resize((prev_size + n_entry) * code_size);
  memcpy(codes[list_no].data() + prev_size * code_size, new_codes,
         n_entry * code_size);
  return prev_size;
}

void ArrayInvertedLists::update_entries(size_t list_no, size_t offset,
                                         size_t n_entry, const idx_t* new_ids,
                                         const uint8_t* new_codes) {
  HYPERVEC_THROW_IF_NOT(offset + n_entry <= ids[list_no].size());
  memcpy(ids[list_no].data() + offset, new_ids, n_entry * sizeof(idx_t));
  memcpy(codes[list_no].data() + offset * code_size, new_codes,
         n_entry * code_size);
}

void ArrayInvertedLists::resize(size_t list_no, size_t new_size) {
  ids[list_no].resize(new_size);
  codes[list_no].resize(new_size * code_size);
}

void ArrayInvertedLists::permute_invlists(const idx_t* map) {
  std::vector<MaybeOwnedVector<uint8_t>> new_codes(nlist);
  std::vector<MaybeOwnedVector<idx_t>> new_ids(nlist);
  for (size_t i = 0; i < nlist; i++) {
    new_codes[i] = std::move(codes[(size_t)map[i]]);
    new_ids[i] = std::move(ids[(size_t)map[i]]);
  }
  codes = std::move(new_codes);
  ids = std::move(new_ids);
}

bool ArrayInvertedLists::is_empty(size_t list_no, void*) const {
  return ids[list_no].size() == 0;
}

}  // namespace hypervec
