# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

"""Python mirror of the C++ TermDictionary (src/utils/structures/term_dictionary.cpp).

Maps opaque string terms (words / tokens) to dense unsigned 32-bit ids assigned
in first-seen order (0, 1, 2, ...).  This ordering and the serialized byte
layout are a cross-language contract shared with the C++ side, so the same
stream of terms produces identical ids and identical bytes on both sides.

Serialized layout (little-endian, tightly packed, no padding):

    [uint32 n_terms][ (uint32 term_len, bytes term_utf8) * n_terms ]

Entries are written in ascending term-id order (first-seen order); the i-th
term has id i, so no explicit id is stored.
"""

from __future__ import annotations

import struct


class TermDictionary:
    def __init__(self) -> None:
        self._term_to_id: dict[str, int] = {}
        self._id_to_term: list[str] = []

    def get_or_add(self, term: str) -> int:
        """Return the id of ``term``, allocating the next id if unseen."""
        existing = self._term_to_id.get(term)
        if existing is not None:
            return existing
        new_id = len(self._id_to_term)
        self._id_to_term.append(term)
        self._term_to_id[term] = new_id
        return new_id

    def contains(self, term: str) -> bool:
        return term in self._term_to_id

    def id_of(self, term: str) -> int:
        if term not in self._term_to_id:
            raise KeyError(f"TermDictionary: unknown term {term!r}")
        return self._term_to_id[term]

    def term_of(self, term_id: int) -> str:
        if term_id < 0 or term_id >= len(self._id_to_term):
            raise IndexError(f"TermDictionary: term id {term_id} out of range (size {len(self._id_to_term)})")
        return self._id_to_term[term_id]

    def __len__(self) -> int:
        return len(self._id_to_term)

    def items(self):
        """Yield (term_id, term) pairs in ascending id order."""
        return enumerate(self._id_to_term)

    def serialize(self) -> bytes:
        """Encode to the cross-language byte layout (contract B)."""
        parts = [struct.pack("<I", len(self._id_to_term))]
        for term in self._id_to_term:
            encoded = term.encode("utf-8")
            parts.append(struct.pack("<I", len(encoded)))
            parts.append(encoded)
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "TermDictionary":
        """Decode bytes written by serialize() (or the C++ writer)."""
        dictionary = cls()
        n_terms = struct.unpack_from("<I", data, 0)[0]
        offset = 4
        for _ in range(n_terms):
            (term_len,) = struct.unpack_from("<I", data, offset)
            offset += 4
            term = data[offset : offset + term_len].decode("utf-8")
            offset += term_len
            # First-seen order == ascending id, so this rebuilds the same map.
            dictionary._id_to_term.append(term)
            dictionary._term_to_id[term] = len(dictionary._id_to_term) - 1
        return dictionary
