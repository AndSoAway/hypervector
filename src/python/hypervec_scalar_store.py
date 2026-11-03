# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .hypervec_term_dictionary import TermDictionary
except ImportError:  # pragma: no cover - supports direct file loading in tests
    sys.path.insert(0, str(Path(__file__).parent))
    from hypervec_term_dictionary import TermDictionary


# Lazy, cached probe for an optional C++ SparseRow codec.  Returns the
# hypervec.SparseRow class if a usable compiled extension is importable, else
# None.  Robust by design: a crashing / absent / stale `hypervec` must NEVER
# break the pure-Python fallback.  hypervec/loader.py can call sys.exit(1) on an
# unsupported CPU (raising SystemExit, which is NOT an Exception subclass), so
# we catch BaseException — this is load-bearing, not over-broad.  Tests
# monkeypatch this function or reset the module globals to force either path.
_CPP_SPARSE_ROW = None       # cached class-or-None after first probe
_CPP_SPARSE_PROBED = False   # whether the probe has already run


def _cpp_sparse_codec():
    """Return hypervec.SparseRow (a usable C++ codec) or None.  Cached."""
    global _CPP_SPARSE_ROW, _CPP_SPARSE_PROBED
    if _CPP_SPARSE_PROBED:
        return _CPP_SPARSE_ROW
    _CPP_SPARSE_PROBED = True
    try:
        import hypervec  # may raise ImportError / DLL load error / SystemExit
        row_cls = getattr(hypervec, "SparseRow", None)
        if row_cls is None:
            _CPP_SPARSE_ROW = None
            return _CPP_SPARSE_ROW
        # Smoke-test the surface once so a stale/partial binding is rejected.
        probe = row_cls({0: 1.0})
        if hasattr(probe, "serialize") and hasattr(probe, "to_dict"):
            _CPP_SPARSE_ROW = row_cls
        else:
            _CPP_SPARSE_ROW = None
    except BaseException:  # noqa: BLE001 - ImportError/AttributeError/SystemExit/DLL
        _CPP_SPARSE_ROW = None
    return _CPP_SPARSE_ROW


class ScalarStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA secure_delete=ON")
            self._local.conn = conn
        return self._local.conn

    @staticmethod
    def _table(collection_name: str) -> str:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in collection_name)
        return f"docs_{safe}"

    @classmethod
    def _staging_table(cls, collection_name: str) -> str:
        """Name of the transient import-staging table for a collection.

        Rows are loaded here first so the live docs_<name> table is never
        dropped until an atomic commit rename swaps staging into place.
        """
        return cls._table(collection_name) + "__import"

    @staticmethod
    def _encode_sparse(sparse: dict) -> bytes:
        """Encode a sparse vector (dict[int, float]) to binary BLOB.

        Prefers the C++ SparseRow codec (hypervec.SparseRow.serialize) when the
        compiled extension is available, falling back to the pure-Python struct
        implementation otherwise.  Both produce the identical byte layout
        (cross-language contract A): 4-byte LE nnz, then nnz*(uint32 idx +
        float32 val), sorted by index.  Aligns with Milvus/knowhere SparseRow.
        """
        row_cls = _cpp_sparse_codec()
        if row_cls is not None:
            try:
                return bytes(row_cls(dict(sparse)).serialize())
            except Exception:
                pass  # any C++-path failure -> fall through to pure Python
        return ScalarStore._encode_sparse_py(sparse)

    @staticmethod
    def _encode_sparse_py(sparse: dict) -> bytes:
        """Pure-Python sparse encoder (fallback / source of truth for bytes)."""
        nnz = len(sparse)
        parts = [struct.pack("<I", nnz)]
        for idx, val in sorted(sparse.items()):
            parts.append(struct.pack("<If", int(idx), float(val)))
        return b"".join(parts)

    @staticmethod
    def _decode_sparse(data: bytes) -> dict:
        """Decode a binary BLOB back to a sparse vector dict[int, float].

        Pure-Python only: the SWIG surface exposes SparseRow(dict)/serialize()/
        to_dict() but NO bytes->SparseRow constructor, so the C++ codec cannot
        round-trip bytes back to a row.  Contract A is an encode-side byte
        claim; decode stays the single Python source of truth.
        """
        nnz = struct.unpack_from("<I", data, 0)[0]
        result = {}
        offset = 4
        for _ in range(nnz):
            idx, val = struct.unpack_from("<If", data, offset)
            result[idx] = val
            offset += 8
        return result

    @staticmethod
    def _encode_vector(vector: Any) -> bytes:
        if isinstance(vector, dict):
            return ScalarStore._encode_sparse(vector)
        arr = np.asarray(vector, dtype=np.float32, order="C")
        if arr.ndim != 1:
            raise ValueError("vector must be a 1-D array.")
        return arr.tobytes()

    @staticmethod
    def _decode_vector(data: bytes, dim: int | None) -> Any:
        if dim is None:
            return ScalarStore._decode_sparse(data)
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size != int(dim):
            raise ValueError(f"stored vector dim {arr.size} does not match collection dim {dim}.")
        return arr.copy()

    @staticmethod
    def _create_table_ddl(table: str) -> str:
        return f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
              row_id INTEGER PRIMARY KEY,
              doc_id TEXT UNIQUE NOT NULL,
              vector BLOB NOT NULL,
              sparse_vector BLOB,
              text_content TEXT,
              metadata TEXT,
              created_at REAL,
              updated_at REAL
            )
            """

    @staticmethod
    def _terms_table(collection_name: str) -> str:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in collection_name)
        return f"terms_{safe}"

    def ensure_table(self, collection_name: str) -> None:
        table = self._table(collection_name)
        conn = self._conn()
        conn.execute(self._create_table_ddl(table))
        conn.execute(f'CREATE INDEX IF NOT EXISTS "{table}_doc_id" ON "{table}"(doc_id)')
        self._ensure_sparse_column(table)
        conn.commit()

    def _ensure_sparse_column(self, table: str) -> None:
        """Add the nullable sparse_vector column to a pre-existing table.

        Old collections created before dual-field support have only the dense
        `vector` column; this migrates them idempotently on next open.  Fresh
        tables already have the column from the DDL, so the ALTER is skipped.
        """
        conn = self._conn()
        cols = {row["name"] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if "sparse_vector" not in cols:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN sparse_vector BLOB')

    def ensure_terms_table(self, collection_name: str) -> None:
        """Create the per-collection term dictionary table if absent.

        Stores the term-string <-> term-id mapping (see hypervec_term_dictionary
        .TermDictionary) so string-keyed sparse vectors survive a restart with
        stable ids.
        """
        terms = self._terms_table(collection_name)
        conn = self._conn()
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{terms}" (
              term_id INTEGER PRIMARY KEY,
              term TEXT UNIQUE NOT NULL
            )
            """
        )
        conn.commit()

    def load_term_dictionary(self, collection_name: str) -> "TermDictionary":
        """Load the persisted term dictionary, ordered by ascending term_id.

        Returns an empty dictionary if the collection has no terms table.
        """
        terms = self._terms_table(collection_name)
        exists = self._conn().execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
            (terms,),
        ).fetchone()
        dictionary = TermDictionary()
        if exists is None:
            return dictionary
        cur = self._conn().execute(
            f'SELECT term_id, term FROM "{terms}" ORDER BY term_id ASC'
        )
        for row in cur.fetchall():
            # Rows are ordered by id, which equals first-seen order, so
            # get_or_add reproduces the same id assignment.
            dictionary.get_or_add(row["term"])
        return dictionary

    def save_term_dictionary(self, collection_name: str, dictionary: "TermDictionary") -> None:
        """Upsert every (term_id, term) pair.  Idempotent and incremental."""
        self.ensure_terms_table(collection_name)
        terms = self._terms_table(collection_name)
        conn = self._conn()
        conn.executemany(
            f'INSERT OR IGNORE INTO "{terms}" (term_id, term) VALUES (?, ?)',
            [(term_id, term) for term_id, term in dictionary.items()],
        )
        conn.commit()

    def drop_table(self, collection_name: str) -> None:
        self._conn().execute(f'DROP TABLE IF EXISTS "{self._table(collection_name)}"')
        self._conn().execute(f'DROP TABLE IF EXISTS "{self._terms_table(collection_name)}"')
        self._conn().commit()

    def count(self, collection_name: str) -> int:
        try:
            cur = self._conn().execute(f'SELECT COUNT(*) FROM "{self._table(collection_name)}"')
            return int(cur.fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    def next_row_id(self, collection_name: str) -> int:
        try:
            cur = self._conn().execute(f'SELECT COALESCE(MAX(row_id), -1) + 1 FROM "{self._table(collection_name)}"')
            return int(cur.fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    def insert_batch(
        self,
        collection_name: str,
        rows: list[tuple],
    ) -> None:
        """Insert rows into a collection.

        Each row is either a 5-tuple ``(row_id, doc_id, vector, text, metadata)``
        (single-vector collections — dense OR sparse, stored in the `vector`
        column) or a 6-tuple ``(row_id, doc_id, dense_vector, sparse_vector,
        text, metadata)`` (dual-field collections; either vector may be None).
        Dense goes in `vector`, sparse in the nullable `sparse_vector` column.
        """
        table = self._table(collection_name)
        now = time.time()

        def _row_values(row: tuple) -> tuple:
            if len(row) == 6:
                row_id, doc_id, dense_vector, sparse_vector, text_content, metadata = row
            else:
                row_id, doc_id, dense_vector, text_content, metadata = row
                sparse_vector = None
            dense_blob = sqlite3.Binary(self._encode_vector(dense_vector))
            sparse_blob = (
                sqlite3.Binary(self._encode_vector(sparse_vector))
                if sparse_vector is not None
                else None
            )
            return (
                int(row_id),
                str(doc_id),
                dense_blob,
                sparse_blob,
                text_content,
                json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":")),
                now,
                now,
            )

        try:
            self._conn().executemany(
                f"""
                INSERT INTO "{table}"
                  (row_id, doc_id, vector, sparse_vector, text_content, metadata,
                   created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_row_values(row) for row in rows],
            )
            self._conn().commit()
        except sqlite3.IntegrityError as exc:
            self._conn().rollback()
            raise ValueError(
                f"duplicate row_id or doc_id in collection '{collection_name}'."
            ) from exc

    def get_vectors(self, collection_name: str, dim: int) -> np.ndarray:
        table = self._table(collection_name)
        cur = self._conn().execute(f'SELECT vector FROM "{table}" ORDER BY row_id ASC')
        vectors = [self._decode_vector(row["vector"], dim) for row in cur.fetchall()]
        if not vectors:
            return np.empty((0, int(dim)), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32, copy=False)

    # Alias: dense-column reader (dual-field flush uses this name explicitly).
    def get_dense_vectors(self, collection_name: str, dim: int) -> np.ndarray:
        return self.get_vectors(collection_name, dim)

    def get_sparse_vectors(self, collection_name: str) -> list:
        """Return all sparse vectors ordered by row_id as list[dict[int, float]].

        Reads the dense `vector` column — this is the single-field sparse path
        where the sparse blob lives in `vector` (pure-sparse collections).
        """
        table = self._table(collection_name)
        cur = self._conn().execute(f'SELECT vector FROM "{table}" ORDER BY row_id ASC')
        return [self._decode_sparse(row["vector"]) for row in cur.fetchall()]

    def get_sparse_vectors_from_column(self, collection_name: str) -> list:
        """Return sparse vectors from the dedicated `sparse_vector` column
        (dual-field collections), ordered by row_id.  Rows with a NULL
        sparse_vector yield an empty dict.
        """
        table = self._table(collection_name)
        cur = self._conn().execute(
            f'SELECT sparse_vector FROM "{table}" ORDER BY row_id ASC'
        )
        result = []
        for row in cur.fetchall():
            blob = row["sparse_vector"]
            result.append(self._decode_sparse(blob) if blob is not None else {})
        return result

    def get_by_row_ids(
        self,
        collection_name: str,
        row_ids: list[int],
    ) -> list[dict[str, Any] | None]:
        if not row_ids:
            return []
        table = self._table(collection_name)
        placeholders = ",".join("?" for _ in row_ids)
        cur = self._conn().execute(
            f'SELECT row_id, doc_id, text_content, metadata FROM "{table}" '
            f"WHERE row_id IN ({placeholders})",
            [int(row_id) for row_id in row_ids],
        )
        by_row_id = {
            int(row["row_id"]): {
                "doc_id": row["doc_id"],
                "text_content": row["text_content"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in cur.fetchall()
        }
        return [by_row_id.get(int(row_id)) for row_id in row_ids]

    def load_all_scalars(self, collection_name: str) -> dict[int, dict[str, Any]]:
        table = self._table(collection_name)
        exists = self._conn().execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if exists is None:
            return {}
        cur = self._conn().execute(
            f'SELECT row_id, doc_id, text_content, metadata FROM "{table}"'
        )
        return {
            int(row["row_id"]): {
                "doc_id": row["doc_id"],
                "text_content": row["text_content"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in cur.fetchall()
        }

    # ------------------------------------------------------------------
    # Bundle export / import / purge helpers
    # ------------------------------------------------------------------

    def export_rows(self, collection_name: str) -> list[dict]:
        """Return all rows ordered by row_id, each as a plain dict.

        For dense vectors: "vector" is list[float].
        For sparse vectors: "vector" is {"indices": [...], "values": [...]}.
        Used when building a collection data bundle.
        """
        table = self._table(collection_name)
        try:
            cur = self._conn().execute(
                f'SELECT row_id, doc_id, vector, sparse_vector, text_content, '
                f'metadata, created_at, updated_at FROM "{table}" ORDER BY row_id ASC'
            )
        except sqlite3.OperationalError:
            obj_exists = self._conn().execute(
                "SELECT 1 FROM sqlite_schema WHERE type IN ('table','view') AND name=?",
                (table,),
            ).fetchone()
            if obj_exists:
                raise
            return []
        rows = []
        for row in cur.fetchall():
            raw = bytes(row["vector"])
            vector = self._export_vector(raw)
            entry = {
                "row_id": int(row["row_id"]),
                "doc_id": row["doc_id"],
                "vector": vector,
                "text_content": row["text_content"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            # Dual-field collections carry a second sparse vector in its own
            # column; export it alongside the dense one.
            sparse_raw = row["sparse_vector"]
            if sparse_raw is not None:
                entry["sparse_vector"] = self._export_vector(bytes(sparse_raw))
            rows.append(entry)
        return rows

    @staticmethod
    def _export_vector(raw: bytes) -> Any:
        """Detect dense vs sparse from raw bytes and return JSON-serialisable form.

        Dense: raw length is a multiple of 4 (float32s) AND does not match sparse
        framing → list[float].
        Sparse: first 4 bytes as nnz, remaining = nnz * 8 bytes → {"indices", "values"}.

        Detection heuristic: try sparse decode first (nnz * 8 + 4 == len(raw));
        fall back to dense.
        """
        if len(raw) >= 4:
            nnz = struct.unpack_from("<I", raw, 0)[0]
            if 4 + nnz * 8 == len(raw):
                sparse = ScalarStore._decode_sparse(raw)
                sorted_items = sorted(sparse.items())
                return {
                    "indices": [int(k) for k, _ in sorted_items],
                    "values": [float(v) for _, v in sorted_items],
                }
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.tolist()

    def import_rows(
        self,
        collection_name: str,
        rows: list[dict],
        *,
        replace: bool = True,
    ) -> int:
        """Restore rows exported by export_rows().

        When replace=True (default) the existing table is dropped first so
        row_ids start fresh.  Returns the number of rows inserted.
        """
        if replace:
            self.drop_table(collection_name)
        self.ensure_table(collection_name)
        if not rows:
            return 0
        table = self._table(collection_name)
        now = time.time()
        self._conn().executemany(
            f"""
            INSERT INTO "{table}"
              (row_id, doc_id, vector, sparse_vector, text_content, metadata,
               created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(r["row_id"]),
                    str(r["doc_id"]),
                    sqlite3.Binary(self._encode_vector(self._import_vector(r["vector"]))),
                    self._encode_optional_sparse(r.get("sparse_vector")),
                    r.get("text_content", ""),
                    json.dumps(
                        dict(r.get("metadata") or {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    r.get("created_at") or now,
                    r.get("updated_at") or now,
                )
                for r in rows
            ],
        )
        self._conn().commit()
        return len(rows)

    @classmethod
    def _encode_optional_sparse(cls, sparse_vector: Any):
        """Encode a bundle-JSON sparse vector to a BLOB, or None if absent."""
        if sparse_vector is None:
            return None
        return sqlite3.Binary(cls._encode_vector(cls._import_vector(sparse_vector)))

    @staticmethod
    def _import_vector(vector: Any) -> Any:
        """Normalise a vector from bundle JSON to the internal Python representation.

        Dense: list[float] → kept as-is (encode_vector handles it).
        Sparse: {"indices": [...], "values": [...]} → dict[int, float].
        """
        if isinstance(vector, dict) and "indices" in vector and "values" in vector:
            return {int(k): float(v) for k, v in zip(vector["indices"], vector["values"])}
        return vector

    def purge_collection_rows(self, collection_name: str) -> dict:
        """DROP the collection's table.  Returns summary dict."""
        count_before = self.count(collection_name)
        self.drop_table(collection_name)
        return {"dropped": True, "count_before": count_before}

    # ------------------------------------------------------------------
    # Transactional import staging (Phase 3)
    #
    # Rows are loaded into a transient docs_<name>__import table first, leaving
    # the live docs_<name> table untouched.  commit_staging() then performs an
    # atomic (single-transaction) DROP + RENAME so the collection is never left
    # with a half-imported live table.
    # ------------------------------------------------------------------

    def import_rows_to_staging(self, collection_name: str, rows: list[dict]) -> int:
        """Load rows into the staging table, replacing any previous staging.

        Never touches the live docs_<name> table.  Returns rows inserted.
        """
        staging = self._staging_table(collection_name)
        conn = self._conn()
        conn.execute(f'DROP TABLE IF EXISTS "{staging}"')
        conn.execute(self._create_table_ddl(staging))
        if rows:
            now = time.time()
            conn.executemany(
                f"""
                INSERT INTO "{staging}"
                  (row_id, doc_id, vector, sparse_vector, text_content, metadata,
                   created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(r["row_id"]),
                        str(r["doc_id"]),
                        sqlite3.Binary(self._encode_vector(self._import_vector(r["vector"]))),
                        self._encode_optional_sparse(r.get("sparse_vector")),
                        r.get("text_content", ""),
                        json.dumps(
                            dict(r.get("metadata") or {}),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        r.get("created_at") or now,
                        r.get("updated_at") or now,
                    )
                    for r in rows
                ],
            )
        conn.commit()
        return len(rows)

    def commit_staging(self, collection_name: str) -> None:
        """Atomically swap the staging table into the live table.

        Runs DROP live + RENAME staging -> live in one SQLite transaction, so
        the file is never observed with the live table dropped but staging not
        yet renamed.
        """
        table = self._table(collection_name)
        staging = self._staging_table(collection_name)
        conn = self._conn()
        conn.execute("BEGIN")
        try:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.execute(f'ALTER TABLE "{staging}" RENAME TO "{table}"')
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{table}_doc_id" ON "{table}"(doc_id)'
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def rollback_staging(self, collection_name: str) -> None:
        """Drop the staging table (if any).  Live table is left untouched."""
        self._conn().execute(
            f'DROP TABLE IF EXISTS "{self._staging_table(collection_name)}"'
        )
        self._conn().commit()

    def has_staging(self, collection_name: str) -> bool:
        cur = self._conn().execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?",
            (self._staging_table(collection_name),),
        )
        return cur.fetchone() is not None

    def checkpoint_and_vacuum(self) -> None:
        """Flush WAL and compact the SQLite file.

        This reduces the chance of data residue in WAL/SHM files after purge.
        Note: this is not a cryptographic-erase guarantee — SSD wear-levelling
        and OS-level snapshots may retain data at the block level.
        """
        conn = self._conn()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
