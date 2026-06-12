# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


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
            self._local.conn = conn
        return self._local.conn

    @staticmethod
    def _table(collection_name: str) -> str:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in collection_name)
        return f"docs_{safe}"

    @staticmethod
    def _encode_vector(vector: Any) -> bytes:
        arr = np.asarray(vector, dtype=np.float32, order="C")
        if arr.ndim != 1:
            raise ValueError("vector must be a 1-D array.")
        return arr.tobytes()

    @staticmethod
    def _decode_vector(data: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size != int(dim):
            raise ValueError(f"stored vector dim {arr.size} does not match collection dim {dim}.")
        return arr.copy()

    def ensure_table(self, collection_name: str) -> None:
        table = self._table(collection_name)
        conn = self._conn()
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
              row_id INTEGER PRIMARY KEY,
              doc_id TEXT UNIQUE NOT NULL,
              vector BLOB NOT NULL,
              text_content TEXT,
              metadata TEXT,
              created_at REAL,
              updated_at REAL
            )
            """
        )
        conn.execute(f'CREATE INDEX IF NOT EXISTS "{table}_doc_id" ON "{table}"(doc_id)')
        conn.commit()

    def drop_table(self, collection_name: str) -> None:
        self._conn().execute(f'DROP TABLE IF EXISTS "{self._table(collection_name)}"')
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
        rows: list[tuple[int, str, Any, str, dict[str, Any]]],
    ) -> None:
        table = self._table(collection_name)
        now = time.time()
        try:
            self._conn().executemany(
                f"""
                INSERT INTO "{table}"
                  (row_id, doc_id, vector, text_content, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(row_id),
                        str(doc_id),
                        sqlite3.Binary(self._encode_vector(vector)),
                        text_content,
                        json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":")),
                        now,
                        now,
                    )
                    for row_id, doc_id, vector, text_content, metadata in rows
                ],
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

    def get_by_row_ids(
        self,
        collection_name: str,
        row_ids: list[int],
    ) -> list[dict[str, Any] | None]:
        if not row_ids:
            return []
        table = self._table(collection_name)
        placeholders = ",".join("?" for _ in row_ids)
        cur = self._conn().execute(            f'SELECT row_id, doc_id, text_content, metadata FROM "{table}" '
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
        """一次性把整个 collection 的标量字段读入内存，用于查询期间旁路 SQLite 回表。

        返回 {row_id: {"doc_id", "text_content", "metadata"}}。
        """
        table = self._table(collection_name)
        try:
            cur = self._conn().execute(
                f'SELECT row_id, doc_id, text_content, metadata FROM "{table}"'
            )
        except sqlite3.OperationalError:
            return {}
        cache: dict[int, dict[str, Any]] = {}
        for row in cur.fetchall():
            cache[int(row["row_id"])] = {
                "doc_id": row["doc_id"],
                "text_content": row["text_content"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }
        return cache
