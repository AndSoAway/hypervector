"""
SQLite-backed scalar field store.

Replaces Milvus dynamic-field storage.
Each collection gets its own table: docs_{collection_name}.

Schema per table:
  row_id       INTEGER PRIMARY KEY  -- maps to HyperVec sequential index (0, 1, 2, ...)
  doc_id       TEXT UNIQUE          -- caller-supplied string ID
  text_content TEXT                 -- original document content
  metadata     TEXT                 -- JSON blob of all scalar fields
  created_at   REAL
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

_DB_PATH = Path(__file__).parent / "scalar.db"


class ScalarStore:
    def __init__(self, path: Path = _DB_PATH):
        self._path = path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _table(self, collection: str) -> str:
        # sanitize: only alphanum + underscore
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in collection)
        return f"docs_{safe}"

    def ensure_table(self, collection: str) -> None:
        tbl = self._table(collection)
        self._conn().execute(f"""
            CREATE TABLE IF NOT EXISTS "{tbl}" (
                row_id       INTEGER PRIMARY KEY,
                doc_id       TEXT UNIQUE NOT NULL,
                text_content TEXT,
                metadata     TEXT,
                created_at   REAL
            )
        """)
        self._conn().execute(
            f'CREATE INDEX IF NOT EXISTS "{tbl}_doc_id" ON "{tbl}"(doc_id)'
        )
        self._conn().commit()

    def drop_table(self, collection: str) -> None:
        tbl = self._table(collection)
        self._conn().execute(f'DROP TABLE IF EXISTS "{tbl}"')
        self._conn().commit()

    def insert_batch(
        self,
        collection: str,
        rows: list[tuple[int, str, str, dict[str, Any]]],
    ) -> None:
        """
        rows: list of (row_id, doc_id, text_content, metadata_dict)
        row_id must match HyperVec sequential index.
        """
        tbl = self._table(collection)
        now = time.time()
        self._conn().executemany(
            f"""
            INSERT OR REPLACE INTO "{tbl}"
                (row_id, doc_id, text_content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (row_id, doc_id, text, json.dumps(metadata or {}), now)
                for row_id, doc_id, text, metadata in rows
            ],
        )
        self._conn().commit()

    def get_by_row_ids(
        self,
        collection: str,
        row_ids: list[int],
    ) -> list[Optional[dict[str, Any]]]:
        """Return rows in the same order as row_ids. Missing rows → None."""
        if not row_ids:
            return []
        tbl = self._table(collection)
        placeholders = ",".join("?" * len(row_ids))
        cur = self._conn().execute(
            f'SELECT row_id, doc_id, text_content, metadata FROM "{tbl}" '
            f"WHERE row_id IN ({placeholders})",
            row_ids,
        )
        by_id = {}
        for r in cur.fetchall():
            by_id[r["row_id"]] = {
                "doc_id": r["doc_id"],
                "text_content": r["text_content"],
                "metadata": json.loads(r["metadata"] or "{}"),
            }
        return [by_id.get(rid) for rid in row_ids]

    def count(self, collection: str) -> int:
        tbl = self._table(collection)
        try:
            cur = self._conn().execute(f'SELECT COUNT(*) FROM "{tbl}"')
            return cur.fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    def list_collections(self) -> list[str]:
        cur = self._conn().execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'docs_%'"
        )
        return [r[0][len("docs_"):] for r in cur.fetchall()]
