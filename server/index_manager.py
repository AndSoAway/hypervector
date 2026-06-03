"""
HyperVec index lifecycle manager.

collection_name is the single key used everywhere (matches Milvus interface).
Row ID convention: HyperVec sequential int64 (0, 1, ...) == SQLite row_id.
"""

import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from scalar_store import ScalarStore
from store import MetaStore, CollectionMeta

_INDEX_DIR = Path(__file__).parent / "indexes"
_INDEX_DIR.mkdir(exist_ok=True)

def _hv():
    import hv
    return hv


class IndexManager:
    def __init__(self, store: MetaStore, scalar: ScalarStore):
        self._store = store
        self._scalar = scalar
        self._lock = threading.Lock()
        self._indexes: dict[str, Any] = {}  # collection_name -> Index
        self._restore_from_disk()

    def _restore_from_disk(self) -> None:
        for meta in self._store.list_all():
            path = Path(meta.index_path)
            if path.exists():
                try:
                    self._indexes[meta.name] = _hv().read_index(str(path))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, name: str, dim: int) -> CollectionMeta:
        index_path = str(_INDEX_DIR / f"{name}.index")
        index = _hv().IndexFlatL2(dim)
        with self._lock:
            self._indexes[name] = index
        self._scalar.ensure_table(name)
        meta = self._store.create(name, index_path)
        self._persist(name, index, index_path)
        return meta

    def delete_collection(self, name: str) -> bool:
        with self._lock:
            meta = self._store.get(name)
            if meta is None:
                return False
            self._indexes.pop(name, None)
            Path(meta.index_path).unlink(missing_ok=True)
        self._scalar.drop_table(name)
        return self._store.delete(name)

    def get_meta(self, name: str) -> Optional[CollectionMeta]:
        return self._store.get(name)

    def list_all(self) -> list[CollectionMeta]:
        return self._store.list_all()

    # ------------------------------------------------------------------
    # Index mutation
    # ------------------------------------------------------------------

    def add_vectors(
        self,
        name: str,
        embeddings: np.ndarray,
        doc_ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> CollectionMeta:
        with self._lock:
            index = self._indexes.get(name)
            if index is None:
                raise KeyError(f"Collection not found: {name}")
            start_row = index.ntotal
            index.add(embeddings)
            meta = self._store.get(name)
            self._persist(name, index, meta.index_path)

        rows = [
            (start_row + i, doc_ids[i], texts[i], metadatas[i] if metadatas else {})
            for i in range(len(doc_ids))
        ]
        self._scalar.insert_batch(name, rows)
        return self._store.bump_version(name)

    def rebuild(
        self,
        name: str,
        embeddings: np.ndarray,
        doc_ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> CollectionMeta:
        with self._lock:
            index = self._indexes.get(name)
            if index is None:
                raise KeyError(f"Collection not found: {name}")
            index.reset()
            index.add(embeddings)
            meta = self._store.get(name)
            self._persist(name, index, meta.index_path)

        self._scalar.drop_table(name)
        self._scalar.ensure_table(name)
        rows = [
            (i, doc_ids[i], texts[i], metadatas[i] if metadatas else {})
            for i in range(len(doc_ids))
        ]
        self._scalar.insert_batch(name, rows)
        return self._store.bump_version(name)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_with_meta(
        self,
        name: str,
        query: np.ndarray,
        top_k: int,
    ) -> list[list[dict]]:
        with self._lock:
            index = self._indexes.get(name)
            if index is None:
                raise KeyError(f"Collection not found: {name}")
            distances, labels = index.search(query, top_k)

        results = []
        for q_idx in range(len(query)):
            row_ids = [int(l) for l in labels[q_idx] if l >= 0]
            scalars = self._scalar.get_by_row_ids(name, row_ids)
            row = []
            for rank, (row_id, scalar) in enumerate(zip(row_ids, scalars)):
                if scalar is None:
                    continue
                row.append({
                    "doc_id": scalar["doc_id"],
                    "content": scalar["text_content"],
                    "score": float(distances[q_idx][rank]),
                    "rank": rank,
                    "metadata": scalar["metadata"],
                })
            results.append(row)
        return results

    def search(self, name: str, query: np.ndarray, top_k: int) -> list[list[str]]:
        docs = self.search_with_meta(name, query, top_k)
        return [[d["content"] for d in row] for row in docs]

    # ------------------------------------------------------------------

    def _persist(self, name: str, index: Any, path: str) -> None:
        _hv().write_index(index, path)
