"""
HyperVec drop-in replacement for MilvusIndexBackend.

Implements the same interface as MilvusIndexBackend:
  - build_index(embeddings, ids, overwrite, **kwargs)
  - search(query_embeddings, top_k, **kwargs) -> List[List[str]]
  - search_with_meta(query_embeddings, top_k, **kwargs) -> List[List[dict]]

Scalar fields from METADATA_FIELD_LIMITS are stored in SQLite.
Vectors are stored in HyperVec IndexFlatL2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

try:
    from .base import BaseIndexBackend
except ImportError:
    # Fallback when running outside the ultrarag package (e.g. standalone tests)
    class BaseIndexBackend:  # type: ignore[no-redef]
        def __init__(self, contents, config, logger, **_):
            self.config = config or {}
            self.logger = logger

from index_manager import IndexManager
from scalar_store import ScalarStore
from store import MetaStore

# Mirrors milvus_backend.METADATA_FIELD_LIMITS — enforced at insert time.
METADATA_FIELD_LIMITS: dict[str, int] = {
    "source": 256,
    "file_name": 256,
    "json_name": 256,
    "title": 256,
    "doc_id": 384,
    "source_type": 32,
    "chunk_backend": 32,
    "marker_heading": 512,
    "heading_path": 512,
    "category_path": 512,
    "term_id": 128,
    "term_zh": 128,
    "term_en": 256,
    "entry_id": 128,
    "legal_article": 128,
    "uploaded_at": 64,
    "updated_at": 64,
    "completed_at": 64,
}
DEFAULT_METADATA_STRING_LIMIT = 256

_SERVER_DIR = Path(__file__).parent


def _make_manager() -> IndexManager:
    store = MetaStore()
    scalar = ScalarStore()
    return IndexManager(store, scalar)


# Module-level singletons (one per process).
_store = MetaStore()
_scalar = ScalarStore()
_manager = IndexManager(_store, _scalar)


def _sanitize_metadata(meta: dict[str, Any], banned: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        k = str(k or "").strip()
        if not k or k in banned:
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str):
            limit = METADATA_FIELD_LIMITS.get(k, DEFAULT_METADATA_STRING_LIMIT)
            out[k] = v.strip()[:limit] if limit > 0 else v.strip()
    return out


class HyperVecIndexBackend(BaseIndexBackend):
    """HyperVec-based index backend — drop-in replacement for MilvusIndexBackend."""

    def __init__(
        self,
        contents: Sequence[str],
        config: Optional[dict[str, Any]],
        logger,
        **_: Any,
    ) -> None:
        super().__init__(contents=[], config=config, logger=logger)

        self.collection_name: str = str(self.config.get("collection_name", "default"))
        self.collection_display_name: str = str(
            self.config.get("collection_display_name", self.collection_name)
        )
        self.id_field: str = str(self.config.get("id_field_name", "id"))
        self.vector_field: str = str(self.config.get("vector_field_name", "vector"))
        self.text_field: str = str(self.config.get("text_field_name", "contents"))

        self._manager = _manager

    # ------------------------------------------------------------------
    # BaseIndexBackend interface
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """No-op: indexes are loaded from disk automatically on startup."""
        pass

    def build_index(
        self,
        *,
        embeddings: np.ndarray,
        ids: np.ndarray,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        collection_name = str(kwargs.get("collection_name", self.collection_name))
        contents = kwargs.get("contents") or []
        metadatas = kwargs.get("metadatas") or []

        if not contents:
            raise ValueError("[hypervec] 'contents' is required for build_index.")

        embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
        doc_ids = [str(i) for i in np.array(ids)]

        if embeddings.ndim != 2:
            raise ValueError("[hypervec] embeddings must be 2-D.")
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("[hypervec] ids must align with embeddings.")

        dim = int(embeddings.shape[1])
        banned = {self.id_field, self.vector_field, self.text_field}

        sanitized_meta = [
            _sanitize_metadata(metadatas[i] if i < len(metadatas) else {}, banned)
            for i in range(len(doc_ids))
        ]
        texts = list(contents)

        meta = self._manager.get_meta(collection_name)

        if meta is None:
            self.logger.info(f"[hypervec] Creating collection '{collection_name}'.")
            self._manager.create_collection(collection_name, dim)
            self._manager.add_vectors(
                collection_name, embeddings, doc_ids, texts, sanitized_meta
            )
        elif overwrite:
            self.logger.info(f"[hypervec] Rebuilding collection '{collection_name}'.")
            self._manager.rebuild(
                collection_name, embeddings, doc_ids, texts, sanitized_meta
            )
        else:
            self.logger.info(
                f"[hypervec] Appending {len(doc_ids)} vectors to '{collection_name}'."
            )
            self._manager.add_vectors(
                collection_name, embeddings, doc_ids, texts, sanitized_meta
            )

        self.logger.info(
            f"[hypervec] Index ready on collection '{collection_name}'."
        )

    def search_with_meta(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        **kwargs: Any,
    ) -> List[List[dict[str, Any]]]:
        collection_name = str(kwargs.get("collection_name", self.collection_name))
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if query_embeddings.ndim != 2:
            raise ValueError("[hypervec] query embeddings must be 2-D.")

        route = str(kwargs.get("route") or "dense")

        try:
            raw = self._manager.search_with_meta(collection_name, query_embeddings, top_k)
        except KeyError as e:
            raise RuntimeError(f"[hypervec] Collection not found: {e}") from e

        results = []
        for hits in raw:
            row = []
            for hit in hits:
                metadata = hit["metadata"]
                row.append({
                    "content": hit["content"],
                    "score": hit["score"],
                    "rank": hit["rank"],
                    "collection_name": collection_name,
                    "route": route,
                    "chunk_id": hit["doc_id"],
                    "source": (
                        metadata.get("source")
                        or metadata.get("file_name")
                        or metadata.get("json_name")
                    ),
                    "metadata": metadata,
                })
            results.append(row)
        return results

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        **kwargs: Any,
    ) -> List[List[str]]:
        docs = self.search_with_meta(query_embeddings, top_k, **kwargs)
        return [[d["content"] for d in row] for row in docs]
