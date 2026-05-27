# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for this backend
    tqdm = None

try:
    from ultrarag.mcp_logging import tqdm_disabled
except ImportError:  # pragma: no cover - keep the backend usable standalone
    def tqdm_disabled() -> bool:
        return True

try:
    from .base import BaseIndexBackend
except ImportError:
    class BaseIndexBackend:
        def __init__(
            self,
            contents: Sequence[str],
            config: Optional[dict[str, Any]],
            logger,
            **_: Any,
        ) -> None:
            self.contents = list(contents)
            self.config = dict(config or {})
            self.logger = logger

import hypervec


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


class HyperVectorIndexBackend(BaseIndexBackend):
    """Local HyperVector-backed replacement for the Milvus index backend.

    The backend stores vectors in a HyperVector index file and stores scalar
    fields in local JSON files. It intentionally does not provide server
    semantics; each collection is a directory on local disk.
    """

    MANIFEST_FILE = "manifest.json"
    INDEX_FILE = "index.hypervec"
    ROWS_FILE = "rows.jsonl"

    def __init__(
        self,
        contents: Sequence[str],
        config: Optional[dict[str, Any]],
        logger,
        **_: Any,
    ) -> None:
        super().__init__(contents=[], config=config, logger=logger)

        self.uri = str(self._resolve_index_path(self.config.get("uri")))
        self.collection_name = self.config.get("collection_name")
        self.collection_display_name = self.config.get("collection_display_name")

        self.id_field: str = str(self.config.get("id_field_name", "id"))
        self.vector_field: str = str(self.config.get("vector_field_name", "vector"))
        self.text_field: str = str(self.config.get("text_field_name", "contents"))

        self.metric_type: str = str(self.config.get("metric_type", "IP")).upper()
        self.index_params: dict[str, Any] = dict(self.config.get("index_params", {}))
        self.search_params: dict[str, Any] = dict(self.config.get("search_params", {}))
        self.id_max_length = int(self.config.get("id_max_length", 64))
        self.text_max_length = int(self.config.get("text_max_length", 60000))

        self.index = None
        self.rows: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self._loaded_collection: Optional[str] = None

    @staticmethod
    def _read_index(path: str):
        if hasattr(hypervec, "read_index"):
            return hypervec.read_index(path)
        if hasattr(hypervec, "ReadIndex"):
            return hypervec.ReadIndex(path)
        raise RuntimeError("[hypervector] Python binding does not expose read_index/ReadIndex.")

    @staticmethod
    def _write_index(index, path: str) -> None:
        if hasattr(hypervec, "write_index"):
            hypervec.write_index(index, path)
            return
        if hasattr(hypervec, "WriteIndex"):
            hypervec.WriteIndex(index, path)
            return
        raise RuntimeError("[hypervector] Python binding does not expose write_index/WriteIndex.")

    @staticmethod
    def _validate_collection_name(name: str) -> bool:
        if not name or not isinstance(name, str):
            return False
        if len(name) > 255:
            return False
        return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))

    def _resolve_index_path(self, index_path: Optional[str]) -> str:
        if not index_path:
            raise ValueError("[hypervector] 'uri' (index_path) is required in config.")
        path = Path(str(index_path)).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _collection_dir(self, collection_name: Optional[str] = None) -> Path:
        target = collection_name or self.collection_name
        if not target:
            raise ValueError("[hypervector] collection_name is required.")
        if not self._validate_collection_name(str(target)):
            raise ValueError(
                f"[hypervector] Invalid collection name: '{target}'. "
                "Collection names must contain only alphanumeric characters, underscores, and hyphens."
            )
        return Path(self.uri) / str(target)

    def _sanitize_metadata(
        self,
        meta: dict[str, Any],
        banned_keys: set[str],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (meta or {}).items():
            key_s = str(key or "").strip()
            if not key_s or key_s in banned_keys:
                continue
            if isinstance(value, bool):
                out[key_s] = value
                continue
            if isinstance(value, (int, float)):
                out[key_s] = value
                continue
            if isinstance(value, str):
                limit = METADATA_FIELD_LIMITS.get(key_s, DEFAULT_METADATA_STRING_LIMIT)
                text = value.strip()
                out[key_s] = text[:limit] if limit > 0 else text
        return out

    def _metric(self) -> int:
        if self.metric_type in {"IP", "INNER_PRODUCT", "COSINE"}:
            return hypervec.kMetricInnerProduct
        if self.metric_type in {"L2", "EUCLIDEAN"}:
            return hypervec.kMetricL2
        raise ValueError(f"[hypervector] Unsupported metric_type: {self.metric_type}")

    def _make_index(self, dim: int):
        index_type = str(
            self.index_params.get("index_type")
            or self.config.get("index_type")
            or "HNSWFlat"
        ).upper()
        metric = self._metric()

        if index_type in {"FLAT", "INDEXFLAT"}:
            if metric == hypervec.kMetricInnerProduct:
                return hypervec.IndexFlatIP(dim)
            return hypervec.IndexFlatL2(dim)

        if index_type in {"HNSW", "HNSWFLAT", "INDEXHNSWFLAT", "AUTOINDEX"}:
            hnsw_m = int(self.index_params.get("M", self.index_params.get("m", 32)))
            return hypervec.IndexHNSWFlat(dim, hnsw_m, metric)

        if index_type in {"HNSWLVQ", "INDEXHNSWLVQ"}:
            if not hasattr(hypervec, "IndexHNSWLVQ"):
                raise RuntimeError(
                    "[hypervector] IndexHNSWLVQ is not exposed by the current Python binding."
                )
            nlocal = int(self.index_params.get("nlocal", 16))
            nbits = int(self.index_params.get("nbits", 8))
            hnsw_m = int(self.index_params.get("M_hnsw", self.index_params.get("M", 32)))
            return hypervec.IndexHNSWLVQ(dim, nlocal, nbits, hnsw_m, metric)

        raise ValueError(f"[hypervector] Unsupported index_type: {index_type}")

    def _save_rows(self, rows_path: Path) -> None:
        with rows_path.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")

    def _load_rows(self, rows_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with rows_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _filter_match(self, row: dict[str, Any], filter_expr: str) -> bool:
        expr = (filter_expr or "").strip()
        if not expr:
            return True
        clauses = [part.strip() for part in re.split(r"\s+and\s+", expr, flags=re.I)]
        for clause in clauses:
            match = re.fullmatch(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(?:'([^']*)'|\"([^\"]*)\"|([^'\"][^\s]*))",
                clause,
            )
            if not match:
                raise ValueError(
                    "[hypervector] MVP filter supports equality clauses joined by AND, "
                    "for example: source == 'manual' and doc_id == '42'."
                )
            key = match.group(1)
            expected = next(v for v in match.groups()[1:] if v is not None)
            actual = row.get("metadata", {}).get(key)
            if actual is None:
                actual = row.get(key)
            if str(actual) != expected:
                return False
        return True

    def _row_to_result(
        self,
        row: dict[str, Any],
        *,
        score: Optional[float],
        rank: int,
        collection_name: str,
        route: str,
    ) -> dict[str, Any]:
        content_s = str(row.get("content") or "")
        metadata = dict(row.get("metadata") or {})
        chunk_key = str(row.get("id") or "").strip()
        if not chunk_key:
            chunk_key = hashlib.sha256(content_s.encode("utf-8")).hexdigest()
        return {
            "content": content_s,
            "score": score,
            "rank": rank,
            "collection_name": collection_name,
            "route": route,
            "chunk_id": chunk_key,
            "source": metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("json_name"),
            "metadata": metadata,
        }

    def load_index(self, *, index_path: Optional[str] = None) -> None:
        collection_name = index_path or self.collection_name
        collection_dir = self._collection_dir(collection_name)
        manifest_path = collection_dir / self.MANIFEST_FILE
        index_file = collection_dir / self.INDEX_FILE
        rows_path = collection_dir / self.ROWS_FILE

        if not manifest_path.exists() or not index_file.exists() or not rows_path.exists():
            raise FileNotFoundError(
                f"[hypervector] Collection '{collection_name}' is incomplete or missing under {collection_dir}."
            )

        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.rows = self._load_rows(rows_path)
        self.index = self._read_index(str(index_file))
        self._loaded_collection = str(collection_name)

        if int(self.manifest.get("total", len(self.rows))) != len(self.rows):
            raise ValueError("[hypervector] manifest row count does not match rows.jsonl.")

    def build_index(
        self,
        *,
        embeddings: np.ndarray,
        ids: np.ndarray,
        index_path: Optional[str] = None,
        overwrite: bool = False,
        index_chunk_size: int = 50000,
        **kwargs: Any,
    ) -> None:
        target_collection = kwargs.get("collection_name", index_path or self.collection_name)
        collection_dir = self._collection_dir(target_collection)

        if collection_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"[hypervector] Collection '{target_collection}' already exists. "
                    "Pass overwrite=True to rebuild it."
                )
            shutil.rmtree(collection_dir)
        collection_dir.mkdir(parents=True, exist_ok=True)

        passed_contents = kwargs.get("contents", None)
        passed_metadatas = kwargs.get("metadatas", None)
        if not passed_contents:
            raise ValueError("[hypervector] 'contents' is required for build_index.")

        embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
        ids = np.array(ids).astype(str)
        if embeddings.ndim != 2:
            raise ValueError("[hypervector] embeddings must be a 2-D array.")
        if ids.shape[0] != embeddings.shape[0]:
            raise ValueError("[hypervector] ids must align with embeddings.")
        if len(passed_contents) != embeddings.shape[0]:
            raise ValueError("[hypervector] contents must align with embeddings.")

        dim = int(embeddings.shape[1])
        total = int(embeddings.shape[0])
        self.index = self._make_index(dim)

        if not self.index.is_trained:
            self.index.Train(embeddings)

        progress_callback = kwargs.get("progress_callback")
        indexing_start = perf_counter()

        def _emit_progress(indexed_chunks: int) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(
                    {
                        "stage": "indexing",
                        "total_chunks": total,
                        "embedded_chunks": total,
                        "indexed_chunks": int(indexed_chunks),
                        "indexing_ms": int((perf_counter() - indexing_start) * 1000),
                    }
                )
            except Exception:
                pass

        _emit_progress(0)
        chunk_size = int(self.config.get("index_chunk_size", index_chunk_size))
        chunk_size = max(chunk_size, 1)
        iterator = range(0, total, chunk_size)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=(total + chunk_size - 1) // chunk_size,
                desc="[hypervector] Indexing",
                unit="batch",
                disable=tqdm_disabled(),
            )
        for start in iterator:
            end = min(start + chunk_size, total)
            self.index.Add(embeddings[start:end])
            _emit_progress(end)

        self.rows = []
        banned = {self.id_field, self.vector_field, self.text_field}
        for i, (doc_id, text) in enumerate(zip(ids, passed_contents)):
            metadata = {}
            if passed_metadatas and i < len(passed_metadatas):
                meta = passed_metadatas[i]
                if isinstance(meta, dict):
                    metadata = self._sanitize_metadata(meta, banned)
            text_s = str(text)
            if self.text_max_length > 0:
                text_s = text_s[: self.text_max_length]
            self.rows.append(
                {
                    "id": str(doc_id)[: self.id_max_length],
                    "content": text_s,
                    "metadata": metadata,
                }
            )

        manifest = {
            "backend": "hypervector",
            "version": 1,
            "collection_name": str(target_collection),
            "collection_display_name": self.collection_display_name,
            "dim": dim,
            "total": total,
            "metric_type": self.metric_type,
            "index_params": self.index_params,
            "search_params": self.search_params,
            "id_field": self.id_field,
            "vector_field": self.vector_field,
            "text_field": self.text_field,
        }
        self._write_index(self.index, str(collection_dir / self.INDEX_FILE))
        self._save_rows(collection_dir / self.ROWS_FILE)
        with (collection_dir / self.MANIFEST_FILE).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        self.manifest = manifest
        self._loaded_collection = str(target_collection)
        self.logger.info("[hypervector] Index ready on collection '%s'.", target_collection)

    def search_with_meta(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        **kwargs: Any,
    ) -> List[List[dict[str, Any]]]:
        target_collection = kwargs.get("collection_name", self.collection_name)
        if self.index is None or self._loaded_collection != str(target_collection):
            self.load_index(index_path=target_collection)

        query_embeddings = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if query_embeddings.ndim != 2:
            raise ValueError("[hypervector] query embeddings must be 2-D.")
        if query_embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"[hypervector] query dim {query_embeddings.shape[1]} != index dim {self.index.d}."
            )

        route = str(kwargs.get("route") or "dense")
        filter_expr = kwargs.get("filter", "") or ""
        search_k = int(kwargs.get("candidate_k", max(top_k, min(len(self.rows), top_k * 8))))
        search_k = max(search_k, top_k)
        search_k = min(search_k, len(self.rows)) if self.rows else search_k

        distances, labels = self.index.Search(query_embeddings, search_k)
        ret: list[list[dict[str, Any]]] = []
        for q_labels, q_distances in zip(labels, distances):
            row_results = []
            for label, distance in zip(q_labels, q_distances):
                label_i = int(label)
                if label_i < 0 or label_i >= len(self.rows):
                    continue
                row = self.rows[label_i]
                if not self._filter_match(row, filter_expr):
                    continue
                try:
                    score = float(distance)
                except (TypeError, ValueError):
                    score = None
                row_results.append(
                    self._row_to_result(
                        row,
                        score=score,
                        rank=len(row_results),
                        collection_name=str(target_collection),
                        route=route,
                    )
                )
                if len(row_results) >= top_k:
                    break
            ret.append(row_results)
        return ret

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        **kwargs: Any,
    ) -> List[List[str]]:
        docs = self.search_with_meta(query_embeddings, top_k, **kwargs)
        return [[str(doc.get("content") or "") for doc in row] for row in docs]

    def close(self) -> None:
        self.index = None
        self.rows = []
        self.manifest = {}
        self._loaded_collection = None
