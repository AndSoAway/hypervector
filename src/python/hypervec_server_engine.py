# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np


class HypervecServerEngine:
    MANIFEST_FILE = "manifest.json"
    ROWS_FILE = "rows.jsonl"
    INDEX_FILE = "index.hypervec"

    def __init__(
        self,
        data_root: str,
        *,
        logger: logging.Logger | None = None,
        hypervec_module: Any | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("hypervec.server")
        if hypervec_module is None:
            import hypervec as hypervec_module
        self.hypervec = hypervec_module
        self._indexes: dict[str, Any] = {}
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()

    @staticmethod
    def validate_collection_name(name: str) -> str:
        if (
            not name
            or not isinstance(name, str)
            or len(name) > 255
            or not re.match(r"^[A-Za-z0-9_-]+$", name)
        ):
            raise ValueError(
                "collection_name must contain only alphanumeric characters, "
                "underscores, and hyphens, and must be at most 255 characters."
            )
        return name

    def _lock_for(self, collection_name: str) -> threading.RLock:
        with self._global_lock:
            return self._locks.setdefault(collection_name, threading.RLock())

    def _collection_dir(self, collection_name: str) -> Path:
        return self.data_root / self.validate_collection_name(collection_name)

    def _manifest_path(self, collection_name: str) -> Path:
        return self._collection_dir(collection_name) / self.MANIFEST_FILE

    def _rows_path(self, collection_name: str) -> Path:
        return self._collection_dir(collection_name) / self.ROWS_FILE

    def _index_path(self, collection_name: str) -> Path:
        return self._collection_dir(collection_name) / self.INDEX_FILE

    def _read_manifest(self, collection_name: str) -> dict[str, Any]:
        path = self._manifest_path(collection_name)
        if not path.exists():
            raise FileNotFoundError(f"collection '{collection_name}' does not exist.")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, collection_name: str, manifest: dict[str, Any]) -> None:
        path = self._manifest_path(collection_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _load_rows_from_disk(self, collection_name: str) -> list[dict[str, Any]]:
        rows_path = self._rows_path(collection_name)
        if not rows_path.exists():
            return []
        rows = []
        with rows_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _save_rows(self, collection_name: str, rows: list[dict[str, Any]]) -> None:
        rows_path = self._rows_path(collection_name)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")

    def _schema_fields(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        return list((manifest.get("schema") or {}).get("fields") or [])

    def _field_name_by_datatype(
        self,
        manifest: dict[str, Any],
        datatype: str,
        *,
        default: str,
    ) -> str:
        for field in self._schema_fields(manifest):
            if str(field.get("datatype", "")).upper() == datatype.upper():
                return str(field.get("name"))
        return default

    def _id_field(self, manifest: dict[str, Any]) -> str:
        for field in self._schema_fields(manifest):
            if bool(field.get("is_primary", False)):
                return str(field.get("name"))
        return str(manifest.get("id_field") or "id")

    def _vector_field(self, manifest: dict[str, Any]) -> str:
        return str(
            manifest.get("vector_field")
            or self._field_name_by_datatype(
                manifest, "FLOAT_VECTOR", default="vector"
            )
        )

    def _text_field(self, manifest: dict[str, Any]) -> str:
        for field in self._schema_fields(manifest):
            if str(field.get("name")) == "contents":
                return "contents"
        return str(manifest.get("text_field") or "contents")

    def _index_config(self, manifest: dict[str, Any]) -> dict[str, Any]:
        indexes = (manifest.get("index_params") or {}).get("indexes") or []
        if indexes:
            return dict(indexes[0])
        return {
            "field_name": self._vector_field(manifest),
            "metric_type": manifest.get("metric_type", "L2"),
            "index_type": manifest.get("index_type", "HNSWFlat"),
            "params": {},
        }

    def _metric(self, metric_type: str) -> int:
        metric = str(metric_type or "L2").upper()
        if metric in {"IP", "INNER_PRODUCT", "COSINE"}:
            return int(self.hypervec.kMetricInnerProduct)
        if metric in {"L2", "EUCLIDEAN"}:
            return int(self.hypervec.kMetricL2)
        raise ValueError(f"unsupported metric_type: {metric_type}")

    def _make_index(self, dim: int, index_config: dict[str, Any]) -> Any:
        metric = self._metric(str(index_config.get("metric_type", "L2")))
        index_type = str(index_config.get("index_type") or "HNSWFlat").upper()
        params = dict(index_config.get("params") or {})

        if index_type in {"FLAT", "INDEXFLAT"}:
            if metric == int(self.hypervec.kMetricInnerProduct):
                return self.hypervec.IndexFlatIP(dim)
            return self.hypervec.IndexFlatL2(dim)
        if index_type in {"HNSW", "HNSWFLAT", "INDEXHNSWFLAT", "AUTOINDEX"}:
            hnsw_m = int(params.get("M", params.get("m", 32)))
            return self.hypervec.IndexHNSWFlat(dim, hnsw_m, metric)
        if index_type in {"HNSWLVQ", "INDEXHNSWLVQ"}:
            nlocal = int(params.get("nlocal", 16))
            nbits = int(params.get("nbits", 8))
            hnsw_m = int(params.get("M_hnsw", params.get("M", 32)))
            return self.hypervec.IndexHNSWLVQ(dim, nlocal, nbits, hnsw_m, metric)
        raise ValueError(f"unsupported index_type: {index_config.get('index_type')}")

    def _write_index(self, index: Any, path: Path) -> None:
        if hasattr(self.hypervec, "write_index"):
            self.hypervec.write_index(index, str(path))
            return
        self.hypervec.WriteIndex(index, str(path))

    def _read_index(self, path: Path) -> Any:
        if hasattr(self.hypervec, "read_index"):
            return self.hypervec.read_index(str(path))
        return self.hypervec.ReadIndex(str(path))

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
                    "filter supports equality clauses joined by AND, "
                    "for example: source == 'manual' and doc_id == '42'."
                )
            key = match.group(1)
            expected = next(v for v in match.groups()[1:] if v is not None)
            if str(row.get(key)) != expected:
                return False
        return True

    def list_collections(self) -> list[str]:
        return sorted(
            path.name
            for path in self.data_root.iterdir()
            if path.is_dir() and (path / self.MANIFEST_FILE).exists()
        )

    def has_collection(self, collection_name: str) -> bool:
        return self._manifest_path(collection_name).exists()

    def create_collection(
        self,
        collection_name: str,
        *,
        schema: dict[str, Any],
        index_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            collection_dir = self._collection_dir(collection_name)
            if collection_dir.exists() and self._manifest_path(collection_name).exists():
                raise FileExistsError(f"collection '{collection_name}' already exists.")
            collection_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "backend": "hypervec-server",
                "version": 1,
                "collection_name": collection_name,
                "schema": dict(schema),
                "index_params": dict(index_params or {"indexes": []}),
                "id_field": "id",
                "vector_field": "vector",
                "text_field": "contents",
                "dim": None,
                "total": 0,
            }
            manifest["id_field"] = self._id_field(manifest)
            manifest["vector_field"] = self._vector_field(manifest)
            manifest["text_field"] = self._text_field(manifest)
            index_config = self._index_config(manifest)
            manifest["metric_type"] = str(index_config.get("metric_type", "L2"))
            manifest["index_type"] = str(index_config.get("index_type", "HNSWFlat"))
            self._write_manifest(collection_name, manifest)
            self._save_rows(collection_name, [])
            return self.describe_collection(collection_name)

    def drop_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            self._indexes.pop(collection_name, None)
            self._rows.pop(collection_name, None)
            collection_dir = self._collection_dir(collection_name)
            if collection_dir.exists():
                shutil.rmtree(collection_dir)
            return {"dropped": True, "collection_name": collection_name}

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        manifest = self._read_manifest(collection_name)
        fields = self._schema_fields(manifest)
        return {
            "collection_name": collection_name,
            "schema": manifest.get("schema", {}),
            "index_params": manifest.get("index_params", {}),
            "fields": fields,
            "dim": manifest.get("dim"),
            "total": manifest.get("total", 0),
            "manifest": manifest,
        }

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            manifest = self._read_manifest(collection_name)
            vector_field = self._vector_field(manifest)
            rows = self._load_rows_from_disk(collection_name)
            dim = manifest.get("dim")
            for row in data:
                if vector_field not in row:
                    raise ValueError(f"row is missing vector field '{vector_field}'.")
                vec = list(row[vector_field])
                if dim is None:
                    dim = len(vec)
                    manifest["dim"] = int(dim)
                elif int(dim) != len(vec):
                    raise ValueError(
                        f"vector dimension {len(vec)} does not match collection dim {dim}."
                    )
                rows.append(dict(row))
            manifest["total"] = len(rows)
            self._save_rows(collection_name, rows)
            self._write_manifest(collection_name, manifest)
            self._indexes.pop(collection_name, None)
            self._rows.pop(collection_name, None)
            return {"insert_count": len(data), "total": len(rows)}

    def flush(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            manifest = self._read_manifest(collection_name)
            rows = self._load_rows_from_disk(collection_name)
            vector_field = self._vector_field(manifest)
            if not rows:
                raise ValueError(f"collection '{collection_name}' has no rows.")
            vectors = np.asarray(
                [row[vector_field] for row in rows],
                dtype=np.float32,
                order="C",
            )
            if vectors.ndim != 2:
                raise ValueError("inserted vectors must form a 2-D matrix.")
            index = self._make_index(int(vectors.shape[1]), self._index_config(manifest))
            if not getattr(index, "is_trained", True):
                index.Train(vectors)
            index.Add(vectors)
            self._write_index(index, self._index_path(collection_name))
            manifest["dim"] = int(vectors.shape[1])
            manifest["total"] = len(rows)
            self._write_manifest(collection_name, manifest)
            self._indexes[collection_name] = index
            self._rows[collection_name] = rows
            return {
                "flushed": True,
                "collection_name": collection_name,
                "total": len(rows),
                "dim": int(vectors.shape[1]),
            }

    def load_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            index_path = self._index_path(collection_name)
            if not index_path.exists():
                raise FileNotFoundError(
                    f"collection '{collection_name}' index has not been flushed."
                )
            index = self._read_index(index_path)
            rows = self._load_rows_from_disk(collection_name)
            self._indexes[collection_name] = index
            self._rows[collection_name] = rows
            manifest = self._read_manifest(collection_name)
            return {
                "loaded": True,
                "collection_name": collection_name,
                "total": len(rows),
                "dim": manifest.get("dim"),
            }

    def close_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            self._indexes.pop(collection_name, None)
            self._rows.pop(collection_name, None)
            return {"closed": True, "collection_name": collection_name}

    def search(
        self,
        collection_name: str,
        *,
        data: Any,
        limit: int,
        search_params: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
        filter: str = "",
        consistency_level: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        del search_params, consistency_level
        collection_name = self.validate_collection_name(collection_name)
        if int(limit) <= 0:
            raise ValueError("limit must be positive.")
        with self._lock_for(collection_name):
            if collection_name not in self._indexes:
                self.load_collection(collection_name)
            index = self._indexes[collection_name]
            rows = self._rows[collection_name]
            manifest = self._read_manifest(collection_name)
            vector_field = self._vector_field(manifest)
            query = np.asarray(data, dtype=np.float32, order="C")
            if query.ndim != 2:
                raise ValueError("search data must be a 2-D matrix.")
            if int(query.shape[1]) != int(manifest.get("dim")):
                raise ValueError(
                    f"query dim {query.shape[1]} != collection dim {manifest.get('dim')}."
                )

            candidate_k = min(len(rows), max(int(limit), int(limit) * 8))
            distances, labels = index.Search(query, candidate_k)
            requested = set(output_fields or [])
            results: list[list[dict[str, Any]]] = []
            for q_labels, q_distances in zip(labels, distances):
                hits = []
                for label, distance in zip(q_labels, q_distances):
                    label_i = int(label)
                    if label_i < 0 or label_i >= len(rows):
                        continue
                    row = rows[label_i]
                    if not self._filter_match(row, filter):
                        continue
                    if requested:
                        entity = {
                            key: value
                            for key, value in row.items()
                            if key in requested and key != vector_field
                        }
                    else:
                        entity = {
                            key: value
                            for key, value in row.items()
                            if key != vector_field
                        }
                    hits.append(
                        {
                            "id": row.get(self._id_field(manifest), label_i),
                            "distance": float(distance),
                            "entity": entity,
                        }
                    )
                    if len(hits) >= int(limit):
                        break
                results.append(hits)
            return results
