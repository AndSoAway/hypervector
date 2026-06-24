# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .hypervec_index_io import index_file_info
    from .hypervec_meta_store import CollectionMeta, MetaStore
    from .hypervec_scalar_store import ScalarStore
except ImportError:  # pragma: no cover - supports direct file loading in tests
    sys.path.insert(0, str(Path(__file__).parent))
    from hypervec_index_io import index_file_info
    from hypervec_meta_store import CollectionMeta, MetaStore
    from hypervec_scalar_store import ScalarStore


def _load_bundle_module():
    try:
        from .hypervec_bundle import create_bundle, read_bundle, bundle_checksum, BUNDLE_FORMAT
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from hypervec_bundle import create_bundle, read_bundle, bundle_checksum, BUNDLE_FORMAT
    return create_bundle, read_bundle, bundle_checksum, BUNDLE_FORMAT


class HypervecServerEngine:
    INDEX_FILE = "index.hypervec"

    def __init__(
        self,
        data_root: str,
        *,
        logger: logging.Logger | None = None,
        hypervec_module: Any | None = None,
        meta_store: MetaStore | None = None,
        scalar_store: ScalarStore | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.collections_root = self.data_root / "collections"
        self.collections_root.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("hypervec.server")
        if hypervec_module is None:
            import hypervec as hypervec_module
        self.hypervec = hypervec_module
        self.meta_store = meta_store or MetaStore(self.data_root / "collections.json")
        self.scalar_store = scalar_store or ScalarStore(self.data_root / "scalar.db")
        self._indexes: dict[str, Any] = {}
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
        return self.collections_root / self.validate_collection_name(collection_name)

    def _index_path(self, collection_name: str) -> Path:
        return self._collection_dir(collection_name) / self.INDEX_FILE

    def _schema_fields(self, meta_or_manifest: CollectionMeta | dict[str, Any]) -> list[dict[str, Any]]:
        schema = meta_or_manifest.schema if isinstance(meta_or_manifest, CollectionMeta) else meta_or_manifest.get("schema", {})
        return list((schema or {}).get("fields") or [])

    def _field_name_by_datatype(
        self,
        meta_or_manifest: CollectionMeta | dict[str, Any],
        datatype: str,
        *,
        default: str,
    ) -> str:
        for field in self._schema_fields(meta_or_manifest):
            if str(field.get("datatype", "")).upper() == datatype.upper():
                return str(field.get("name"))
        return default

    def _id_field(self, meta_or_manifest: CollectionMeta | dict[str, Any]) -> str:
        for field in self._schema_fields(meta_or_manifest):
            if bool(field.get("is_primary", False)):
                return str(field.get("name"))
        if isinstance(meta_or_manifest, CollectionMeta):
            return meta_or_manifest.id_field
        return str(meta_or_manifest.get("id_field") or "id")

    def _vector_field(self, meta_or_manifest: CollectionMeta | dict[str, Any]) -> str:
        if isinstance(meta_or_manifest, CollectionMeta) and meta_or_manifest.vector_field:
            return meta_or_manifest.vector_field
        if isinstance(meta_or_manifest, dict) and meta_or_manifest.get("vector_field"):
            return str(meta_or_manifest["vector_field"])
        return self._field_name_by_datatype(meta_or_manifest, "FLOAT_VECTOR", default="vector")

    def _text_field(self, meta_or_manifest: CollectionMeta | dict[str, Any]) -> str:
        for field in self._schema_fields(meta_or_manifest):
            if str(field.get("name")) == "contents":
                return "contents"
        if isinstance(meta_or_manifest, CollectionMeta):
            return meta_or_manifest.text_field
        return str(meta_or_manifest.get("text_field") or "contents")

    def _index_config(self, meta: CollectionMeta) -> dict[str, Any]:
        indexes = (meta.index_params or {}).get("indexes") or []
        if indexes:
            return dict(indexes[0])
        return {
            "field_name": meta.vector_field,
            "metric_type": "L2",
            "index_type": "HNSWFlat",
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

    def _add_vectors(self, index: Any, vectors: np.ndarray) -> None:
        index.add(vectors)

    def _search_index(self, index: Any, query: np.ndarray, k: int) -> tuple[Any, Any]:
        return index.search(query, k)

    def _write_index(self, index: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        self.hypervec.write_index(index, str(tmp))
        tmp.replace(path)

    def _read_index(self, path: Path) -> Any:
        return self.hypervec.read_index(str(path))

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

    def _meta_or_raise(self, collection_name: str) -> CollectionMeta:
        meta = self.meta_store.get(collection_name)
        if meta is None:
            raise FileNotFoundError(f"collection '{collection_name}' does not exist.")
        return meta

    def _meta_response(self, meta: CollectionMeta) -> dict[str, Any]:
        fields = self._schema_fields(meta)
        manifest = meta.to_dict()
        manifest["backend"] = "hypervec-server"
        return {
            "collection_name": meta.collection_name,
            "schema": meta.schema,
            "index_params": meta.index_params,
            "fields": fields,
            "dim": meta.dim,
            "total": meta.total,
            "version": meta.version,
            "updated_at": meta.updated_at,
            "index_checksum": meta.index_checksum,
            "index_size_bytes": meta.index_size_bytes,
            # Bundle / purge state (new fields — old clients can safely ignore)
            "data_state": meta.data_state,
            "last_known_total": meta.last_known_total,
            "last_exported_at": meta.last_exported_at,
            "last_purged_at": meta.last_purged_at,
            "bundle_format": meta.bundle_format,
            "manifest": manifest,
        }

    def list_collections(self) -> list[str]:
        return sorted(meta.collection_name for meta in self.meta_store.list_all())

    def has_collection(self, collection_name: str) -> bool:
        collection_name = self.validate_collection_name(collection_name)
        return self.meta_store.get(collection_name) is not None

    def create_collection(
        self,
        collection_name: str,
        *,
        schema: dict[str, Any],
        index_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            if self.meta_store.get(collection_name) is not None:
                raise FileExistsError(f"collection '{collection_name}' already exists.")
            self._collection_dir(collection_name).mkdir(parents=True, exist_ok=True)
            manifest = {"schema": dict(schema)}
            meta = self.meta_store.create(
                collection_name,
                schema=dict(schema),
                index_params=dict(index_params or {"indexes": []}),
                id_field=self._id_field(manifest),
                vector_field=self._vector_field(manifest),
                text_field=self._text_field(manifest),
                index_path=str(self._index_path(collection_name)),
            )
            self.scalar_store.ensure_table(collection_name)
            return self._meta_response(meta)

    def drop_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            existed = self.meta_store.delete(collection_name)
            self._indexes.pop(collection_name, None)
            self.scalar_store.drop_table(collection_name)
            collection_dir = self._collection_dir(collection_name)
            if collection_dir.exists():
                shutil.rmtree(collection_dir)
            return {"dropped": True, "collection_name": collection_name, "existed": existed}

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        return self._meta_response(self._meta_or_raise(collection_name))

    def describe_collections(self) -> list[dict[str, Any]]:
        return [
            self._meta_response(meta)
            for meta in sorted(
                self.meta_store.list_all(),
                key=lambda item: item.collection_name,
            )
        ]

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            meta = self._meta_or_raise(collection_name)
            self.scalar_store.ensure_table(collection_name)
            dim = meta.dim
            rows = []
            next_row_id = self.scalar_store.next_row_id(collection_name)
            for i, row in enumerate(data):
                if meta.vector_field not in row:
                    raise ValueError(f"row is missing vector field '{meta.vector_field}'.")
                vector = np.asarray(row[meta.vector_field], dtype=np.float32)
                if vector.ndim != 1:
                    raise ValueError(f"row vector field '{meta.vector_field}' must be 1-D.")
                if dim is None:
                    dim = int(vector.size)
                elif int(dim) != int(vector.size):
                    raise ValueError(
                        f"vector dimension {vector.size} does not match collection dim {dim}."
                    )
                doc_id = row.get(meta.id_field, str(next_row_id + i))
                text_content = row.get(meta.text_field, "")
                structured_fields = {meta.id_field, meta.vector_field, meta.text_field}
                metadata = {
                    key: value for key, value in row.items() if key not in structured_fields
                }
                rows.append((next_row_id + i, str(doc_id), vector, str(text_content), metadata))
            self.scalar_store.insert_batch(collection_name, rows)
            total = self.scalar_store.count(collection_name)
            self.meta_store.update(collection_name, dim=dim, total=total)
            self._indexes.pop(collection_name, None)
            return {"insert_count": len(data), "total": total}

    def flush(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            meta = self._meta_or_raise(collection_name)
            if meta.dim is None:
                raise ValueError(f"collection '{collection_name}' has no rows.")
            vectors = self.scalar_store.get_vectors(collection_name, int(meta.dim))
            if vectors.size == 0:
                raise ValueError(f"collection '{collection_name}' has no rows.")
            index = self._make_index(int(vectors.shape[1]), self._index_config(meta))
            if not getattr(index, "is_trained", True):
                index.train(vectors)
            self._add_vectors(index, vectors)
            index_path = Path(meta.index_path)
            self._write_index(index, index_path)
            file_info = index_file_info(index_path)
            updated = self.meta_store.bump_version(
                collection_name,
                dim=int(vectors.shape[1]),
                total=int(vectors.shape[0]),
                flushed_at=time.time(),
                **file_info,
            )
            self._indexes[collection_name] = index
            return {
                "flushed": True,
                "collection_name": collection_name,
                "total": updated.total,
                "dim": updated.dim,
                "version": updated.version,
                "index_checksum": updated.index_checksum,
                "index_size_bytes": updated.index_size_bytes,
            }

    def load_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            meta = self._meta_or_raise(collection_name)
            index_path = Path(meta.index_path)
            if not index_path.exists():
                raise FileNotFoundError(
                    f"collection '{collection_name}' index has not been flushed."
                )
            self._indexes[collection_name] = self._read_index(index_path)
            return {
                "loaded": True,
                "collection_name": collection_name,
                "total": meta.total,
                "dim": meta.dim,
                "version": meta.version,
            }

    def close_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            self._indexes.pop(collection_name, None)
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
            meta = self._meta_or_raise(collection_name)
            if meta.dim is None:
                raise ValueError(f"collection '{collection_name}' has no vector dimension.")
            query = np.asarray(data, dtype=np.float32, order="C")
            if query.ndim != 2:
                raise ValueError("search data must be a 2-D matrix.")
            if int(query.shape[1]) != int(meta.dim):
                raise ValueError(f"query dim {query.shape[1]} != collection dim {meta.dim}.")

            index = self._indexes[collection_name]
            candidate_k = min(meta.total, max(int(limit), int(limit) * 8))
            distances, labels = self._search_index(index, query, candidate_k)
            requested = set(output_fields or [])
            results: list[list[dict[str, Any]]] = []
            for q_labels, q_distances in zip(labels, distances):
                row_ids = [int(label) for label in q_labels if int(label) >= 0]
                scalars = self.scalar_store.get_by_row_ids(collection_name, row_ids)
                hits = []
                for rank, (row_id, scalar) in enumerate(zip(row_ids, scalars)):
                    if scalar is None:
                        continue
                    row = {
                        **dict(scalar["metadata"] or {}),
                        meta.id_field: scalar["doc_id"],
                        meta.text_field: scalar["text_content"],
                    }
                    if not self._filter_match(row, filter):
                        continue
                    if requested:
                        entity = {key: value for key, value in row.items() if key in requested}
                    else:
                        entity = dict(row)
                    distance_index = list(q_labels).index(row_id)
                    hits.append(
                        {
                            "id": row.get(meta.id_field, row_id),
                            "distance": float(q_distances[distance_index]),
                            "entity": entity,
                        }
                    )
                    if len(hits) >= int(limit):
                        break
                results.append(hits)
            return results

    def get_version(self, collection_name: str) -> dict[str, Any]:
        meta = self._meta_or_raise(self.validate_collection_name(collection_name))
        return {
            "collection_name": meta.collection_name,
            "version": meta.version,
            "updated_at": meta.updated_at,
            "total": meta.total,
            "dim": meta.dim,
            "index_checksum": meta.index_checksum,
            "index_size_bytes": meta.index_size_bytes,
            "data_state": meta.data_state,
            "last_known_total": meta.last_known_total,
            "last_exported_at": meta.last_exported_at,
            "last_purged_at": meta.last_purged_at,
        }

    def sync_check(
        self,
        collection_name: str,
        *,
        client_version: int,
        client_checksum: str | None = None,
    ) -> dict[str, Any]:
        meta = self._meta_or_raise(self.validate_collection_name(collection_name))
        needs_sync = int(client_version) != int(meta.version)
        if client_checksum and meta.index_checksum:
            needs_sync = needs_sync or client_checksum != meta.index_checksum
        return {
            "needs_sync": needs_sync,
            "server_version": meta.version,
            "client_version": int(client_version),
            "download_url": f"/collections/{collection_name}/index",
            "index_checksum": meta.index_checksum,
            "index_size_bytes": meta.index_size_bytes,
        }

    def index_path_for_download(self, collection_name: str) -> Path:
        meta = self._meta_or_raise(self.validate_collection_name(collection_name))
        path = Path(meta.index_path)
        if not path.exists():
            raise FileNotFoundError(f"collection '{collection_name}' index is not available.")
        return path

    def upload_index(
        self,
        collection_name: str,
        source_path: str | Path,
        *,
        version: int | None = None,
        checksum: str | None = None,
    ) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            meta = self._meta_or_raise(collection_name)
            if version is not None and int(version) < int(meta.version):
                raise ValueError(
                    f"uploaded index version {version} is older than server version {meta.version}."
                )
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"uploaded index file does not exist: {source}")
            actual_checksum = index_file_info(source)["index_checksum"]
            if checksum and checksum != actual_checksum:
                raise ValueError("uploaded index checksum does not match request checksum.")
            loaded = self._read_index(source)
            target = Path(meta.index_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".upload.tmp")
            shutil.copyfile(source, tmp)
            tmp.replace(target)
            file_info = index_file_info(target)
            new_version = int(version) if version is not None else meta.version + 1
            updated = self.meta_store.set_version(
                collection_name,
                max(new_version, meta.version),
                flushed_at=time.time(),
                **file_info,
            )
            self._indexes[collection_name] = loaded
            return {
                "uploaded": True,
                "collection_name": collection_name,
                "version": updated.version,
                "index_checksum": updated.index_checksum,
                "index_size_bytes": updated.index_size_bytes,
            }

    # ------------------------------------------------------------------
    # Bundle export / import / purge
    # ------------------------------------------------------------------

    def export_collection_bundle(
        self,
        collection_name: str,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Export scalar rows + index as a self-contained bundle ZIP.

        If output_path is None, the bundle is written alongside the index file
        as {collection_dir}/{collection_name}.hypervec-bundle.
        Updates last_exported_at and bundle_format in metadata.
        Raises FileNotFoundError if the collection has no flushed index.
        Raises ValueError if data_state == "purged" (nothing to export).
        """
        create_bundle, _, _, BUNDLE_FORMAT = _load_bundle_module()
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            meta = self._meta_or_raise(collection_name)
            if meta.data_state == "purged":
                raise ValueError(
                    f"collection '{collection_name}' data has been purged — "
                    "nothing to export."
                )
            index_path = Path(meta.index_path)
            if not index_path.exists():
                raise FileNotFoundError(
                    f"collection '{collection_name}' index has not been flushed; "
                    "call flush() before exporting a bundle."
                )
            scalar_rows = self.scalar_store.export_rows(collection_name)
            if output_path is None:
                output_path = (
                    self._collection_dir(collection_name)
                    / f"{collection_name}.hypervec-bundle"
                )
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = create_bundle(
                collection_name, index_path, scalar_rows, meta, output_path
            )
            bundle_size = output_path.stat().st_size
            import hashlib as _hashlib
            bundle_cksum = "sha256:" + _hashlib.sha256(output_path.read_bytes()).hexdigest()
            self.meta_store.update(
                collection_name,
                last_exported_at=manifest["exported_at"],
                last_known_total=len(scalar_rows),
                bundle_format=BUNDLE_FORMAT,
            )
            return {
                "collection_name": collection_name,
                "path": str(output_path),
                "bytes": bundle_size,
                "version": meta.version,
                "bundle_format": BUNDLE_FORMAT,
                "bundle_checksum": bundle_cksum,
                "manifest": manifest,
            }

    def import_collection_bundle(
        self,
        collection_name: str,
        source_path: str | Path,
        *,
        checksum: str | None = None,
        mode: str = "replace",
    ) -> dict[str, Any]:
        """Restore a collection from a previously exported bundle.

        - Verifies the bundle's collection_name matches the target.
        - Verifies the optional bundle-level checksum (sha256:...) if provided.
        - Writes index.hypervec to the collection's index_path.
        - Restores scalar rows (with replace=True by default).
        - Updates data_state to "ready".
        Raises FileNotFoundError if the collection metadata does not exist.
        Raises ValueError on format errors or checksum mismatches.
        """
        _, read_bundle, _, BUNDLE_FORMAT = _load_bundle_module()
        collection_name = self.validate_collection_name(collection_name)
        source_path = Path(source_path)

        if checksum:
            import hashlib as _hashlib
            actual = "sha256:" + _hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual != checksum:
                raise ValueError(
                    f"bundle checksum mismatch: expected {checksum}, got {actual}"
                )

        manifest, index_bytes, scalar_rows = read_bundle(source_path)

        if manifest.get("collection_name") != collection_name:
            raise ValueError(
                f"bundle collection_name '{manifest.get('collection_name')}' "
                f"does not match target '{collection_name}'."
            )

        with self._lock_for(collection_name):
            meta = self._meta_or_raise(collection_name)

            if meta.dim is not None and manifest.get("dim") is not None:
                if int(meta.dim) != int(manifest["dim"]):
                    raise ValueError(
                        f"bundle dim {manifest['dim']} does not match "
                        f"collection dim {meta.dim}."
                    )

            index_path = Path(meta.index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = index_path.with_suffix(index_path.suffix + ".import.tmp")
            try:
                tmp.write_bytes(index_bytes)
                loaded = self._read_index(tmp)
                tmp.replace(index_path)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise

            self.scalar_store.import_rows(
                collection_name,
                scalar_rows,
                replace=(mode == "replace"),
            )

            file_info = index_file_info(index_path)
            updated = self.meta_store.bump_version(
                collection_name,
                dim=manifest.get("dim") or meta.dim,
                total=len(scalar_rows),
                flushed_at=time.time(),
                data_state="ready",
                last_known_total=len(scalar_rows),
                **file_info,
            )
            self._indexes[collection_name] = loaded
            return {
                "uploaded": True,
                "collection_name": collection_name,
                "version": updated.version,
                "total": len(scalar_rows),
                "dim": updated.dim,
                "data_state": updated.data_state,
                "index_checksum": updated.index_checksum,
                "index_size_bytes": updated.index_size_bytes,
            }

    def purge_collection_data(
        self,
        collection_name: str,
        *,
        require_exported: bool = True,
    ) -> dict[str, Any]:
        """Delete user data (index file + scalar rows) while keeping metadata.

        This is NOT drop_collection — the collection entry in collections.json
        is preserved so users can re-identify their collections after logout.

        require_exported=True (default): refuse to purge if last_exported_at is
        not set on the metadata, preventing accidental data loss when the
        bundle download step was skipped.

        Security note: SQLite DROP + VACUUM + secure_delete reduces plain-file
        residue but is not a cryptographic erase.  SSD wear-levelling, OS
        file-system journals, and system-level snapshots may retain data at the
        block level.
        """
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name):
            meta = self._meta_or_raise(collection_name)
            if require_exported and not meta.last_exported_at:
                raise ValueError(
                    f"collection '{collection_name}' has no recorded export; "
                    "call export_collection_bundle() first, or pass "
                    "require_exported=False to force purge."
                )
            last_known_total = self.scalar_store.count(collection_name)

            # Evict from memory
            self._indexes.pop(collection_name, None)

            # Delete index file(s)
            index_path = Path(meta.index_path)
            index_path.unlink(missing_ok=True)
            for leftover in index_path.parent.glob("*.tmp"):
                leftover.unlink(missing_ok=True)

            # Purge scalar table
            self.scalar_store.purge_collection_rows(collection_name)
            self.scalar_store.checkpoint_and_vacuum()

            purged_at = time.time()
            self.meta_store.update(
                collection_name,
                data_state="purged",
                last_purged_at=purged_at,
                last_known_total=last_known_total,
                # Reset index file info so describe reflects no live index
                index_checksum=None,
                index_size_bytes=None,
                flushed_at=None,
                total=0,
            )
            return {
                "purged": True,
                "collection_name": collection_name,
                "metadata_preserved": True,
                "scalar_deleted": True,
                "index_deleted": True,
                "memory_unloaded": True,
                "data_state": "purged",
                "last_known_total": last_known_total,
                "last_purged_at": purged_at,
            }
