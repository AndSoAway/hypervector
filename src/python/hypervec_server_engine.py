# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .hypervec_index_io import index_file_info, file_sha256
    from .hypervec_meta_store import CollectionMeta, MetaStore
    from .rwlock import RWLock
    from .hypervec_scalar_store import ScalarStore
except ImportError:  # pragma: no cover - supports direct file loading in tests
    sys.path.insert(0, str(Path(__file__).parent))
    from hypervec_index_io import index_file_info, file_sha256
    from hypervec_meta_store import CollectionMeta, MetaStore
    from rwlock import RWLock
    from hypervec_scalar_store import ScalarStore


def _load_bundle_module():
    try:
        from .hypervec_bundle import (
            create_bundle,
            read_bundle,
            bundle_checksum,
            schema_checksum,
            BUNDLE_FORMAT,
        )
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from hypervec_bundle import (
            create_bundle,
            read_bundle,
            bundle_checksum,
            schema_checksum,
            BUNDLE_FORMAT,
        )
    return create_bundle, read_bundle, bundle_checksum, schema_checksum, BUNDLE_FORMAT


class ConflictError(Exception):
    """Raised when an operation conflicts with the current collection state.

    Maps to HTTP 409 Conflict.  Examples: purging before exporting, uploading
    a bundle whose collection_name does not match the target, or trying to
    export from a collection that has already been purged.
    """


class HypervecServerEngine:
    INDEX_FILE = "index.hypervec"
    _IMPORT_STAGING_SUFFIX = ".import.staging"
    _PRE_IMPORT_SUFFIX = ".pre-import"

    _INDEX_EXAMPLES: tuple[dict[str, Any], ...] = (
        {
            "index_type": "IndexIVFFlat",
            "name": "IVFFlat",
            "full_name": "Inverted File Flat Index",
            "description": "倒排聚类索引，通过只搜索部分聚类降低查询开销。",
            "use_case": ["大规模向量粗召回", "可接受近似结果的搜索"],
            "advantages": ["查询成本可控", "适合大规模数据"],
            "limitations": ["需要训练", "召回受 nprobe 影响"],
            "parameters": [
                {"name": "nlist", "type": "int", "default": 1024, "required": False, "description": "聚类中心数"},
                {"name": "nprobe", "type": "int", "default": 10, "required": False, "description": "查询探测聚类数"},
            ],
            "example_code": {
                "Python": {
                    "create": "index_params.add_index(field_name='vector', index_type='IVFFlat', metric_type='L2', params={'nlist': 1024})",
                    "search": "client.search(collection_name='demo_ivf_flat', data=[query], limit=10, search_params={'nprobe': 16})",
                }
            },
            "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 nlist 可提升粗聚类粒度但增加训练和索引开销"],
            "metric_types": ["L2", "IP", "COSINE"],
        },
        {
            "index_type": "IndexIVFLVQ",
            "name": "IVFLVQ",
            "full_name": "Inverted File with LVQ",
            "description": "倒排索引结合 LVQ 量化，兼顾压缩和查询效率。",
            "use_case": ["大规模压缩检索", "内存受限场景"],
            "advantages": ["压缩率高", "适合批量检索"],
            "limitations": ["参数调优复杂", "存在量化误差"],
            "parameters": [
                {"name": "nlist", "type": "int", "default": 1024, "required": False, "description": "聚类中心数"},
                {"name": "nlocal", "type": "int", "default": 16, "required": False, "description": "局部量化参数"},
                {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "量化位数"},
            ],
            "example_code": {
                "Python": {
                    "create": "index_params.add_index(field_name='vector', index_type='IVFLVQ', metric_type='L2', params={'nlist': 1024, 'nlocal': 16, 'nbits': 8})",
                    "search": "client.search(collection_name='demo_ivf_lvq', data=[query], limit=10, search_params={'nprobe': 16})",
                }
            },
            "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 nlocal 和 nbits 会影响压缩率与精度的平衡"],
            "metric_types": ["L2"],
        },
        {
            "index_type": "IndexIVFPQ",
            "name": "IVFPQ",
            "full_name": "Inverted File with Product Quantization",
            "description": "倒排索引结合乘积量化，降低内存占用。",
            "use_case": ["超大规模向量检索", "内存敏感场景"],
            "advantages": ["内存占用低", "查询速度快"],
            "limitations": ["量化会损失精度", "需要训练"],
            "parameters": [
                {"name": "nlist", "type": "int", "default": 1024, "required": False, "description": "聚类中心数"},
                {"name": "m_pq", "type": "int", "default": 8, "required": False, "description": "子量化器数量"},
                {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "编码位数"},
            ],
            "example_code": {
                "Python": {
                    "create": "index_params.add_index(field_name='vector', index_type='IVFPQ', metric_type='L2', params={'nlist': 1024, 'm_pq': 8, 'nbits': 8})",
                    "search": "client.search(collection_name='demo_ivf_pq', data=[query], limit=10, search_params={'nprobe': 16})",
                }
            },
            "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 m_pq 会降低单码压缩比并改善重构精度"],
            "metric_types": ["L2"],
        },
        {
            "index_type": "IndexHNSWFlat",
            "full_name": "Hierarchical Navigable Small World with Flat Vectors",
            "description": "基于多层小世界图的近似最近邻索引，适合高召回、低延迟向量检索。",
            "use_case": ["百万级以上向量检索", "低延迟在线搜索", "高召回召回阶段"],
            "advantages": ["查询速度快", "召回率高", "无需训练"],
            "limitations": ["索引内存占用较高", "构建耗时随 M 和 ef_construction 增加"],
            "parameters": [
                {"name": "m_hnsw", "type": "int", "default": 32, "required": False, "description": "图连接数"},
                {"name": "ef_construction", "type": "int", "default": 100, "required": False, "description": "构建搜索宽度"},
                {"name": "ef_search", "type": "int", "default": 100, "required": False, "description": "查询搜索宽度"},
            ],
            "example_code": {"Python": {"create": "index_params.add_index(field_name='vector', index_type='HNSW', metric_type='L2', params={'m_hnsw': 32, 'ef_construction': 200})", "search": "client.search(collection_name='wiki_hnsw_1m', data=[query], limit=10, search_params={'ef_search': 128})"}},
            "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存"],
            "metric_types": ["L2", "IP", "COSINE"],
        },
        {
            "index_type": "IndexHNSWLVQ",
            "name": "HNSWLVQ",
            "full_name": "Hierarchical Navigable Small World with LVQ",
            "description": "基于多层小世界图的近似最近邻索引，结合 LVQ 压缩以降低内存占用，适合高召回、较低内存场景。",
            "use_case": ["大规模向量近似检索", "内存受限场景", "高召回检索"],
            "advantages": ["查询速度快", "召回率高", "索引占用低于纯浮点 HNSW"],
            "limitations": ["仅支持 L2", "存在量化误差", "构建耗时随 m_hnsw 增加"],
            "parameters": [
                {"name": "nlocal", "type": "int", "default": 16, "required": False, "description": "局部量化参数"},
                {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "量化位数"},
                {"name": "m_hnsw", "type": "int", "default": 32, "required": False, "description": "图连接数"},
            ],
            "example_code": {
                "Python": {
                    "create": "index_params.add_index(field_name='vector', index_type='HNSWLVQ', metric_type='L2', params={'nlocal': 16, 'nbits': 8, 'm_hnsw': 32})",
                    "search": "client.search(collection_name='wiki_hnsw_lvq', data=[query], limit=10, search_params={'ef_search': 128})",
                }
            },
            "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存"],
            "metric_types": ["L2"],
        },
        {
            "index_type": "IndexHNSWPQ",
            "name": "HNSWPQ",
            "full_name": "Hierarchical Navigable Small World with Product Quantization",
            "description": "基于多层小世界图的近似最近邻索引，结合 PQ 压缩以降低内存占用，适合超大规模向量检索。",
            "use_case": ["超大规模向量检索", "内存敏感场景", "高召回检索"],
            "advantages": ["内存占用低", "查询速度快", "索引规模可扩展"],
            "limitations": ["仅支持 L2", "量化会损失精度", "要求维度可被 m_pq 整除"],
            "parameters": [
                {"name": "m_pq", "type": "int", "default": 8, "required": False, "description": "子量化器数量"},
                {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "编码位数"},
                {"name": "m_hnsw", "type": "int", "default": 32, "required": False, "description": "图连接数"},
            ],
            "example_code": {
                "Python": {
                    "create": "index_params.add_index(field_name='vector', index_type='HNSWPQ', metric_type='L2', params={'m_pq': 8, 'nbits': 8, 'm_hnsw': 32})",
                    "search": "client.search(collection_name='wiki_hnsw_pq', data=[query], limit=10, search_params={'ef_search': 128})",
                }
            },
            "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存"],
            "metric_types": ["L2"],
        },
    )

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
        self._scalar_cache: dict[str, dict[int, dict[str, Any]]] = {}
        self._scalar_cache_max_rows = int(os.getenv("HYPERVEC_SCALAR_CACHE_MAX_ROWS", "1000000"))
        self._locks: dict[str, RWLock] = {}
        self._global_lock = threading.RLock()
        self._recover_interrupted_imports()

    def _recover_interrupted_imports(self) -> None:
        """Resolve bundle imports interrupted by a crash/restart.

        For each collection whose metadata is still marked "importing" (or that
        carries an import_txn record), decide whether the commit switch had
        completed and roll forward, otherwise roll back to the pre-import state.
        Idempotent — safe to run on every startup.
        """
        for meta in self.meta_store.list_all():
            if meta.data_state != "importing" and not meta.import_txn:
                continue
            name = meta.collection_name
            txn = meta.import_txn or {}
            index_path = Path(meta.index_path)
            staging_index = index_path.with_suffix(
                index_path.suffix + self._IMPORT_STAGING_SUFFIX
            )
            pre_import = index_path.with_suffix(
                index_path.suffix + self._PRE_IMPORT_SUFFIX
            )
            expected_total = txn.get("total")
            new_data_version = txn.get("new_data_version", meta.data_version + 1)

            switch_done = (
                not self.scalar_store.has_staging(name)
                and index_path.exists()
                and (
                    expected_total is None
                    or self.scalar_store.count(name) == int(expected_total)
                )
            )
            if switch_done:
                # Roll forward: the atomic swaps completed; only the metadata
                # finalize was lost.
                file_info = index_file_info(index_path)
                self.meta_store.bump_version(
                    name,
                    dim=txn.get("dim") or meta.dim,
                    total=int(expected_total) if expected_total is not None else meta.total,
                    flushed_at=time.time(),
                    data_state="ready",
                    last_known_total=int(expected_total) if expected_total is not None else meta.last_known_total,
                    data_version=new_data_version,
                    index_version=new_data_version,
                    import_txn=None,
                    **file_info,
                )
                staging_index.unlink(missing_ok=True)
                pre_import.unlink(missing_ok=True)
                self.logger.warning(
                    "recovered interrupted import for '%s' (rolled forward).", name
                )
                continue

            # Roll back: discard staging, restore the pre-import index if we had
            # renamed it aside.
            self.scalar_store.rollback_staging(name)
            staging_index.unlink(missing_ok=True)
            if pre_import.exists():
                pre_import.replace(index_path)
                restored_state = txn.get("prev_state") or "ready"
                self.meta_store.update(name, data_state=restored_state, import_txn=None)
                self.logger.warning(
                    "recovered interrupted import for '%s' (rolled back).", name
                )
            elif index_path.exists():
                restored_state = txn.get("prev_state") or "ready"
                self.meta_store.update(name, data_state=restored_state, import_txn=None)
                self.logger.warning(
                    "recovered interrupted import for '%s' (kept live index).", name
                )
            else:
                # Neither a live nor a pre-import index survived.
                self.meta_store.update(name, data_state="invalid", import_txn=None)
                self.logger.error(
                    "interrupted import for '%s' left no recoverable index; "
                    "marked invalid.", name
                )

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

    def _lock_for(self, collection_name: str) -> RWLock:
        with self._global_lock:
            return self._locks.setdefault(collection_name, RWLock())

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

    def supported_index_examples(self) -> list[dict[str, Any]]:
        examples = []
        for example in self._INDEX_EXAMPLES:
            if hasattr(self.hypervec, example["index_type"]):
                examples.append(dict(example))
        return examples

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
        deprecated = sorted(set(params) & {"M", "m", "M_hnsw", "M_pq"})
        if deprecated:
            raise ValueError(
                "unsupported index parameter(s) "
                f"{', '.join(deprecated)}; use explicit m_hnsw or m_pq."
            )

        def positive_int(name: str, default: int) -> int:
            value = int(params.get(name, default))
            if value <= 0:
                raise ValueError(f"index parameter '{name}' must be positive.")
            return value

        def validate_pq_dim(m_pq: int) -> None:
            if int(dim) % int(m_pq) != 0:
                raise ValueError(
                    f"vector dim {dim} must be divisible by m_pq {m_pq}."
                )

        if index_type in {"FLAT", "INDEXFLAT"}:
            if metric == int(self.hypervec.kMetricInnerProduct):
                return self.hypervec.IndexFlatIP(dim)
            return self.hypervec.IndexFlatL2(dim)
        if index_type in {"IVF", "IVFFLAT", "INDEXIVFFLAT"}:
            nlist = positive_int("nlist", 1024)
            return self.hypervec.IndexIVFFlat(dim, nlist, metric)
        if index_type in {"IVFLVQ", "INDEXIVFLVQ"}:
            nlist = positive_int("nlist", 1024)
            nlocal = positive_int("nlocal", 16)
            nbits = positive_int("nbits", 8)
            return self.hypervec.IndexIVFLVQ(dim, nlist, nlocal, nbits, metric)
        if index_type in {"IVFPQ", "INDEXIVFPQ"}:
            nlist = positive_int("nlist", 1024)
            m_pq = positive_int("m_pq", 8)
            nbits = positive_int("nbits", 8)
            validate_pq_dim(m_pq)
            return self.hypervec.IndexIVFPQ(dim, nlist, m_pq, nbits, metric)
        if index_type in {"HNSW", "HNSWFLAT", "INDEXHNSWFLAT", "AUTOINDEX"}:
            m_hnsw = positive_int("m_hnsw", 32)
            return self.hypervec.IndexHNSWFlat(dim, m_hnsw, metric)
        if index_type in {"HNSWLVQ", "INDEXHNSWLVQ"}:
            nlocal = positive_int("nlocal", 16)
            nbits = positive_int("nbits", 8)
            m_hnsw = positive_int("m_hnsw", 32)
            return self.hypervec.IndexHNSWLVQ(dim, nlocal, nbits, m_hnsw, metric)
        if index_type in {"HNSWPQ", "INDEXHNSWPQ"}:
            m_pq = positive_int("m_pq", 8)
            nbits = positive_int("nbits", 8)
            m_hnsw = positive_int("m_hnsw", 32)
            validate_pq_dim(m_pq)
            return self.hypervec.IndexHNSWPQ(dim, m_pq, nbits, m_hnsw, metric)
        raise ValueError(f"unsupported index_type: {index_config.get('index_type')}")

    def _add_vectors(self, index: Any, vectors: np.ndarray) -> None:
        index.add(vectors)

    def _search_index(
        self,
        index: Any,
        query: np.ndarray,
        k: int,
        search_params: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        params = dict(search_params or {})
        ef_search = params.get("ef_search", params.get("ef"))
        if ef_search is not None and hasattr(index, "search_with_ef"):
            return index.search_with_ef(query, k, int(ef_search))
        nprobe = params.get("nprobe")
        if nprobe is not None and hasattr(index, "search_with_nprobe"):
            return index.search_with_nprobe(query, k, int(nprobe))
        return index.search(query, k)

    def _write_index(self, index: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        self.hypervec.write_index(index, str(tmp))
        tmp.replace(path)

    def _read_index(self, path: Path) -> Any:
        return self.hypervec.read_index(str(path))

    def _refresh_scalar_cache(self, collection_name: str, meta: CollectionMeta) -> None:
        if self._scalar_cache_max_rows <= 0:
            self._scalar_cache.pop(collection_name, None)
            return
        if meta.total is not None and int(meta.total) > self._scalar_cache_max_rows:
            self._scalar_cache.pop(collection_name, None)
            self.logger.info(
                "Skipping scalar cache for collection '%s': %d rows exceeds limit %d.",
                collection_name,
                int(meta.total),
                self._scalar_cache_max_rows,
            )
            return
        self._scalar_cache[collection_name] = self.scalar_store.load_all_scalars(collection_name)

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
            "data_version": meta.data_version,
            "index_version": meta.index_version,
            "exported_data_version": meta.exported_data_version,
            "exported_index_version": meta.exported_index_version,
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
        with self._lock_for(collection_name).write_lock():
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
        with self._lock_for(collection_name).write_lock():
            existed = self.meta_store.delete(collection_name)
            self._indexes.pop(collection_name, None)
            self._scalar_cache.pop(collection_name, None)
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
        with self._lock_for(collection_name).write_lock():
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

            # Crash-window safety (PR13-3.3): the scalar write (SQLite) and the
            # data_version bump (collections.json) cannot commit in one atomic
            # transaction.  Bump data_version FIRST so that if we crash between
            # the two writes, export eligibility is already invalidated
            # (exported_data_version != data_version) and purge is refused —
            # failing safe (a spurious re-export) rather than unsafe (purging
            # rows that were just added but never exported).
            self.meta_store.update(
                collection_name,
                dim=dim,
                data_version=meta.data_version + 1,
            )
            self.scalar_store.insert_batch(collection_name, rows)
            total = self.scalar_store.count(collection_name)
            self.meta_store.update(collection_name, total=total)
            self._indexes.pop(collection_name, None)
            self._scalar_cache.pop(collection_name, None)
            return {"insert_count": len(data), "total": total}

    def flush(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name).write_lock():
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
                index_version=meta.data_version,
                **file_info,
            )
            self._indexes[collection_name] = index
            self._refresh_scalar_cache(collection_name, updated)
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
        with self._lock_for(collection_name).write_lock():
            meta = self._meta_or_raise(collection_name)
            index_path = Path(meta.index_path)
            if not index_path.exists():
                raise FileNotFoundError(
                    f"collection '{collection_name}' index has not been flushed."
                )
            return self._load_collection_unlocked(collection_name, meta=meta)

    def _load_collection_unlocked(
        self,
        collection_name: str,
        *,
        meta: CollectionMeta | None = None,
    ) -> dict[str, Any]:
        meta = meta or self._meta_or_raise(collection_name)
        self._indexes[collection_name] = self._read_index(Path(meta.index_path))
        self._refresh_scalar_cache(collection_name, meta)
        return {
            "loaded": True,
            "collection_name": collection_name,
            "total": meta.total,
            "dim": meta.dim,
            "version": meta.version,
        }

    def close_collection(self, collection_name: str) -> dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name).write_lock():
            self._indexes.pop(collection_name, None)
            self._scalar_cache.pop(collection_name, None)
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
        del consistency_level
        collection_name = self.validate_collection_name(collection_name)
        if int(limit) <= 0:
            raise ValueError("limit must be positive.")
        lock = self._lock_for(collection_name)
        if collection_name not in self._indexes:
            with lock.write_lock():
                if collection_name not in self._indexes:
                    meta = self._meta_or_raise(collection_name)
                    if meta.data_state == "invalid":
                        raise ConflictError(
                            f"collection '{collection_name}' is in 'invalid' state; "
                            "restore a valid bundle before searching."
                        )
                    index_path = Path(meta.index_path)
                    if not index_path.exists():
                        raise FileNotFoundError(
                            f"collection '{collection_name}' index has not been flushed."
                        )
                    self._load_collection_unlocked(collection_name, meta=meta)

        with lock.read_lock():
            meta = self._meta_or_raise(collection_name)
            if meta.dim is None:
                raise ValueError(f"collection '{collection_name}' has no vector dimension.")
            query = np.asarray(data, dtype=np.float32, order="C")
            if query.ndim != 2:
                raise ValueError("search data must be a 2-D matrix.")
            if int(query.shape[1]) != int(meta.dim):
                raise ValueError(f"query dim {query.shape[1]} != collection dim {meta.dim}.")

            index = self._indexes[collection_name]
            if filter:
                candidate_k = min(meta.total, max(int(limit), int(limit) * 8))
            else:
                candidate_k = min(meta.total, int(limit))
            distances, labels = self._search_index(index, query, candidate_k, search_params)
            requested = set(output_fields or [])
            cache = self._scalar_cache.get(collection_name)
            results: list[list[dict[str, Any]]] = []
            for q_labels, q_distances in zip(labels, distances):
                pairs = [
                    (int(label), float(distance))
                    for label, distance in zip(q_labels, q_distances)
                    if int(label) >= 0
                ]
                row_ids = [row_id for row_id, _ in pairs]
                if cache is not None:
                    scalars = [cache.get(row_id) for row_id in row_ids]
                else:
                    scalars = self.scalar_store.get_by_row_ids(collection_name, row_ids)
                hits = []
                for (row_id, distance), scalar in zip(pairs, scalars):
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
                    hits.append(
                        {
                            "id": row.get(meta.id_field, row_id),
                            "distance": distance,
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
            "bundle_format": meta.bundle_format,
            "data_version": meta.data_version,
            "index_version": meta.index_version,
            "exported_data_version": meta.exported_data_version,
            "exported_index_version": meta.exported_index_version,
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
        with self._lock_for(collection_name).write_lock():
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
                index_version=meta.data_version,
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
        Raises ConflictError if data_state == "purged" (nothing to export),
        if the index is stale (index_version != data_version), or if the
        index / scalar / metadata row counts and dimensions do not agree.
        """
        create_bundle, _, _, _, BUNDLE_FORMAT = _load_bundle_module()
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name).write_lock():
            meta = self._meta_or_raise(collection_name)
            if meta.data_state == "purged":
                raise ConflictError(
                    f"collection '{collection_name}' data has been purged — "
                    "nothing to export."
                )
            if meta.data_state in ("importing", "invalid"):
                raise ConflictError(
                    f"collection '{collection_name}' is in '{meta.data_state}' "
                    "state and cannot be exported; restore a valid bundle first."
                )
            index_path = Path(meta.index_path)
            if not index_path.exists():
                raise FileNotFoundError(
                    f"collection '{collection_name}' index has not been flushed; "
                    "call flush() before exporting a bundle."
                )
            # P0-1 freshness gate: refuse to export a bundle whose index does
            # not reflect the current data (e.g. insert-after-flush).  Otherwise
            # the bundle could ship index.n_total != scalar rows.
            if meta.index_version != meta.data_version:
                raise ConflictError(
                    f"collection '{collection_name}' index is stale "
                    f"(index_version={meta.index_version} != "
                    f"data_version={meta.data_version}); call flush() before "
                    "exporting a bundle."
                )
            # P0-1 consistency triad: index.n_total == scalar count == meta.total
            # and index.d == meta.dim, verified against the live index object.
            index = self._indexes.get(collection_name)
            if index is None:
                index = self._read_index(index_path)
                self._indexes[collection_name] = index
            scalar_count = self.scalar_store.count(collection_name)
            index_total = int(index.n_total)
            if not (index_total == scalar_count == int(meta.total or 0)):
                raise ConflictError(
                    f"collection '{collection_name}' is inconsistent: "
                    f"index.n_total={index_total}, scalar_count={scalar_count}, "
                    f"meta.total={meta.total}; call flush() to rebuild the index."
                )
            if meta.dim is not None and int(index.d) != int(meta.dim):
                raise ConflictError(
                    f"collection '{collection_name}' dimension mismatch: "
                    f"index.d={index.d} != meta.dim={meta.dim}."
                )
            scalar_rows = self.scalar_store.export_rows(collection_name)
            # When no explicit destination is given (the HTTP download path),
            # build into a controlled temp subdir so a failed/cancelled export
            # never leaves a stray bundle in the collection root.  purge sweeps
            # this directory unconditionally.
            export_tmp_dir: Path | None = None
            if output_path is None:
                export_tmp_dir = self._collection_dir(collection_name) / ".export.tmp"
                export_tmp_dir.mkdir(parents=True, exist_ok=True)
                output_path = export_tmp_dir / f"{collection_name}.hypervec-bundle"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                manifest = create_bundle(
                    collection_name, index_path, scalar_rows, meta, output_path
                )
                # Defense in depth: the manifest is derived from meta, so confirm
                # it agrees with the freshly-observed index and scalar counts.
                if not (
                    int(manifest["total"]) == scalar_count
                    and (meta.dim is None or int(manifest["dim"]) == int(index.d))
                ):
                    raise ConflictError(
                        f"collection '{collection_name}' bundle manifest disagrees "
                        f"with observed state (manifest.total={manifest['total']}, "
                        f"scalar_count={scalar_count})."
                    )
                bundle_size = output_path.stat().st_size
                bundle_cksum = file_sha256(output_path)
            except BaseException:
                # Any failure (or cancellation) removes the partial artifact and
                # the controlled temp dir so no residue survives.
                output_path.unlink(missing_ok=True)
                if export_tmp_dir is not None:
                    shutil.rmtree(export_tmp_dir, ignore_errors=True)
                raise
            self.meta_store.update(
                collection_name,
                last_exported_at=manifest["exported_at"],
                last_known_total=len(scalar_rows),
                bundle_format=BUNDLE_FORMAT,
                exported_data_version=meta.data_version,
                exported_bundle_checksum=bundle_cksum,
                exported_index_version=meta.index_version,
                exported_index_checksum=meta.index_checksum,
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

        Transactional restore: the index is written to a staging file and the
        scalar rows to a staging table, both leaving the live state untouched.
        Only after everything validates does a commit step atomically swap the
        index file, scalar table, and metadata into place.  Any failure before
        the swap leaves the collection exactly as it was; a crash mid-swap is
        resolved on the next startup by _recover_interrupted_imports().

        Raises FileNotFoundError if the collection metadata does not exist.
        Raises ValueError on format errors or checksum mismatches.
        Raises ConflictError on collection_name / dim mismatch.
        """
        _, read_bundle, _, schema_checksum, BUNDLE_FORMAT = _load_bundle_module()
        collection_name = self.validate_collection_name(collection_name)
        source_path = Path(source_path)

        # ---- Validate everything BEFORE touching any live state ----
        if mode != "replace":
            raise ValueError(
                f"unsupported import mode '{mode}'; only 'replace' is supported."
            )

        if checksum:
            actual = file_sha256(source_path)
            if actual != checksum:
                raise ValueError(
                    f"bundle checksum mismatch: expected {checksum}, got {actual}"
                )

        manifest, index_bytes, scalar_rows = read_bundle(source_path)

        if manifest.get("collection_name") != collection_name:
            raise ConflictError(
                f"bundle collection_name '{manifest.get('collection_name')}' "
                f"does not match target '{collection_name}'."
            )

        with self._lock_for(collection_name).write_lock():
            meta = self._meta_or_raise(collection_name)

            if meta.dim is not None and manifest.get("dim") is not None:
                if int(meta.dim) != int(manifest["dim"]):
                    raise ConflictError(
                        f"bundle dim {manifest['dim']} does not match "
                        f"collection dim {meta.dim}."
                    )

            # Schema compatibility: the bundle's schema must match the target
            # collection's, verified via the same deterministic checksum used
            # when the bundle was created.
            target_schema_cksum = schema_checksum(meta.schema)
            if manifest.get("schema_checksum") != target_schema_cksum:
                raise ConflictError(
                    f"bundle schema is incompatible with collection "
                    f"'{collection_name}' (schema_checksum mismatch)."
                )
            for field, meta_value in (
                ("id_field", meta.id_field),
                ("vector_field", meta.vector_field),
                ("text_field", meta.text_field),
            ):
                bundle_value = manifest.get(field)
                if bundle_value is not None and bundle_value != meta_value:
                    raise ConflictError(
                        f"bundle {field} '{bundle_value}' does not match "
                        f"collection '{collection_name}' {field} '{meta_value}'."
                    )

            index_path = Path(meta.index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            staging_index = index_path.with_suffix(
                index_path.suffix + self._IMPORT_STAGING_SUFFIX
            )
            pre_import = index_path.with_suffix(
                index_path.suffix + self._PRE_IMPORT_SUFFIX
            )
            new_data_version = meta.data_version + 1
            prev_state = meta.data_state

            # ---- Prepare phase: stage everything, validate, leave live state
            # untouched.  A single try/except guarantees that ANY failure here
            # (index write, corrupt-index deserialization, dim/count mismatch,
            # illegal manifest["total"], staging-table write, commit-intent
            # write, etc.) cleans up both the staging index file and the
            # staging scalar table before propagating — no partial residue.
            committing = False
            try:
                # Stage index into a side file and validate it.
                staging_index.write_bytes(index_bytes)
                loaded = self._read_index(staging_index)
                if manifest.get("dim") is not None and int(loaded.d) != int(manifest["dim"]):
                    raise ConflictError(
                        f"staged index dim {loaded.d} does not match manifest dim "
                        f"{manifest['dim']}."
                    )
                if not (int(loaded.n_total) == int(manifest["total"]) == len(scalar_rows)):
                    raise ConflictError(
                        f"bundle is inconsistent: index.n_total={loaded.n_total}, "
                        f"manifest.total={manifest['total']}, "
                        f"scalar_rows={len(scalar_rows)}."
                    )

                # Stage scalar rows into a side table.
                self.scalar_store.import_rows_to_staging(collection_name, scalar_rows)

                # Durable commit-intent marker.  Written inside the try so that
                # if it fails, the staging artifacts are still cleaned up here.
                self.meta_store.update(
                    collection_name,
                    data_state="importing",
                    import_txn={
                        "stage": "prepared",
                        "new_data_version": new_data_version,
                        "prev_state": prev_state,
                        "total": len(scalar_rows),
                        "dim": manifest.get("dim") or meta.dim,
                        "source_checksum": manifest.get("index_checksum"),
                    },
                )
            except BaseException:
                # Still fully reversible — nothing live was touched.  Sweep the
                # staging index + staging table and restore data_state if the
                # commit-intent marker had already been written.
                staging_index.unlink(missing_ok=True)
                self.scalar_store.rollback_staging(collection_name)
                cur = self.meta_store.get(collection_name)
                if cur is not None and (cur.data_state == "importing" or cur.import_txn):
                    self.meta_store.update(
                        collection_name, data_state=prev_state, import_txn=None
                    )
                raise

            # ---- Commit switch: index -> scalar -> metadata.  Once we begin
            # replacing live state this is no longer locally reversible; a crash
            # mid-switch is resolved deterministically by
            # _recover_interrupted_imports() on the next startup.
            try:
                committing = True
                if index_path.exists():
                    index_path.replace(pre_import)
                staging_index.replace(index_path)
                self.scalar_store.commit_staging(collection_name)
                file_info = index_file_info(index_path)
                updated = self.meta_store.bump_version(
                    collection_name,
                    dim=manifest.get("dim") or meta.dim,
                    total=len(scalar_rows),
                    flushed_at=time.time(),
                    data_state="ready",
                    last_known_total=len(scalar_rows),
                    data_version=new_data_version,
                    index_version=new_data_version,
                    import_txn=None,
                    **file_info,
                )
                pre_import.unlink(missing_ok=True)
            except BaseException:
                # Mid-switch failure — leave the durable markers (data_state=
                # importing + import_txn) in place and let startup recovery roll
                # forward or back.  Do NOT delete staging blindly here: recovery
                # needs it to decide direction.
                assert committing  # documents that we are past the reversible point
                raise
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

        require_exported=True (default): refuse to purge unless the most recent
        export covers the current data snapshot (exported_data_version ==
        data_version).  This prevents data loss when new rows were inserted
        after the last export, or when the bundle download step was skipped
        entirely.

        Security note: SQLite DROP + VACUUM + secure_delete reduces plain-file
        residue but is not a cryptographic erase.  SSD wear-levelling, OS
        file-system journals, and system-level snapshots may retain data at the
        block level.
        """
        collection_name = self.validate_collection_name(collection_name)
        with self._lock_for(collection_name).write_lock():
            meta = self._meta_or_raise(collection_name)
            if require_exported:
                # Purge eligibility must bind BOTH the data snapshot and the
                # index snapshot.  Checking data_version alone is insufficient:
                # upload_index() can swap the live index without touching
                # data_version, so an export taken before that swap would still
                # match on data_version yet no longer reflect the live index.
                data_covered = meta.exported_data_version == meta.data_version
                index_covered = (
                    meta.exported_index_version == meta.index_version
                    and meta.exported_index_checksum == meta.index_checksum
                )
                if not (data_covered and index_covered):
                    raise ConflictError(
                        f"collection '{collection_name}' has no export matching "
                        f"the current data+index snapshot "
                        f"(exported_data_version={meta.exported_data_version} vs "
                        f"data_version={meta.data_version}; "
                        f"exported_index_version={meta.exported_index_version} vs "
                        f"index_version={meta.index_version}; "
                        f"index_checksum match={meta.exported_index_checksum == meta.index_checksum}); "
                        "call export_collection_bundle() first, or pass "
                        "require_exported=False to force purge."
                    )
            last_known_total = self.scalar_store.count(collection_name)

            # Evict from memory
            self._indexes.pop(collection_name, None)

            # Delete index file + all server-generated bundle / temp / staging
            # residue in the collection directory.  Anything the export or
            # import paths could leave behind must be swept here so no user data
            # survives a purge (P0-4).
            collection_dir = self._collection_dir(collection_name)
            index_path = Path(meta.index_path)
            index_path.unlink(missing_ok=True)
            residue_globs = (
                "*.tmp",
                "*.hypervec-bundle",
                "*.hypervec-bundle.tmp",
                "*.import.staging",
                "*.pre-import",
                "*.import.tmp",
                "*.upload.tmp",
            )
            for pattern in residue_globs:
                for leftover in collection_dir.glob(pattern):
                    if leftover.is_file():
                        leftover.unlink(missing_ok=True)
            # Controlled export temp dir (see export_collection_bundle).
            export_tmp = collection_dir / ".export.tmp"
            if export_tmp.exists():
                shutil.rmtree(export_tmp, ignore_errors=True)
            # Drop any orphan import-staging table.
            self.scalar_store.rollback_staging(collection_name)

            # Purge scalar table
            self.scalar_store.purge_collection_rows(collection_name)
            self.scalar_store.checkpoint_and_vacuum()

            purged_at = time.time()
            self.meta_store.update(
                collection_name,
                data_state="purged",
                last_purged_at=purged_at,
                last_known_total=last_known_total,
                data_version=meta.data_version + 1,
                index_version=0,
                # Reset index file info so describe reflects no live index
                index_checksum=None,
                index_size_bytes=None,
                flushed_at=None,
                total=0,
                # Reset export-eligibility snapshot: a fresh export must be taken
                # before the (now empty / re-imported) collection can be purged
                # again.  Leaving stale exported_index_* here could spuriously
                # match the post-purge index_version=0 / checksum=None state.
                exported_data_version=None,
                exported_bundle_checksum=None,
                exported_index_version=None,
                exported_index_checksum=None,
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
