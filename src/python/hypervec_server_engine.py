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


class HypervecServerEngine:
    INDEX_FILE = "index.hypervec"

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
        del consistency_level
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
            distances, labels = self._search_index(index, query, candidate_k, search_params)
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
