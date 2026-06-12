# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the License) found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import threading
import time
from pathlib import Path
from typing import Any


@dataclass
class CollectionMeta:
    collection_name: str
    version: int
    schema: dict[str, Any]
    index_params: dict[str, Any]
    id_field: str
    vector_field: str
    text_field: str
    dim: int | None
    total: int
    index_path: str
    index_checksum: str | None
    index_size_bytes: int | None
    created_at: float
    updated_at: float
    flushed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollectionMeta":
        return cls(
            collection_name=str(data["collection_name"]),
            version=int(data.get("version", 1)),
            schema=dict(data.get("schema") or {}),
            index_params=dict(data.get("index_params") or {"indexes": []}),
            id_field=str(data.get("id_field") or "id"),
            vector_field=str(data.get("vector_field") or "vector"),
            text_field=str(data.get("text_field") or "contents"),
            dim=data.get("dim"),
            total=int(data.get("total", 0)),
            index_path=str(data.get("index_path") or ""),
            index_checksum=data.get("index_checksum"),
            index_size_bytes=data.get("index_size_bytes"),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            flushed_at=data.get("flushed_at"),
        )


class MetaStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict[str, CollectionMeta] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._data = {
            name: CollectionMeta.from_dict(meta) for name, meta in raw.items()
        }

    def _flush(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {name: meta.to_dict() for name, meta in self._data.items()},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def list_all(self) -> list[CollectionMeta]:
        with self._lock:
            return list(self._data.values())

    def get(self, collection_name: str) -> CollectionMeta | None:
        with self._lock:
            return self._data.get(collection_name)

    def create(
        self,
        collection_name: str,
        *,
        schema: dict[str, Any],
        index_params: dict[str, Any],
        id_field: str,
        vector_field: str,
        text_field: str,
        index_path: str,
    ) -> CollectionMeta:
        with self._lock:
            if collection_name in self._data:
                raise FileExistsError(f"collection '{collection_name}' already exists.")
            now = time.time()
            meta = CollectionMeta(
                collection_name=collection_name,
                version=1,
                schema=dict(schema),
                index_params=dict(index_params),
                id_field=id_field,
                vector_field=vector_field,
                text_field=text_field,
                dim=None,
                total=0,
                index_path=index_path,
                index_checksum=None,
                index_size_bytes=None,
                created_at=now,
                updated_at=now,
            )
            self._data[collection_name] = meta
            self._flush()
            return CollectionMeta.from_dict(meta.to_dict())

    def update(self, collection_name: str, **changes: Any) -> CollectionMeta:
        with self._lock:
            meta = self._data[collection_name]
            for key, value in changes.items():
                setattr(meta, key, value)
            meta.updated_at = time.time()
            self._flush()
            return CollectionMeta.from_dict(meta.to_dict())

    def bump_version(self, collection_name: str, **changes: Any) -> CollectionMeta:
        with self._lock:
            meta = self._data[collection_name]
            meta.version += 1
            for key, value in changes.items():
                setattr(meta, key, value)
            meta.updated_at = time.time()
            self._flush()
            return CollectionMeta.from_dict(meta.to_dict())

    def set_version(self, collection_name: str, version: int, **changes: Any) -> CollectionMeta:
        with self._lock:
            meta = self._data[collection_name]
            meta.version = int(version)
            for key, value in changes.items():
                setattr(meta, key, value)
            meta.updated_at = time.time()
            self._flush()
            return CollectionMeta.from_dict(meta.to_dict())

    def delete(self, collection_name: str) -> bool:
        with self._lock:
            if collection_name not in self._data:
                return False
            del self._data[collection_name]
            self._flush()
            return True
