"""
Persistent metadata store for collections.
Backed by a single JSON file; safe for single-process use.

Key convention: collection_name is the primary key everywhere.
collection_id is kept as an alias equal to collection_name for HTTP route compatibility.
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

_STORE_PATH = Path(__file__).parent / "collections.json"


class CollectionMeta:
    __slots__ = ("collection_id", "name", "version", "index_path", "updated_at")

    def __init__(self, collection_id: str, name: str, version: int,
                 index_path: str, updated_at: float):
        self.collection_id = collection_id
        self.name = name
        self.version = version
        self.index_path = index_path
        self.updated_at = updated_at

    def to_dict(self) -> dict:
        return {
            "collection_id": self.collection_id,
            "name": self.name,
            "version": self.version,
            "index_path": self.index_path,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CollectionMeta":
        return cls(
            collection_id=d["collection_id"],
            name=d["name"],
            version=d["version"],
            index_path=d["index_path"],
            updated_at=d["updated_at"],
        )


class MetaStore:
    def __init__(self, path: Path = _STORE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, CollectionMeta] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = {k: CollectionMeta.from_dict(v) for k, v in raw.items()}

    def _flush(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({k: v.to_dict() for k, v in self._data.items()}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    # ------------------------------------------------------------------

    def get(self, collection_id: str) -> Optional[CollectionMeta]:
        with self._lock:
            return self._data.get(collection_id)

    def create(self, name: str, index_path: str) -> CollectionMeta:
        with self._lock:
            meta = CollectionMeta(
                collection_id=name,
                name=name,
                version=1,
                index_path=index_path,
                updated_at=time.time(),
            )
            self._data[name] = meta
            self._flush()
            return meta

    def bump_version(self, collection_id: str) -> CollectionMeta:
        """Increment version after index is updated. Raises KeyError if not found."""
        with self._lock:
            meta = self._data[collection_id]
            meta.version += 1
            meta.updated_at = time.time()
            self._flush()
            # Return a snapshot copy so callers don't share the mutable reference
            return CollectionMeta(
                collection_id=meta.collection_id,
                name=meta.name,
                version=meta.version,
                index_path=meta.index_path,
                updated_at=meta.updated_at,
            )

    def delete(self, collection_id: str) -> bool:
        with self._lock:
            if collection_id not in self._data:
                return False
            del self._data[collection_id]
            self._flush()
            return True

    def list_all(self) -> list[CollectionMeta]:
        with self._lock:
            return list(self._data.values())
