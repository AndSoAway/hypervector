from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


class FakeIndexFlatL2:
    def __init__(self, d: int) -> None:
        self.d = d
        self.is_trained = True
        self.vectors = np.empty((0, d), dtype=np.float32)

    def add(self, x) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(x, dtype=np.float32)])

    def search(self, x, k: int):
        x = np.asarray(x, dtype=np.float32)
        distances = ((x[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        labels = np.argsort(distances, axis=1)[:, :k].astype(np.int64)
        dists = np.take_along_axis(distances, labels, axis=1).astype(np.float32)
        return dists, labels


class FakeHypervec:
    kMetricL2 = 1
    kMetricInnerProduct = 0

    def __init__(self) -> None:
        self.saved_index = None

    def IndexFlatL2(self, d: int):
        return FakeIndexFlatL2(d)

    def IndexFlatIP(self, d: int):
        return FakeIndexFlatL2(d)

    def IndexHNSWFlat(self, d: int, m: int, metric: int):
        return FakeIndexFlatL2(d)

    def write_index(self, index, path: str) -> None:
        self.saved_index = index
        Path(path).write_text("fake", encoding="utf-8")

    def read_index(self, path: str):
        return self.saved_index


def load_engine_module():
    module_path = Path(__file__).parents[3] / "src" / "python" / "hypervec_server_engine.py"
    spec = importlib.util.spec_from_file_location("hypervec_server_engine_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hypervec_server_engine_create_insert_flush_load_search(tmp_path):
    module = load_engine_module()
    fake = FakeHypervec()
    engine = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake)

    schema = {
        "auto_id": False,
        "enable_dynamic_field": True,
        "fields": [
            {"name": "id", "datatype": "VARCHAR", "is_primary": True},
            {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 2},
            {"name": "contents", "datatype": "VARCHAR"},
        ],
    }
    index_params = {
        "indexes": [
            {
                "field_name": "vector",
                "metric_type": "L2",
                "index_type": "Flat",
                "params": {},
            }
        ]
    }

    created = engine.create_collection("demo", schema=schema, index_params=index_params)
    assert created["collection_name"] == "demo"
    assert engine.has_collection("demo")
    engine.create_collection("alpha", schema=schema, index_params=index_params)
    described = engine.describe_collections()
    assert [desc["collection_name"] for desc in described] == ["alpha", "demo"]
    assert described[0]["schema"] == schema
    engine.drop_collection("alpha")

    inserted = engine.insert(
        "demo",
        [
            {"id": "a", "vector": [0, 0], "contents": "zero", "source": "manual"},
            {"id": "b", "vector": [1, 1], "contents": "one", "source": "manual"},
            {"id": "c", "vector": [10, 10], "contents": "ten", "source": "other"},
        ],
    )
    assert inserted["total"] == 3
    stored = engine.scalar_store.get_by_row_ids("demo", [0])[0]
    assert stored["metadata"] == {"source": "manual"}
    assert engine.get_version("demo")["version"] == 1
    flushed = engine.flush("demo")
    assert flushed["dim"] == 2
    assert flushed["version"] == 2
    assert engine.sync_check("demo", client_version=1)["needs_sync"]
    assert not engine.sync_check("demo", client_version=2)["needs_sync"]

    engine.close_collection("demo")
    loaded = engine.load_collection("demo")
    assert loaded["loaded"]
    assert loaded["version"] == 2

    results = engine.search(
        "demo",
        data=[[0.1, 0.1]],
        limit=2,
        output_fields=["id", "contents", "source"],
        filter="source == 'manual'",
    )
    assert results == [
        [
            {
                "id": "a",
                "distance": results[0][0]["distance"],
                "entity": {"id": "a", "contents": "zero", "source": "manual"},
            },
            {
                "id": "b",
                "distance": results[0][1]["distance"],
                "entity": {"id": "b", "contents": "one", "source": "manual"},
            },
        ]
    ]

    dropped = engine.drop_collection("demo")
    assert dropped["dropped"]
    assert not engine.has_collection("demo")


# ---------------------------------------------------------------------------
# Bundle / purge tests
# ---------------------------------------------------------------------------

_SCHEMA = {
    "auto_id": False,
    "enable_dynamic_field": True,
    "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": True},
        {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 2},
        {"name": "contents", "datatype": "VARCHAR"},
    ],
}
_INDEX_PARAMS = {
    "indexes": [
        {"field_name": "vector", "metric_type": "L2", "index_type": "Flat", "params": {}}
    ]
}


def make_engine(tmp_path):
    module = load_engine_module()
    fake = FakeHypervec()
    return module.HypervecServerEngine(str(tmp_path), hypervec_module=fake), fake


def test_engine_export_bundle_creates_zip(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert(
        "col1",
        [
            {"id": "a", "vector": [0.0, 1.0], "contents": "hello"},
            {"id": "b", "vector": [2.0, 3.0], "contents": "world"},
        ],
    )
    engine.flush("col1")

    result = engine.export_collection_bundle("col1")
    assert result["bundle_format"].startswith("hypervector.collection.bundle")
    assert result["bytes"] > 0

    import zipfile

    with zipfile.ZipFile(result["path"]) as zf:
        names = zf.namelist()
    assert "manifest.json" in names
    assert "index.hypervec" in names
    assert "scalar.jsonl" in names

    meta = engine.meta_store.get("col1")
    assert meta.last_exported_at is not None
    assert meta.bundle_format is not None


def test_engine_purge_removes_data_keeps_metadata(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hello"}])
    engine.flush("col1")
    engine.export_collection_bundle("col1")

    result = engine.purge_collection_data("col1")
    assert result["purged"] is True
    assert result["metadata_preserved"] is True
    assert result["data_state"] == "purged"

    # Metadata still exists
    assert engine.has_collection("col1")
    meta = engine.meta_store.get("col1")
    assert meta.data_state == "purged"
    assert meta.last_purged_at is not None

    # Index file gone
    from pathlib import Path
    assert not Path(meta.index_path).exists()

    # Scalar count = 0
    assert engine.scalar_store.count("col1") == 0


def test_engine_purge_requires_export_by_default(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")

    try:
        engine.purge_collection_data("col1", require_exported=True)
    except ValueError as exc:
        assert "no recorded export" in str(exc)
    else:
        raise AssertionError("should have raised ValueError")


def test_engine_import_bundle_restores_data(tmp_path):
    engine, fake = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert(
        "col1",
        [
            {"id": "a", "vector": [0.0, 1.0], "contents": "hello"},
            {"id": "b", "vector": [2.0, 3.0], "contents": "world"},
        ],
    )
    engine.flush("col1")
    export_result = engine.export_collection_bundle("col1")
    engine.purge_collection_data("col1", require_exported=True)

    # Verify purged state
    assert engine.scalar_store.count("col1") == 0

    # Restore
    restore_result = engine.import_collection_bundle("col1", export_result["path"])
    assert restore_result["uploaded"] is True
    assert restore_result["total"] == 2
    assert restore_result["data_state"] == "ready"

    meta = engine.meta_store.get("col1")
    assert meta.data_state == "ready"
    assert engine.scalar_store.count("col1") == 2


def test_engine_import_bundle_rejects_wrong_collection_name(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    export_result = engine.export_collection_bundle("col1")

    engine.create_collection("other", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    try:
        engine.import_collection_bundle("other", export_result["path"])
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("should have raised ValueError")


def test_engine_import_bundle_rejects_bad_checksum(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    export_result = engine.export_collection_bundle("col1")

    engine.purge_collection_data("col1")
    try:
        engine.import_collection_bundle(
            "col1",
            export_result["path"],
            checksum="sha256:deadbeef",
        )
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("should have raised ValueError")


def test_engine_describe_includes_data_state(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)

    described = engine.describe_collection("col1")
    assert described["data_state"] == "ready"
    assert "last_exported_at" in described
    assert "last_purged_at" in described

    version = engine.get_version("col1")
    assert "data_state" in version
