from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


class FakeIndexFlatL2:
    def __init__(self, d: int, *, trained: bool = True) -> None:
        self.d = d
        self.is_trained = trained
        self.vectors = np.empty((0, d), dtype=np.float32)

    def train(self, x) -> None:
        self.is_trained = True

    @property
    def n_total(self) -> int:
        return int(self.vectors.shape[0])

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

    def __init__(self, *, write_index_fail: bool = False, delay: float = 0.0) -> None:
        self.saved_index = None
        self.constructor_calls = []
        self.write_index_fail = write_index_fail
        self.delay = delay

    def IndexFlatL2(self, d: int):
        self.constructor_calls.append(("IndexFlatL2", d))
        return FakeIndexFlatL2(d)

    def IndexFlatIP(self, d: int):
        self.constructor_calls.append(("IndexFlatIP", d))
        return FakeIndexFlatL2(d)

    def IndexIVFFlat(self, d: int, nlist: int, metric: int):
        self.constructor_calls.append(("IndexIVFFlat", d, nlist, metric))
        return FakeIndexFlatL2(d, trained=False)

    def IndexIVFLVQ(self, d: int, nlist: int, nlocal: int, nbits: int, metric: int):
        self.constructor_calls.append(("IndexIVFLVQ", d, nlist, nlocal, nbits, metric))
        return FakeIndexFlatL2(d, trained=False)

    def IndexIVFPQ(self, d: int, nlist: int, m_pq: int, nbits: int, metric: int):
        self.constructor_calls.append(("IndexIVFPQ", d, nlist, m_pq, nbits, metric))
        return FakeIndexFlatL2(d, trained=False)

    def IndexHNSWFlat(self, d: int, m_hnsw: int, metric: int):
        self.constructor_calls.append(("IndexHNSWFlat", d, m_hnsw, metric))
        return FakeIndexFlatL2(d)

    def IndexHNSWLVQ(self, d: int, nlocal: int, nbits: int, m_hnsw: int, metric: int):
        self.constructor_calls.append(("IndexHNSWLVQ", d, nlocal, nbits, m_hnsw, metric))
        return FakeIndexFlatL2(d, trained=False)

    def IndexHNSWPQ(self, d: int, m_pq: int, nbits: int, m_hnsw: int, metric: int):
        self.constructor_calls.append(("IndexHNSWPQ", d, m_pq, nbits, m_hnsw, metric))
        return FakeIndexFlatL2(d, trained=False)

    def write_index(self, index, path: str) -> None:
        if self.delay:
            import time as _time
            _time.sleep(self.delay)
        if self.write_index_fail:
            raise RuntimeError("injected write_index failure")
        self.saved_index = index
        Path(path).write_text("fake", encoding="utf-8")

    def read_index(self, path: str):
        if self.delay:
            import time as _time
            _time.sleep(self.delay)
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


def test_hypervec_server_engine_maps_supported_index_types_to_cpp_classes(tmp_path):
    module = load_engine_module()
    fake = FakeHypervec()
    engine = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake)

    cases = [
        (
            "IndexIVFFlat",
            {"nlist": 2},
            ("IndexIVFFlat", 4, 2, fake.kMetricL2),
        ),
        (
            "IndexIVFLVQ",
            {"nlist": 2, "nlocal": 2, "nbits": 1},
            ("IndexIVFLVQ", 4, 2, 2, 1, fake.kMetricL2),
        ),
        (
            "IndexIVFPQ",
            {"nlist": 2, "m_pq": 2, "nbits": 1},
            ("IndexIVFPQ", 4, 2, 2, 1, fake.kMetricL2),
        ),
        (
            "IndexHNSWFlat",
            {"m_hnsw": 8},
            ("IndexHNSWFlat", 4, 8, fake.kMetricL2),
        ),
        (
            "IndexHNSWLVQ",
            {"m_hnsw": 8, "nlocal": 2, "nbits": 1},
            ("IndexHNSWLVQ", 4, 2, 1, 8, fake.kMetricL2),
        ),
        (
            "IndexHNSWPQ",
            {"m_hnsw": 8, "m_pq": 2, "nbits": 1},
            ("IndexHNSWPQ", 4, 2, 1, 8, fake.kMetricL2),
        ),
    ]

    for index_type, params, expected in cases:
        engine._make_index(
            4,
            {
                "field_name": "vector",
                "metric_type": "L2",
                "index_type": index_type,
                "params": params,
            },
        )
        assert fake.constructor_calls[-1] == expected


def test_hypervec_server_engine_rejects_ambiguous_index_m_params(tmp_path):
    module = load_engine_module()
    fake = FakeHypervec()
    engine = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake)

    for bad_param in ["M", "m", "M_hnsw", "M_pq"]:
        try:
            engine._make_index(
                4,
                {
                    "field_name": "vector",
                    "metric_type": "L2",
                    "index_type": "IndexHNSWPQ",
                    "params": {bad_param: 2},
                },
            )
        except ValueError as exc:
            assert "use explicit m_hnsw or m_pq" in str(exc)
        else:
            raise AssertionError(f"expected {bad_param} to be rejected")


def test_hypervec_server_engine_supported_index_examples_follow_exports():
    module = load_engine_module()
    fake = FakeHypervec()
    engine = module.HypervecServerEngine("unused", hypervec_module=fake)

    examples = engine.supported_index_examples()

    assert [item["index_type"] for item in examples] == [
        "IndexIVFFlat",
        "IndexIVFLVQ",
        "IndexIVFPQ",
        "IndexHNSWFlat",
        "IndexHNSWLVQ",
        "IndexHNSWPQ",
    ]
    assert all(item["index_type"].startswith("Index") for item in examples)


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
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "no export matching the current data" in str(exc)
    else:
        raise AssertionError("should have raised ConflictError")


def test_engine_export_bundle_rejects_stale_index(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    # Insert more rows WITHOUT flushing — index is now stale.
    engine.insert("col1", [{"id": "b", "vector": [2.0, 3.0], "contents": "yo"}])

    try:
        engine.export_collection_bundle("col1")
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale-index export should raise ConflictError")

    # After re-flush the export succeeds and records the exported data version.
    engine.flush("col1")
    result = engine.export_collection_bundle("col1")
    assert result["bundle_checksum"].startswith("sha256:")
    meta = engine.meta_store.get("col1")
    assert meta.exported_data_version == meta.data_version
    assert meta.exported_bundle_checksum == result["bundle_checksum"]


def test_engine_flush_marks_index_fresh(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    meta = engine.meta_store.get("col1")
    assert meta.index_version == meta.data_version


def test_engine_purge_blocked_when_data_changed_after_export(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    engine.export_collection_bundle("col1")
    # New data lands after the export — the historical export no longer covers
    # the current snapshot, so purge must be refused.
    engine.insert("col1", [{"id": "b", "vector": [2.0, 3.0], "contents": "yo"}])

    try:
        engine.purge_collection_data("col1", require_exported=True)
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "no export matching the current data" in str(exc)
    else:
        raise AssertionError("purge after post-export insert should raise ConflictError")

    # Re-export to cover the new data, then purge is allowed.
    engine.flush("col1")
    engine.export_collection_bundle("col1")
    result = engine.purge_collection_data("col1", require_exported=True)
    assert result["purged"] is True


def test_engine_purge_blocked_when_index_changed_after_export(tmp_path):
    """PR13-3.2: upload_index after export changes the index snapshot without
    touching data_version, so a historical export must not authorize purge."""
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    engine.export_collection_bundle("col1")

    # Replace the live index via upload_index (data unchanged).  Write a source
    # with different bytes so the uploaded index has a different checksum than
    # the one captured at export time.
    from pathlib import Path
    src = tmp_path / "reupload.hypervec"
    src.write_bytes(b"a-different-index-payload")
    engine.upload_index("col1", src)

    try:
        engine.purge_collection_data("col1", require_exported=True)
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "index" in str(exc).lower()
    else:
        raise AssertionError("purge after post-export upload_index should raise ConflictError")

    # Re-export to cover the new index snapshot, then purge is allowed.
    engine.export_collection_bundle("col1")
    result = engine.purge_collection_data("col1", require_exported=True)
    assert result["purged"] is True


def test_engine_export_rejects_inconsistent_counts(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    # Force a mismatch between scalar count and meta.total without going stale
    engine.scalar_store.insert_batch("col1", [(1, "b", [2.0, 3.0], "yo", {})])

    try:
        engine.export_collection_bundle("col1")
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "inconsistent" in str(exc)
    else:
        raise AssertionError("inconsistent counts should raise ConflictError")


def test_engine_insert_crash_between_scalar_and_version_fails_safe(tmp_path):
    """PR13-3.3: if we crash between the scalar write and the data_version
    bump, export eligibility must already be invalidated so purge is refused
    (fail-safe) rather than allowed (which would lose the just-added rows)."""
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    engine.export_collection_bundle("col1")
    # At this point export covers the data: purge would be allowed.
    meta = engine.meta_store.get("col1")
    assert meta.exported_data_version == meta.data_version

    # Simulate a crash during the second insert: the scalar write raises after
    # data_version has already been bumped.
    orig_insert_batch = engine.scalar_store.insert_batch

    def _boom(*args, **kwargs):
        raise RuntimeError("injected crash during scalar write")

    engine.scalar_store.insert_batch = _boom
    try:
        engine.insert("col1", [{"id": "b", "vector": [2.0, 3.0], "contents": "yo"}])
    except RuntimeError:
        pass
    finally:
        engine.scalar_store.insert_batch = orig_insert_batch

    # data_version was bumped first, so eligibility is invalidated even though
    # the crash prevented the row from being written.
    meta = engine.meta_store.get("col1")
    assert meta.exported_data_version != meta.data_version

    try:
        engine.purge_collection_data("col1", require_exported=True)
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
    else:
        raise AssertionError("purge after crashed insert must be refused (fail-safe)")


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
    # Export to an explicit path outside the collection dir — this mirrors the
    # real flow where the client downloads the bundle off-server before purge,
    # which wipes the collection directory.
    bundle_dst = tmp_path / "downloaded.hypervec-bundle"
    export_result = engine.export_collection_bundle("col1", bundle_dst)
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

    results = engine.search("col1", data=[[0.0, 1.0]], limit=2)
    assert len(results[0]) == 2
    hit_ids = {h["id"] for h in results[0]}
    assert hit_ids == {"a", "b"}
    entities_by_id = {h["id"]: h["entity"] for h in results[0]}
    assert entities_by_id["a"]["contents"] == "hello"
    assert entities_by_id["b"]["contents"] == "world"


def test_engine_import_bundle_rejects_wrong_collection_name(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    export_result = engine.export_collection_bundle("col1")

    engine.create_collection("other", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    try:
        engine.import_collection_bundle("other", export_result["path"])
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "does not match" in str(exc)
    else:
        raise AssertionError("should have raised ConflictError")


def test_engine_import_bundle_rejects_bad_checksum(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    export_result = engine.export_collection_bundle("col1", tmp_path / "b.hypervec-bundle")

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


# ---------------------------------------------------------------------------
# Post-purge behavior tests
# ---------------------------------------------------------------------------


def test_engine_search_after_purge_raises(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    engine.export_collection_bundle("col1")
    engine.purge_collection_data("col1", require_exported=True)

    try:
        engine.search("col1", data=[[0.0, 1.0]], limit=1, output_fields=["id"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("search after purge should raise FileNotFoundError")


def test_engine_index_path_for_download_after_purge_raises(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    engine.export_collection_bundle("col1")
    engine.purge_collection_data("col1", require_exported=True)

    try:
        engine.index_path_for_download("col1")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("index_path_for_download after purge should raise FileNotFoundError")


def test_engine_export_bundle_after_purge_raises_conflict(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    engine.export_collection_bundle("col1")
    engine.purge_collection_data("col1", require_exported=True)

    try:
        engine.export_collection_bundle("col1")
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "purged" in str(exc)
    else:
        raise AssertionError("export_collection_bundle after purge should raise ConflictError")


# ---------------------------------------------------------------------------
# Phase 3: import transaction + crash recovery
# ---------------------------------------------------------------------------


def _make_exported_bundle(tmp_path):
    """Create a collection and export a bundle to a stable off-collection path.

    Returns (engine, fake, bundle_path).  The bundle is written outside the
    collection directory so it survives a subsequent purge (as it would in the
    real flow, where the client downloads it off-server first).
    """
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
    bundle_dst = tmp_path / "downloaded.hypervec-bundle"
    export_result = engine.export_collection_bundle("col1", bundle_dst)
    return engine, fake, export_result["path"]


def test_engine_import_index_write_failure_leaves_live_state_intact(tmp_path):
    from pathlib import Path

    engine, fake, bundle_path = _make_exported_bundle(tmp_path)
    # Keep a live copy of data (don't purge) so we can prove it survives.
    assert engine.scalar_store.count("col1") == 2
    meta_before = engine.meta_store.get("col1")

    # Arm read_index to fail while validating the staged index.
    def _boom_read(path):
        raise RuntimeError("injected read_index failure")

    original = engine._read_index
    engine._read_index = _boom_read
    try:
        try:
            engine.import_collection_bundle("col1", bundle_path)
        except RuntimeError as exc:
            assert "injected" in str(exc)
        else:
            raise AssertionError("import should have raised on staged-index failure")
    finally:
        engine._read_index = original

    # Live scalar rows untouched; no residue; state not left "importing".
    assert engine.scalar_store.count("col1") == 2
    assert not engine.scalar_store.has_staging("col1")
    coll_dir = engine._collection_dir("col1")
    assert not list(coll_dir.glob("*.import.staging"))
    meta_after = engine.meta_store.get("col1")
    assert meta_after.data_state == meta_before.data_state


def test_engine_import_staging_table_failure_cleans_up(tmp_path):
    """PR13-3.4: a failure in the scalar-staging step must sweep the staging
    index file too (unified prepare-phase cleanup), leaving no residue."""
    engine, fake, bundle_path = _make_exported_bundle(tmp_path)
    meta_before = engine.meta_store.get("col1")

    orig = engine.scalar_store.import_rows_to_staging

    def _boom(name, rows):
        raise RuntimeError("injected staging-table failure")

    engine.scalar_store.import_rows_to_staging = _boom
    try:
        try:
            engine.import_collection_bundle("col1", bundle_path)
        except RuntimeError as exc:
            assert "injected" in str(exc)
        else:
            raise AssertionError("import should have raised on staging-table failure")
    finally:
        engine.scalar_store.import_rows_to_staging = orig

    coll_dir = engine._collection_dir("col1")
    assert not list(coll_dir.glob("*.import.staging"))
    assert not engine.scalar_store.has_staging("col1")
    meta_after = engine.meta_store.get("col1")
    assert meta_after.data_state == meta_before.data_state
    assert meta_after.import_txn is None


def test_engine_import_commit_intent_failure_cleans_up(tmp_path):
    """PR13-3.4: if the durable commit-intent write fails, the prepare-phase
    cleanup must remove the staging index + table and restore data_state."""
    engine, fake, bundle_path = _make_exported_bundle(tmp_path)
    meta_before = engine.meta_store.get("col1")

    orig_update = engine.meta_store.update

    def _boom_update(name, **changes):
        if changes.get("data_state") == "importing":
            raise RuntimeError("injected commit-intent write failure")
        return orig_update(name, **changes)

    engine.meta_store.update = _boom_update
    try:
        try:
            engine.import_collection_bundle("col1", bundle_path)
        except RuntimeError as exc:
            assert "injected" in str(exc)
        else:
            raise AssertionError("import should have raised on commit-intent failure")
    finally:
        engine.meta_store.update = orig_update

    coll_dir = engine._collection_dir("col1")
    assert not list(coll_dir.glob("*.import.staging"))
    assert not engine.scalar_store.has_staging("col1")
    meta_after = engine.meta_store.get("col1")
    assert meta_after.data_state == meta_before.data_state
    assert meta_after.import_txn is None


def test_engine_import_commit_staging_failure_recoverable(tmp_path):
    engine, fake, bundle_path = _make_exported_bundle(tmp_path)
    engine.purge_collection_data("col1", require_exported=True)

    # Fail during the scalar commit switch.
    def _boom_commit(name):
        raise RuntimeError("injected commit_staging failure")

    original = engine.scalar_store.commit_staging
    engine.scalar_store.commit_staging = _boom_commit
    try:
        try:
            engine.import_collection_bundle("col1", bundle_path)
        except RuntimeError as exc:
            assert "injected" in str(exc)
        else:
            raise AssertionError("import should have raised on commit failure")
    finally:
        engine.scalar_store.commit_staging = original

    # Metadata is left mid-import; a fresh engine on the same data_root must
    # recover it (roll back, since the scalar switch never completed).
    module = load_engine_module()
    fake2 = FakeHypervec()
    fake2.saved_index = fake.saved_index
    engine2 = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake2)
    meta = engine2.meta_store.get("col1")
    assert meta.data_state != "importing"
    assert meta.import_txn is None
    assert not engine2.scalar_store.has_staging("col1")


def test_engine_recover_rolls_forward_completed_switch(tmp_path):
    """If the switch completed but metadata finalize was lost, roll forward."""
    engine, fake, bundle_path = _make_exported_bundle(tmp_path)
    engine.purge_collection_data("col1", require_exported=True)
    engine.import_collection_bundle("col1", bundle_path)

    # Simulate a crash right before the metadata finalize by re-marking the
    # collection "importing" while the live index + scalar rows are in place.
    meta = engine.meta_store.get("col1")
    engine.meta_store.update(
        "col1",
        data_state="importing",
        import_txn={
            "stage": "prepared",
            "new_data_version": meta.data_version + 1,
            "prev_state": "purged",
            "total": 2,
            "dim": 2,
        },
    )

    module = load_engine_module()
    fake2 = FakeHypervec()
    fake2.saved_index = fake.saved_index
    engine2 = module.HypervecServerEngine(str(tmp_path), hypervec_module=fake2)
    recovered = engine2.meta_store.get("col1")
    assert recovered.data_state == "ready"
    assert recovered.import_txn is None
    assert recovered.index_version == recovered.data_version
    assert engine2.scalar_store.count("col1") == 2


# ---------------------------------------------------------------------------
# Phase 4: purge residue cleanup + controlled export temp dir
# ---------------------------------------------------------------------------


def test_engine_purge_leaves_no_bundle_residue(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")
    # Default (HTTP-style) export builds into the controlled .export.tmp dir.
    engine.export_collection_bundle("col1")
    # Simulate stray server-side artifacts a crash/abort could leave behind.
    coll_dir = engine._collection_dir("col1")
    (coll_dir / "col1.hypervec-bundle").write_bytes(b"leftover")
    (coll_dir / "index.hypervec.import.staging").write_bytes(b"leftover")
    (coll_dir / "index.hypervec.pre-import").write_bytes(b"leftover")

    engine.purge_collection_data("col1", require_exported=True)

    residue = []
    for pattern in (
        "*.hypervec-bundle",
        "*.hypervec-bundle.tmp",
        "*.import.staging",
        "*.pre-import",
        "*.tmp",
    ):
        residue += list(coll_dir.glob(pattern))
    assert residue == [], f"purge left residue: {residue}"
    assert not (coll_dir / ".export.tmp").exists()
    assert not (coll_dir / "index.hypervec").exists()


def test_engine_export_failure_leaves_no_residue(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.create_collection("col1", schema=_SCHEMA, index_params=_INDEX_PARAMS)
    engine.insert("col1", [{"id": "a", "vector": [0.0, 1.0], "contents": "hi"}])
    engine.flush("col1")

    # Force create_bundle to raise mid-export by making the index file vanish
    # right after the freshness/consistency checks would have read it... instead
    # inject a failure via a monkeypatched scalar export.
    def _boom_export(name):
        raise RuntimeError("injected export_rows failure")

    original = engine.scalar_store.export_rows
    engine.scalar_store.export_rows = _boom_export
    try:
        try:
            engine.export_collection_bundle("col1")
        except RuntimeError as exc:
            assert "injected" in str(exc)
        else:
            raise AssertionError("export should have raised")
    finally:
        engine.scalar_store.export_rows = original

    coll_dir = engine._collection_dir("col1")
    # No controlled temp dir or stray bundle survives a failed export.
    assert not list(coll_dir.glob("*.hypervec-bundle"))
    # export_rows failed before the temp dir was created, but if it exists it
    # must be empty of bundles.
    export_tmp = coll_dir / ".export.tmp"
    if export_tmp.exists():
        assert not list(export_tmp.glob("*.hypervec-bundle"))


# ---------------------------------------------------------------------------
# Phase 5: import validation (mode / schema)
# ---------------------------------------------------------------------------


def test_engine_import_rejects_non_replace_mode(tmp_path):
    engine, _, bundle_path = _make_exported_bundle(tmp_path)
    try:
        engine.import_collection_bundle("col1", bundle_path, mode="append")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("non-replace mode should raise ValueError")


def test_engine_import_rejects_incompatible_schema(tmp_path):
    engine, _, bundle_path = _make_exported_bundle(tmp_path)
    engine.purge_collection_data("col1", require_exported=True)
    # Mutate the target schema so its checksum no longer matches the bundle.
    engine.meta_store.update("col1", schema={"fields": [{"name": "different"}]})

    try:
        engine.import_collection_bundle("col1", bundle_path)
    except Exception as exc:
        assert type(exc).__name__ == "ConflictError"
        assert "schema" in str(exc)
    else:
        raise AssertionError("incompatible schema should raise ConflictError")
