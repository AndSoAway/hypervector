from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module(name: str):
    module_path = Path(__file__).parents[3] / "src" / "python" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_meta_store_persists_versions_and_index_info(tmp_path):
    module = load_module("hypervec_meta_store")
    store = module.MetaStore(tmp_path / "collections.json")
    meta = store.create(
        "demo",
        schema={"fields": []},
        index_params={"indexes": []},
        id_field="id",
        vector_field="vector",
        text_field="contents",
        index_path=str(tmp_path / "demo" / "index.hypervec"),
    )
    assert meta.version == 1

    updated = store.bump_version(
        "demo",
        total=2,
        dim=4,
        index_checksum="sha256:abc",
        index_size_bytes=10,
    )
    assert updated.version == 2

    reloaded = module.MetaStore(tmp_path / "collections.json")
    assert reloaded.get("demo").index_checksum == "sha256:abc"


def test_scalar_store_keeps_vectors_and_metadata_by_row_id(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch(
        "demo",
        [
            (0, "a", [0.0, 1.0], "zero", {"source": "manual"}),
            (1, "b", [2.0, 3.0], "one", {"source": "other"}),
        ],
    )

    assert store.count("demo") == 2
    assert store.next_row_id("demo") == 2
    vectors = store.get_vectors("demo", 2)
    np.testing.assert_array_equal(vectors, np.array([[0, 1], [2, 3]], dtype=np.float32))

    rows = store.get_by_row_ids("demo", [1, 99, 0])
    assert rows[0]["doc_id"] == "b"
    assert rows[0]["metadata"]["source"] == "other"
    assert rows[1] is None
    assert rows[2]["text_content"] == "zero"


def test_scalar_store_load_all_scalars_uses_exact_table_lookup(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")

    assert store.load_all_scalars("demo") == {}

    store.ensure_table("demo")
    store.insert_batch(
        "demo",
        [
            (0, "a", [0.0, 1.0], "zero", {"source": "manual"}),
            (1, "b", [2.0, 3.0], "one", {"source": "other"}),
        ],
    )

    assert store.load_all_scalars("demo") == {
        0: {"doc_id": "a", "text_content": "zero", "metadata": {"source": "manual"}},
        1: {"doc_id": "b", "text_content": "one", "metadata": {"source": "other"}},
    }
    assert store.load_all_scalars("dem") == {}


def test_scalar_store_rejects_duplicate_doc_id(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch("demo", [(0, "a", [0.0, 1.0], "zero", {})])

    try:
        store.insert_batch("demo", [(1, "a", [2.0, 3.0], "replacement", {})])
    except ValueError as exc:
        assert "duplicate row_id or doc_id" in str(exc)
    else:
        raise AssertionError("duplicate doc_id should be rejected")

    rows = store.get_by_row_ids("demo", [0, 1])
    assert rows[0]["doc_id"] == "a"
    assert rows[0]["text_content"] == "zero"
    assert rows[1] is None


def test_scalar_store_export_rows_preserves_all_fields(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch(
        "demo",
        [
            (0, "a", [0.0, 1.0], "zero", {"source": "manual"}),
            (1, "b", [2.0, 3.0], "one", {"source": "other"}),
        ],
    )

    rows = store.export_rows("demo")
    assert len(rows) == 2
    assert rows[0]["row_id"] == 0
    assert rows[0]["doc_id"] == "a"
    assert rows[0]["vector"] == [0.0, 1.0]
    assert rows[0]["text_content"] == "zero"
    assert rows[0]["metadata"] == {"source": "manual"}
    assert rows[0]["created_at"] is not None
    assert rows[1]["row_id"] == 1


def test_scalar_store_import_rows_round_trip(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch(
        "demo",
        [
            (0, "a", [0.0, 1.0], "zero", {"source": "manual"}),
            (1, "b", [2.0, 3.0], "one", {"source": "other"}),
        ],
    )

    exported = store.export_rows("demo")
    count = store.import_rows("demo", exported, replace=True)
    assert count == 2

    re_exported = store.export_rows("demo")
    assert re_exported[0]["doc_id"] == "a"
    assert re_exported[0]["vector"] == [0.0, 1.0]
    assert re_exported[1]["doc_id"] == "b"


def test_scalar_store_purge_collection_rows(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch("demo", [(0, "a", [1.0], "txt", {})])

    result = store.purge_collection_rows("demo")
    assert result["dropped"] is True
    assert result["count_before"] == 1
    assert store.count("demo") == 0


def test_scalar_store_checkpoint_and_vacuum(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch("demo", [(0, "a", [1.0], "txt", {})])
    store.checkpoint_and_vacuum()


def test_scalar_store_export_rows_missing_table_returns_empty(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    # Never created "demo" — the table does not exist, which is the only case
    # that legitimately means "empty export".
    assert store.export_rows("demo") == []


def test_scalar_store_export_rows_propagates_locked_error(tmp_path):
    import sqlite3

    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch("demo", [(0, "a", [1.0], "txt", {})])

    class _Boom:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

    store._conn = lambda: _Boom()
    try:
        store.export_rows("demo")
    except sqlite3.OperationalError as exc:
        assert "locked" in str(exc)
    else:
        raise AssertionError("locked database should propagate, not return []")


def test_scalar_store_export_rows_propagates_database_error(tmp_path):
    import sqlite3

    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")

    class _Corrupt:
        def execute(self, *args, **kwargs):
            raise sqlite3.DatabaseError("database disk image is malformed")

    store._conn = lambda: _Corrupt()
    try:
        store.export_rows("demo")
    except sqlite3.DatabaseError as exc:
        assert "malformed" in str(exc)
    else:
        raise AssertionError("corrupt database should propagate, not return []")


def test_scalar_store_staging_commit_swaps_live_table(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch("demo", [(0, "old", [1.0], "old", {})])

    # Stage new rows — live table must be untouched until commit.
    store.import_rows_to_staging(
        "demo",
        [
            {"row_id": 0, "doc_id": "a", "vector": [0.0, 1.0], "text_content": "hello", "metadata": {}},
            {"row_id": 1, "doc_id": "b", "vector": [2.0, 3.0], "text_content": "world", "metadata": {}},
        ],
    )
    assert store.has_staging("demo") is True
    assert store.count("demo") == 1  # live still has the old row
    assert store.get_by_row_ids("demo", [0])[0]["doc_id"] == "old"

    store.commit_staging("demo")
    assert store.has_staging("demo") is False
    assert store.count("demo") == 2
    assert store.get_by_row_ids("demo", [0])[0]["doc_id"] == "a"


def test_scalar_store_rollback_staging_keeps_live_table(tmp_path):
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("demo")
    store.insert_batch("demo", [(0, "live", [1.0], "live", {})])

    store.import_rows_to_staging(
        "demo",
        [{"row_id": 0, "doc_id": "staged", "vector": [9.0], "text_content": "x", "metadata": {}}],
    )
    store.rollback_staging("demo")

    assert store.has_staging("demo") is False
    assert store.count("demo") == 1
    assert store.get_by_row_ids("demo", [0])[0]["doc_id"] == "live"


def test_meta_store_new_fields_default_on_old_collections_json(tmp_path):
    module = load_module("hypervec_meta_store")
    import json

    old_data = {
        "demo": {
            "collection_name": "demo",
            "version": 1,
            "schema": {"fields": []},
            "index_params": {"indexes": []},
            "id_field": "id",
            "vector_field": "vector",
            "text_field": "contents",
            "dim": None,
            "total": 0,
            "index_path": "/data/demo/index.hypervec",
            "index_checksum": None,
            "index_size_bytes": None,
            "created_at": 1000.0,
            "updated_at": 1000.0,
        }
    }
    path = tmp_path / "collections.json"
    path.write_text(json.dumps(old_data), encoding="utf-8")

    store = module.MetaStore(path)
    meta = store.get("demo")
    assert meta.data_state == "ready"
    assert meta.last_exported_at is None
    assert meta.last_purged_at is None
    assert meta.last_known_total is None
    assert meta.bundle_format is None
    # Dual-version fields default so old metadata loads as "index never built".
    assert meta.data_version == 1
    assert meta.index_version == 0
    assert meta.exported_data_version is None
    assert meta.exported_bundle_checksum is None
    assert meta.exported_index_version is None
    assert meta.exported_index_checksum is None
    assert meta.import_txn is None


def test_scalar_store_export_rows_raises_for_anomalous_view(tmp_path):
    """PR12: a view whose dependency is missing reports 'no such table: dep'
    but the object exists in sqlite_schema — export_rows must raise, not
    silently return an empty list."""
    import sqlite3

    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    conn = store._conn()
    # Create an anomalous view: docs_demo exists in sqlite_schema but its
    # dependency is missing, so a SELECT raises OperationalError.
    conn.execute('CREATE VIEW "docs_demo" AS SELECT * FROM "missing_dep"')
    conn.commit()

    try:
        store.export_rows("demo")
    except sqlite3.OperationalError:
        pass  # correct — object exists but is broken
    else:
        raise AssertionError(
            "export_rows should propagate OperationalError for a broken view, "
            "not silently return []"
        )


def test_scalar_store_export_rows_genuinely_missing_still_returns_empty(tmp_path):
    """PR12: a collection whose table truly does not exist must still return []."""
    module = load_module("hypervec_scalar_store")
    store = module.ScalarStore(tmp_path / "scalar.db")
    # Never created "never_existed"
    assert store.export_rows("never_existed") == []
