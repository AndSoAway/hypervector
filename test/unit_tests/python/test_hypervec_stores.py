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
