from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


def load_module(name: str):
    module_path = Path(__file__).parents[3] / "src" / "python" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_fake_meta(module):
    """Create a minimal CollectionMeta for bundle tests."""
    return module.CollectionMeta(
        collection_name="testcol",
        version=2,
        schema={"fields": []},
        index_params={"indexes": []},
        id_field="id",
        vector_field="vector",
        text_field="contents",
        dim=2,
        total=2,
        index_path="/data/testcol/index.hypervec",
        index_checksum=None,
        index_size_bytes=None,
        created_at=1000.0,
        updated_at=1000.0,
    )


def test_bundle_create_and_read_round_trip(tmp_path):
    bundle_mod = load_module("hypervec_bundle")
    meta_mod = load_module("hypervec_meta_store")

    index_path = tmp_path / "index.hypervec"
    index_path.write_bytes(b"\x00fake-index-bytes\xff")

    scalar_rows = [
        {
            "row_id": 0,
            "doc_id": "a",
            "vector": [0.1, 0.2],
            "text_content": "hello",
            "metadata": {"src": "test"},
            "created_at": 1000.0,
            "updated_at": 1000.0,
        },
        {
            "row_id": 1,
            "doc_id": "b",
            "vector": [0.3, 0.4],
            "text_content": "world",
            "metadata": {},
            "created_at": 1001.0,
            "updated_at": 1001.0,
        },
    ]

    meta = make_fake_meta(meta_mod)
    output_path = tmp_path / "testcol.hypervec-bundle"

    manifest = bundle_mod.create_bundle(
        "testcol", index_path, scalar_rows, meta, output_path
    )

    assert output_path.exists()
    assert manifest["format"] == bundle_mod.BUNDLE_FORMAT
    assert manifest["collection_name"] == "testcol"
    assert manifest["total"] == 2
    assert manifest["dim"] == 2
    assert manifest["index_checksum"].startswith("sha256:")
    assert manifest["scalar_checksum"].startswith("sha256:")

    # Verify ZIP structure
    with zipfile.ZipFile(output_path) as zf:
        names = zf.namelist()
    assert "manifest.json" in names
    assert "index.hypervec" in names
    assert "scalar.jsonl" in names

    # Round-trip read
    m2, idx_bytes, rows2 = bundle_mod.read_bundle(output_path)
    assert m2["format"] == bundle_mod.BUNDLE_FORMAT
    assert idx_bytes == b"\x00fake-index-bytes\xff"
    assert len(rows2) == 2
    assert rows2[0]["doc_id"] == "a"
    assert rows2[0]["vector"] == [0.1, 0.2]
    assert rows2[1]["doc_id"] == "b"


def test_bundle_read_detects_missing_member(tmp_path):
    bundle_mod = load_module("hypervec_bundle")

    bad_bundle = tmp_path / "bad.hypervec-bundle"
    with zipfile.ZipFile(bad_bundle, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": bundle_mod.BUNDLE_FORMAT}))
        # Missing index.hypervec and scalar.jsonl

    try:
        bundle_mod.read_bundle(bad_bundle)
    except ValueError as exc:
        assert "missing required file" in str(exc)
    else:
        raise AssertionError("should have raised ValueError")


def test_bundle_read_detects_wrong_format(tmp_path):
    bundle_mod = load_module("hypervec_bundle")

    bad_bundle = tmp_path / "bad.hypervec-bundle"
    with zipfile.ZipFile(bad_bundle, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "some.other.format"}))
        zf.writestr("index.hypervec", b"x")
        zf.writestr("scalar.jsonl", b"")

    try:
        bundle_mod.read_bundle(bad_bundle)
    except ValueError as exc:
        assert "unsupported bundle format" in str(exc)
    else:
        raise AssertionError("should have raised ValueError")


def test_bundle_read_detects_index_checksum_mismatch(tmp_path):
    bundle_mod = load_module("hypervec_bundle")
    meta_mod = load_module("hypervec_meta_store")

    index_path = tmp_path / "index.hypervec"
    index_path.write_bytes(b"real-index")
    meta = make_fake_meta(meta_mod)
    output_path = tmp_path / "t.hypervec-bundle"
    bundle_mod.create_bundle("testcol", index_path, [], meta, output_path)

    # Tamper with the index inside the zip
    tampered = tmp_path / "tampered.hypervec-bundle"
    with zipfile.ZipFile(output_path) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            if item.filename == "index.hypervec":
                zout.writestr(item, b"tampered-index")
            else:
                zout.writestr(item, zin.read(item.filename))

    try:
        bundle_mod.read_bundle(tampered)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("should have raised ValueError for checksum mismatch")
