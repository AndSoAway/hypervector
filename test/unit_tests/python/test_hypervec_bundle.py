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





def test_bundle_read_rejects_missing_required_checksum(tmp_path):
    bundle_mod = load_module("hypervec_bundle")

    bad = tmp_path / "bad.hypervec-bundle"
    with zipfile.ZipFile(bad, "w") as zf:
        # Valid format + members but missing scalar_checksum/schema_checksum.
        zf.writestr(
            "manifest.json",
            json.dumps({"format": bundle_mod.BUNDLE_FORMAT, "index_checksum": "sha256:x"}),
        )
        zf.writestr("index.hypervec", b"x")
        zf.writestr("scalar.jsonl", b"")

    try:
        bundle_mod.read_bundle(bad)
    except ValueError as exc:
        assert "missing required v1 field" in str(exc)
    else:
        raise AssertionError("should reject manifest missing required checksums")


def test_bundle_read_rejects_extra_zip_entries(tmp_path):
    bundle_mod = load_module("hypervec_bundle")
    meta_mod = load_module("hypervec_meta_store")

    index_path = tmp_path / "index.hypervec"
    index_path.write_bytes(b"real-index")
    meta = make_fake_meta(meta_mod)
    good = tmp_path / "good.hypervec-bundle"
    bundle_mod.create_bundle("testcol", index_path, [], meta, good)

    # Repack with an extra stray entry.
    bomb = tmp_path / "bomb.hypervec-bundle"
    with zipfile.ZipFile(good) as zin, zipfile.ZipFile(bomb, "w") as zout:
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        zout.writestr("evil.bin", b"0" * 1024)

    try:
        bundle_mod.read_bundle(bomb)
    except ValueError as exc:
        assert "unexpected entries" in str(exc)
    else:
        raise AssertionError("should reject bundle with extra zip entries")


def test_bundle_read_rejects_noncontiguous_row_ids(tmp_path):
    bundle_mod = load_module("hypervec_bundle")
    meta_mod = load_module("hypervec_meta_store")

    index_path = tmp_path / "index.hypervec"
    index_path.write_bytes(b"idx")
    meta = make_fake_meta(meta_mod)
    rows = [
        {"row_id": 0, "doc_id": "a", "vector": [0.1, 0.2], "text_content": "", "metadata": {}},
        {"row_id": 5, "doc_id": "b", "vector": [0.3, 0.4], "text_content": "", "metadata": {}},
    ]
    out = tmp_path / "gap.hypervec-bundle"
    bundle_mod.create_bundle("testcol", index_path, rows, meta, out)

    try:
        bundle_mod.read_bundle(out)
    except ValueError as exc:
        assert "cover 0" in str(exc) or "contiguous" in str(exc) or "total" in str(exc)
    else:
        raise AssertionError("should reject non-contiguous row_ids")


# ---------------------------------------------------------------------------
# validate_index_label_mapping (PR13-3.1)
# ---------------------------------------------------------------------------


def _rows(*row_ids, dim=2):
    return [
        {"row_id": rid, "doc_id": f"d{i}", "vector": [0.1] * dim,
         "text_content": "", "metadata": {}}
        for i, rid in enumerate(row_ids)
    ]


def test_validate_label_mapping_accepts_contiguous():
    m = load_module("hypervec_bundle")
    # 0..2 in any order is valid
    m.validate_index_label_mapping(_rows(2, 0, 1), total=3, dim=2)


def test_validate_label_mapping_rejects_gap():
    m = load_module("hypervec_bundle")
    try:
        m.validate_index_label_mapping(_rows(0, 2), total=2, dim=2)
    except ValueError as exc:
        assert "cover 0" in str(exc)
    else:
        raise AssertionError("gap in row_ids should be rejected")


def test_validate_label_mapping_rejects_duplicate():
    m = load_module("hypervec_bundle")
    try:
        m.validate_index_label_mapping(_rows(0, 0), total=2, dim=2)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate row_id should be rejected")


def test_validate_label_mapping_rejects_bool_row_id():
    m = load_module("hypervec_bundle")
    # True would pass isinstance(x, int); must be rejected explicitly.
    rows = [{"row_id": True, "doc_id": "a", "vector": [0.1, 0.2], "metadata": {}}]
    try:
        m.validate_index_label_mapping(rows, total=1, dim=2)
    except ValueError as exc:
        assert "non-integer" in str(exc)
    else:
        raise AssertionError("bool row_id should be rejected")


def test_validate_label_mapping_rejects_non_integer_row_id():
    m = load_module("hypervec_bundle")
    rows = [{"row_id": "0", "doc_id": "a", "vector": [0.1, 0.2], "metadata": {}}]
    try:
        m.validate_index_label_mapping(rows, total=1, dim=2)
    except ValueError as exc:
        assert "non-integer" in str(exc)
    else:
        raise AssertionError("string row_id should be rejected")


def test_validate_label_mapping_rejects_wrong_vector_dim():
    m = load_module("hypervec_bundle")
    rows = [{"row_id": 0, "doc_id": "a", "vector": [0.1, 0.2, 0.3], "metadata": {}}]
    try:
        m.validate_index_label_mapping(rows, total=1, dim=2)
    except ValueError as exc:
        assert "vector dim" in str(exc)
    else:
        raise AssertionError("wrong vector dim should be rejected")


def test_validate_label_mapping_rejects_2d_vector():
    m = load_module("hypervec_bundle")
    rows = [{"row_id": 0, "doc_id": "a", "vector": [[0.1, 0.2]], "metadata": {}}]
    try:
        m.validate_index_label_mapping(rows, total=1, dim=None)
    except ValueError as exc:
        assert "1-D" in str(exc)
    else:
        raise AssertionError("2-D vector should be rejected")


def test_validate_label_mapping_rejects_unknown_strategy():
    m = load_module("hypervec_bundle")
    try:
        m.validate_index_label_mapping(_rows(0), total=1, dim=2,
                                       label_strategy="explicit_map")
    except ValueError as exc:
        assert "label_strategy" in str(exc)
    else:
        raise AssertionError("unknown strategy should be rejected")


def test_bundle_manifest_records_label_strategy(tmp_path):
    bundle_mod = load_module("hypervec_bundle")
    meta_mod = load_module("hypervec_meta_store")
    index_path = tmp_path / "index.hypervec"
    index_path.write_bytes(b"idx")
    meta = make_fake_meta(meta_mod)
    rows = _rows(0, 1)
    out = tmp_path / "b.hypervec-bundle"
    manifest = bundle_mod.create_bundle("testcol", index_path, rows, meta, out)
    assert manifest["label_strategy"] == bundle_mod.LABEL_STRATEGY_IMPLICIT_SEQUENTIAL
    # round-trips through read_bundle
    m2, _, rows2 = bundle_mod.read_bundle(out)
    assert m2["label_strategy"] == bundle_mod.LABEL_STRATEGY_IMPLICIT_SEQUENTIAL
    assert len(rows2) == 2
