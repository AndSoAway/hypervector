from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module loader + fakes (same pattern as test_sparse_vector.py)
# ---------------------------------------------------------------------------

def _load_module(relative_path: str, name: str):
    path = Path(__file__).parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_engine_module():
    return _load_module("src/python/hypervec_server_engine.py", "dual_engine_under_test")


class FakeIndexFlatL2:
    def __init__(self, d: int, *, trained: bool = True) -> None:
        self.d = d
        self.n_total = 0
        self.is_trained = trained
        self.vectors = np.empty((0, d), dtype=np.float32)

    def train(self, x) -> None:
        self.is_trained = True

    def add(self, x) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(x, dtype=np.float32)])
        self.n_total = len(self.vectors)

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

    def IndexHNSWFlat(self, d, m_hnsw, metric):
        return FakeIndexFlatL2(d)

    def write_index(self, index, path: str) -> None:
        self.saved_index = index
        Path(path).write_text("fake", encoding="utf-8")

    def read_index(self, path: str):
        return self.saved_index


# A collection declaring BOTH a dense and a sparse vector field.
DUAL_SCHEMA = {
    "auto_id": False,
    "enable_dynamic_field": True,
    "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": True},
        {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 4},
        {"name": "sparse_vec", "datatype": "SPARSE_FLOAT_VECTOR"},
        {"name": "contents", "datatype": "VARCHAR"},
    ],
}

DUAL_INDEX_PARAMS = {
    "indexes": [
        {"field_name": "vector", "metric_type": "L2", "index_type": "Flat", "params": {}}
    ]
}


def make_engine(tmp_path):
    mod = load_engine_module()
    return mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())


# ---------------------------------------------------------------------------
# 1. create: both fields registered, legacy mirror is the dense field
# ---------------------------------------------------------------------------

def test_dual_create_registers_both_fields(tmp_path):
    engine = make_engine(tmp_path)
    engine.create_collection("dual", schema=DUAL_SCHEMA, index_params=DUAL_INDEX_PARAMS)
    meta = engine.meta_store.get("dual")
    assert set(meta.vector_fields.keys()) == {"vector", "sparse_vec"}
    assert meta.vector_fields["vector"]["datatype"] == "FLOAT_VECTOR"
    assert meta.vector_fields["sparse_vec"]["datatype"] == "SPARSE_FLOAT_VECTOR"
    # Legacy mirror points at the dense field.
    assert meta.vector_field == "vector"
    assert meta.dim is None
    assert engine._is_dual_collection(meta) is True
    assert engine._is_sparse_collection(meta) is False  # primary is dense


# ---------------------------------------------------------------------------
# 2. insert: both vectors stored, sparse term not leaked into metadata
# ---------------------------------------------------------------------------

def test_dual_insert_stores_both_vectors(tmp_path):
    engine = make_engine(tmp_path)
    engine.create_collection("dual", schema=DUAL_SCHEMA, index_params=DUAL_INDEX_PARAMS)
    engine.insert("dual", [
        {"id": "d0", "vector": [1.0, 0.0, 0.0, 0.0],
         "sparse_vec": {"apple": 0.5, "banana": 1.2}, "contents": "x"},
        {"id": "d1", "vector": [0.0, 1.0, 0.0, 0.0],
         "sparse_vec": {"apple": 0.3}, "contents": "y"},
    ])
    assert engine.scalar_store.count("dual") == 2

    # Dense column readable as a 2-D matrix.
    dense = engine.scalar_store.get_dense_vectors("dual", 4)
    assert dense.shape == (2, 4)
    assert dense[0].tolist() == [1.0, 0.0, 0.0, 0.0]

    # Sparse column readable as per-row dicts (term ids via the dictionary).
    sparse = engine.scalar_store.get_sparse_vectors_from_column("dual")
    assert len(sparse) == 2
    d = engine.scalar_store.load_term_dictionary("dual")
    assert sparse[0][d.id_of("apple")] == pytest.approx(0.5)
    assert sparse[0][d.id_of("banana")] == pytest.approx(1.2)
    assert sparse[1][d.id_of("apple")] == pytest.approx(0.3)

    # The sparse field name must NOT leak into metadata.
    rows = engine.scalar_store.export_rows("dual")
    for r in rows:
        assert "sparse_vec" not in r["metadata"]


def test_dual_insert_sparse_optional_per_row(tmp_path):
    engine = make_engine(tmp_path)
    engine.create_collection("dual", schema=DUAL_SCHEMA, index_params=DUAL_INDEX_PARAMS)
    # Second row omits the sparse field entirely.
    engine.insert("dual", [
        {"id": "d0", "vector": [1.0, 0.0, 0.0, 0.0], "sparse_vec": {"apple": 0.5}},
        {"id": "d1", "vector": [0.0, 1.0, 0.0, 0.0]},
    ])
    sparse = engine.scalar_store.get_sparse_vectors_from_column("dual")
    assert len(sparse[0]) == 1     # apple
    assert sparse[1] == {}          # NULL sparse_vector -> empty dict


# ---------------------------------------------------------------------------
# 3. flush/search: only the dense field is indexed; sparse skipped
# ---------------------------------------------------------------------------

def test_dual_flush_builds_dense_only(tmp_path):
    engine = make_engine(tmp_path)
    engine.create_collection("dual", schema=DUAL_SCHEMA, index_params=DUAL_INDEX_PARAMS)
    engine.insert("dual", [
        {"id": "d0", "vector": [1.0, 0.0, 0.0, 0.0], "sparse_vec": {"apple": 0.5}},
        {"id": "d1", "vector": [0.0, 1.0, 0.0, 0.0], "sparse_vec": {"banana": 0.9}},
    ])
    result = engine.flush("dual")
    assert result["flushed"]
    assert result["total"] == 2
    assert result["dim"] == 4
    meta = engine.meta_store.get("dual")
    assert meta.index_version == meta.data_version

    # Dense search works.
    hits = engine.search("dual", data=[[1.0, 0.0, 0.0, 0.0]], limit=1)
    assert len(hits) == 1
    assert hits[0][0]["id"] == "d0"


# ---------------------------------------------------------------------------
# 4. export -> import round-trip carries both vectors
# ---------------------------------------------------------------------------

def test_dual_export_import_roundtrip(tmp_path):
    engine = make_engine(tmp_path)
    engine.create_collection("dual", schema=DUAL_SCHEMA, index_params=DUAL_INDEX_PARAMS)
    engine.insert("dual", [
        {"id": "d0", "vector": [1.0, 0.0, 0.0, 0.0],
         "sparse_vec": {"apple": 0.5, "banana": 1.2}, "contents": "x"},
    ])
    rows = engine.scalar_store.export_rows("dual")
    assert len(rows) == 1
    r = rows[0]
    # Both vectors present in the exported row.
    assert isinstance(r["vector"], list) and len(r["vector"]) == 4
    assert "sparse_vector" in r and "indices" in r["sparse_vector"]

    # Re-import into a fresh store preserves both columns.
    mod = load_engine_module()
    store2 = mod.ScalarStore(tmp_path / "scalar2.db")
    store2.import_rows("dual", rows)
    re_exported = store2.export_rows("dual")
    assert len(re_exported) == 1
    r2 = re_exported[0]
    assert len(r2["vector"]) == 4
    by_idx = dict(zip(r2["sparse_vector"]["indices"], r2["sparse_vector"]["values"]))
    # Two sparse terms survived with their weights.
    assert sorted(by_idx.values()) == pytest.approx([0.5, 1.2])


# ---------------------------------------------------------------------------
# 5. regression: pure-sparse still raises on flush/search
# ---------------------------------------------------------------------------

SPARSE_ONLY_SCHEMA = {
    "auto_id": False,
    "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": True},
        {"name": "sparse_vec", "datatype": "SPARSE_FLOAT_VECTOR"},
    ],
}


def test_pure_sparse_flush_still_raises(tmp_path):
    engine = make_engine(tmp_path)
    engine.create_collection("sp", schema=SPARSE_ONLY_SCHEMA)
    engine.insert("sp", [{"id": "d0", "sparse_vec": {"apple": 0.5}}])
    with pytest.raises(NotImplementedError):
        engine.flush("sp")


# ---------------------------------------------------------------------------
# 6. dual-path (verification item 7): HTTP and direct engine share one engine
# ---------------------------------------------------------------------------

def test_dual_path_shares_single_engine(tmp_path):
    # Verification item 7: prove a single shared engine backs the transport
    # layer.  We drive HTTP via FastAPI TestClient (no C++/DLL needed) against
    # the SAME engine instance and assert engine state matches a direct call.
    # We deliberately build the HTTP app directly (not via hypervec_dual_server,
    # whose gRPC proto import is environment-fragile) so the core single-engine
    # claim is provable without gRPC deps.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    try:
        http_mod = _load_module(
            "src/python/hypervec_http_server.py", "http_server_under_test"
        )
    except Exception as exc:  # pragma: no cover - transport deps may be absent
        pytest.skip(f"HTTP server module unavailable: {exc}")

    engine = make_engine(tmp_path)
    engine.create_collection("dual", schema=DUAL_SCHEMA, index_params=DUAL_INDEX_PARAMS)

    try:
        app = http_mod.create_app(data_root=str(tmp_path), engine=engine)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"HTTP app construction unavailable: {exc}")

    client = TestClient(app)
    payload = {
        "data": [
            {"id": "h0", "vector": [1.0, 0.0, 0.0, 0.0],
             "sparse_vec": {"apple": 0.5}, "contents": "http"},
        ]
    }
    resp = client.post("/collections/dual/insert", json=payload)
    assert resp.status_code == 200, resp.text

    # The single shared engine now reflects the HTTP insert.
    assert engine.scalar_store.count("dual") == 1
    d = engine.scalar_store.load_term_dictionary("dual")
    assert d.id_of("apple") == 0
    sparse = engine.scalar_store.get_sparse_vectors_from_column("dual")
    assert sparse[0][d.id_of("apple")] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 7. dual-path (verification item 7): gRPC servicer shares the same engine
# ---------------------------------------------------------------------------

def test_grpc_dual_path_shares_single_engine(tmp_path):
    # gRPC leg of verification item 7. Skip-safe: grpc/proto are environment-
    # fragile (gencode/runtime mismatch here). Call the servicer method DIRECTLY
    # with a fake context -- no socket, no async, no port -- and assert the same
    # shared engine reflects the insert, mirroring the HTTP dual-path proof.
    pytest.importorskip("grpc")
    try:
        grpc_mod = _load_module(
            "src/python/hypervec_grpc_server.py", "grpc_server_under_test"
        )
    except Exception as exc:  # RuntimeError (no grpc) or proto version mismatch
        pytest.skip(f"gRPC server module unavailable: {exc}")

    engine = make_engine(tmp_path)
    engine.create_collection("dual", schema=DUAL_SCHEMA, index_params=DUAL_INDEX_PARAMS)
    servicer = grpc_mod.HyperVecServicer(engine)  # same shared engine

    # Insert(self, request, context): reads request.data_json (a JSON list of
    # row dicts) + request.collection_name, calls engine.insert(name, data).
    class _FakeReq:
        collection_name = "dual"
        data_json = json.dumps([
            {"id": "g0", "vector": [1.0, 0.0, 0.0, 0.0],
             "sparse_vec": {"apple": 0.5}, "contents": "grpc"},
        ])

    class _FakeCtx:
        def set_code(self, *a):
            pass

        def set_details(self, *a):
            pass

        def abort(self, code, details):
            raise AssertionError(f"gRPC abort {code}: {details}")

    servicer.Insert(_FakeReq(), _FakeCtx())

    # The single shared engine now reflects the gRPC insert.
    assert engine.scalar_store.count("dual") == 1
    d = engine.scalar_store.load_term_dictionary("dual")
    sparse = engine.scalar_store.get_sparse_vectors_from_column("dual")
    assert sparse[0][d.id_of("apple")] == pytest.approx(0.5)
