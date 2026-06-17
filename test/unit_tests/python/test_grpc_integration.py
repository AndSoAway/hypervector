"""
Integration tests: gRPC main chain and HTTP/gRPC parity.

Requires the hypervec C++ module to be importable.
Uses the grpc_port / http_port fixtures from conftest.py (real engine,
shared tmp_path, in-process servers).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT / "pyhypervec"))

from pyhypervec import HypervecClient
from pyhypervec.schema import CollectionSchema

DIM = 8


def _schema() -> CollectionSchema:
    s = CollectionSchema()
    s.add_field("id", "VARCHAR", is_primary=True, max_length=64)
    s.add_field("vector", "FLOAT_VECTOR", dim=DIM)
    s.add_field("contents", "VARCHAR", max_length=60000)
    return s


def _rows(n: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, DIM), dtype=np.float32).tolist()
    return [
        {"id": f"doc{i}", "vector": vecs[i], "contents": f"text {i}"}
        for i in range(n)
    ]


def _col() -> str:
    return f"t_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# gRPC main chain
# ---------------------------------------------------------------------------

class TestGrpcMainChain:
    @pytest.fixture(autouse=True)
    def _client(self, grpc_port):
        self.client = HypervecClient(uri=f"tcp://localhost:{grpc_port}")
        yield
        self.client.close()

    def test_health(self):
        resp = self.client.health()
        assert resp["status"] == "ok"

    def test_create_list_drop(self):
        name = _col()
        self.client.create_collection(name, schema=_schema())
        assert name in self.client.list_collections()
        self.client.drop_collection(name)
        assert name not in self.client.list_collections()

    def test_has_collection(self):
        name = _col()
        assert not self.client.has_collection(name)
        self.client.create_collection(name, schema=_schema())
        try:
            assert self.client.has_collection(name)
        finally:
            self.client.drop_collection(name)

    def test_describe_collection(self):
        name = _col()
        self.client.create_collection(name, schema=_schema())
        try:
            info = self.client.describe_collection(name)
            assert info["collection_name"] == name
        finally:
            self.client.drop_collection(name)

    def test_insert_flush_load_search(self):
        name = _col()
        self.client.create_collection(name, schema=_schema())
        try:
            self.client.insert(name, _rows(5))
            self.client.flush(name)
            self.client.load_collection(name)

            rng = np.random.default_rng(99)
            query = rng.random((1, DIM), dtype=np.float32).tolist()
            results = self.client.search(collection_name=name, data=query, limit=3)

            assert len(results) == 1
            assert len(results[0]) == 3
            for hit in results[0]:
                assert "id" in hit or "distance" in hit  # engine returns hit dict
        finally:
            self.client.drop_collection(name)

    def test_version(self):
        name = _col()
        self.client.create_collection(name, schema=_schema())
        try:
            self.client.insert(name, _rows(2))
            ver = self.client.get_version(name)
            assert ver["version"] >= 0
        finally:
            self.client.drop_collection(name)

    def test_sync_check_needs_sync(self):
        name = _col()
        self.client.create_collection(name, schema=_schema())
        try:
            self.client.insert(name, _rows(2))
            self.client.flush(name)
            result = self.client.sync_check(name, client_version=0)
            assert result["needs_sync"] is True
            assert result["server_version"] >= 1
        finally:
            self.client.drop_collection(name)

    def test_close_collection(self):
        name = _col()
        self.client.create_collection(name, schema=_schema())
        try:
            result = self.client.close_collection(name)
            assert result.get("closed") is True
        finally:
            self.client.drop_collection(name)

    def test_collection_not_found(self):
        from pyhypervec import HypervecClientError
        with pytest.raises(HypervecClientError):
            self.client.describe_collection("no_such_collection_xyz")

    def test_collection_already_exists(self):
        from pyhypervec import HypervecClientError
        name = _col()
        self.client.create_collection(name, schema=_schema())
        try:
            with pytest.raises(HypervecClientError):
                self.client.create_collection(name, schema=_schema())
        finally:
            self.client.drop_collection(name)

    def test_download_upload_index(self, tmp_path):
        name = _col()
        self.client.create_collection(name, schema=_schema())
        try:
            self.client.insert(name, _rows(3))
            self.client.flush(name)

            target = tmp_path / "downloaded.hypervec"
            dl = self.client.download_index(name, target)
            assert target.exists()
            assert int(dl["bytes"]) > 0

            # Upload the same index back (version bump)
            ul = self.client.upload_index(name, target, version=99)
            assert ul.get("version") is not None
        finally:
            self.client.drop_collection(name)


# ---------------------------------------------------------------------------
# HTTP / gRPC parity (shared engine)
# ---------------------------------------------------------------------------

class TestHTTPGRPCParity:
    @pytest.fixture(autouse=True)
    def _clients(self, grpc_port, http_port):
        self.grpc = HypervecClient(uri=f"tcp://localhost:{grpc_port}")
        self.http = HypervecClient(uri=f"http://localhost:{http_port}")
        yield
        self.grpc.close()

    def test_search_results_match(self):
        name = _col()
        rows = _rows(10)
        self.grpc.create_collection(name, schema=_schema())
        try:
            self.grpc.insert(name, rows)
            self.grpc.flush(name)

            rng = np.random.default_rng(42)
            query = rng.random((1, DIM), dtype=np.float32).tolist()

            grpc_res = self.grpc.search(collection_name=name, data=query, limit=5)
            http_res = self.http.search(collection_name=name, data=query, limit=5)

            assert len(grpc_res) == len(http_res) == 1
            grpc_ids = {h.get("id") or h.get("doc_id") for h in grpc_res[0]}
            http_ids = {h.get("id") or h.get("doc_id") for h in http_res[0]}
            assert grpc_ids == http_ids, f"gRPC ids {grpc_ids} != HTTP ids {http_ids}"
        finally:
            self.grpc.drop_collection(name)

    def test_version_parity(self):
        name = _col()
        self.grpc.create_collection(name, schema=_schema())
        try:
            self.grpc.insert(name, _rows(3))
            self.grpc.flush(name)

            grpc_ver = self.grpc.get_version(name)
            http_ver = self.http.get_version(name)

            assert grpc_ver["version"] == http_ver["version"]
        finally:
            self.grpc.drop_collection(name)

    def test_list_collections_parity(self):
        name = _col()
        self.http.create_collection(name, schema=_schema())
        try:
            assert name in self.grpc.list_collections()
            assert name in self.http.list_collections()
        finally:
            self.grpc.drop_collection(name)
