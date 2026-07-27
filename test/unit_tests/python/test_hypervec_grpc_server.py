from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(_REPO_ROOT / "pyhypervec"))

grpc = pytest.importorskip("grpc")

from hypervec_grpc_server import bind_server, create_server
from hypervec_dual_server import create_dual_services
from hypervec_server_engine import ConflictError
from pyhypervec import (
    CollectionSchema,
    HypervecClient,
    HypervecGrpcError,
)
from pyhypervec import hypervec_pb2


def test_original_grpc_v2_wire_contract_is_preserved():
    service = hypervec_pb2.DESCRIPTOR.services_by_name["HyperVec"]
    original_methods = [
        "Health",
        "ListCollections",
        "HasCollection",
        "DescribeCollection",
        "CreateCollection",
        "DropCollection",
        "Insert",
        "Flush",
        "LoadCollection",
        "CloseCollection",
        "Search",
        "GetVersion",
        "SyncCheck",
        "DownloadIndex",
        "UploadIndex",
    ]
    assert hypervec_pb2.DESCRIPTOR.package == "hypervec"
    assert all(name in service.methods_by_name for name in original_methods)
    assert hypervec_pb2.SearchRequest.DESCRIPTOR.fields_by_name["limit"].number == 3
    assert hypervec_pb2.UploadIndexRequest.DESCRIPTOR.fields_by_name["data"].number == 2


class FakeEngine:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.collections = {"demo"}

    def list_collections(self):
        return sorted(self.collections)

    def describe_collections(self):
        return [self.describe_collection(name) for name in self.list_collections()]

    def supported_index_examples(self):
        return [{"index_type": "IndexIVFFlat"}]

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def describe_collection(self, collection_name):
        if collection_name not in self.collections:
            raise FileNotFoundError(collection_name)
        return {
            "collection_name": collection_name,
            "schema": {"description": "demo"},
            "total": 2,
            "version": 2,
            "data_state": "ready",
        }

    def create_collection(self, collection_name, *, schema, index_params):
        if collection_name in self.collections:
            raise FileExistsError(collection_name)
        self.collections.add(collection_name)
        return {"collection_name": collection_name, "schema": schema, "version": 1}

    def drop_collection(self, collection_name):
        existed = collection_name in self.collections
        self.collections.discard(collection_name)
        return {"dropped": True, "collection_name": collection_name, "existed": existed}

    def insert(self, collection_name, data):
        return {"insert_count": len(data), "total": len(data)}

    def flush(self, collection_name):
        return {"flushed": True, "collection_name": collection_name, "version": 2}

    def load_collection(self, collection_name):
        return {"loaded": True, "collection_name": collection_name}

    def close_collection(self, collection_name):
        return {"closed": True, "collection_name": collection_name}

    def search(self, collection_name, *, data, limit, **kwargs):
        del data, kwargs
        return [[{"id": f"doc-{i}", "distance": float(i), "entity": {}} for i in range(limit)]]

    def get_version(self, collection_name):
        return {
            "collection_name": collection_name,
            "version": 2,
            "index_checksum": "sha256:index",
            "index_size_bytes": 10,
        }

    def sync_check(self, collection_name, *, client_version, client_checksum=None):
        del collection_name, client_checksum
        return {
            "needs_sync": client_version != 2,
            "server_version": 2,
            "client_version": client_version,
        }

    def index_path_for_download(self, collection_name):
        path = self.root / f"{collection_name}.hypervec"
        path.write_bytes(b"index-data")
        return path

    def upload_index(self, collection_name, source_path, *, version=None, checksum=None):
        del source_path, checksum
        return {
            "uploaded": True,
            "collection_name": collection_name,
            "version": version or 3,
        }

    def export_collection_bundle(self, collection_name):
        path = self.root / f"{collection_name}.hypervec-bundle"
        path.write_bytes(b"bundle-data")
        return {
            "path": str(path),
            "version": 2,
            "bundle_format": "hypervector.collection.bundle.v1",
            "bundle_checksum": "sha256:bundle",
            "bytes": len(b"bundle-data"),
        }

    def import_collection_bundle(
        self,
        collection_name,
        source_path,
        *,
        checksum=None,
        mode="replace",
    ):
        del source_path, checksum, mode
        return {
            "uploaded": True,
            "collection_name": collection_name,
            "data_state": "ready",
        }

    def purge_collection_data(self, collection_name, *, require_exported=True):
        if collection_name == "not-exported" and require_exported:
            raise ConflictError("bundle export required")
        return {
            "purged": True,
            "collection_name": collection_name,
            "data_state": "purged",
        }


@pytest.fixture()
def grpc_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERVEC_GRPC_MAX_MESSAGE_MB", "8")
    engine = FakeEngine(tmp_path)
    server = create_server(
        data_root=str(tmp_path),
        engine=engine,
        max_workers=2,
        max_message_mb=8,
    )
    port = bind_server(server, "127.0.0.1:0")
    server.start()
    client = HypervecClient(f"tcp://127.0.0.1:{port}")
    try:
        yield client
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_grpc_current_main_collection_interfaces(grpc_client):
    assert grpc_client.health() == {"status": "ok"}
    assert grpc_client.list_collections() == ["demo"]
    assert grpc_client.describe_collections()[0]["description"] == "demo"
    assert grpc_client.examples()[0]["index_type"] == "IndexIVFFlat"
    assert grpc_client.has_collection("demo")
    assert grpc_client.get_collection_stats("demo") == {"row_count": 2}


def test_grpc_create_insert_search_and_drop(grpc_client):
    schema = CollectionSchema(description="created over grpc")
    created = grpc_client.create_collection("new_collection", schema=schema)
    assert created["collection_name"] == "new_collection"
    assert grpc_client.insert("new_collection", [{"id": "1", "vector": [1.0]}])[
        "insert_count"
    ] == 1
    assert grpc_client.flush("new_collection")["flushed"]
    assert grpc_client.load_collection("new_collection")["loaded"]
    results = grpc_client.search(
        collection_name="new_collection",
        data=[[1.0]],
        limit=2,
    )
    assert [hit["id"] for hit in results[0]] == ["doc-0", "doc-1"]
    assert grpc_client.close_collection("new_collection")["closed"]
    assert grpc_client.drop_collection("new_collection")["existed"]


def test_grpc_version_sync_and_index_transfer(grpc_client, tmp_path):
    assert grpc_client.get_version("demo")["version"] == 2
    assert grpc_client.sync_check("demo", 1)["needs_sync"]

    target = tmp_path / "download.hypervec"
    downloaded = grpc_client.download_index("demo", target)
    assert target.read_bytes() == b"index-data"
    assert downloaded["index_checksum"] == "sha256:index"
    assert grpc_client.upload_index("demo", target, version=9)["version"] == 9


def test_grpc_bundle_and_purge_interfaces(grpc_client, tmp_path):
    target = tmp_path / "demo.hypervec-bundle"
    downloaded = grpc_client.download_collection_bundle("demo", target)
    assert target.read_bytes() == b"bundle-data"
    assert downloaded["bundle_format"] == "hypervector.collection.bundle.v1"

    restored = grpc_client.upload_collection_bundle(
        "demo",
        target,
        checksum="sha256:bundle",
    )
    assert restored["data_state"] == "ready"
    assert grpc_client.purge_collection_data("demo")["data_state"] == "purged"


def test_grpc_errors_keep_status_code(grpc_client):
    with pytest.raises(HypervecGrpcError) as missing:
        grpc_client.describe_collection("missing")
    assert missing.value.status_code == "NOT_FOUND"

    with pytest.raises(HypervecGrpcError) as conflict:
        grpc_client.purge_collection_data("not-exported")
    assert conflict.value.status_code == "FAILED_PRECONDITION"
    assert grpc_client.purge_collection_data(
        "not-exported",
        require_exported=False,
    )["purged"]


def test_dual_protocols_share_one_engine(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("HYPERVEC_GRPC_MAX_MESSAGE_MB", "8")
    engine = FakeEngine(tmp_path)
    shared, http_app, grpc_server = create_dual_services(
        data_root=str(tmp_path),
        engine=engine,
        grpc_workers=2,
        grpc_max_message_mb=8,
    )
    assert shared is engine

    port = bind_server(grpc_server, "127.0.0.1:0")
    grpc_server.start()
    grpc_client = HypervecClient(f"grpc://127.0.0.1:{port}")
    try:
        http_client = TestClient(http_app)
        response = http_client.post(
            "/collections/from_http/create",
            json={"schema": {"fields": []}, "index_params": {"indexes": []}},
        )
        assert response.status_code == 200
        assert "from_http" in grpc_client.list_collections()
    finally:
        grpc_client.close()
        grpc_server.stop(grace=0).wait()
