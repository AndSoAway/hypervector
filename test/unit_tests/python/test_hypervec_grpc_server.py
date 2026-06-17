"""
Unit tests for the HyperVec gRPC server using a FakeEngine.

Follows the same pattern as test_hypervec_http_server.py — no real
index or SWIG module needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SRC_PYTHON = Path(__file__).parents[3] / "src" / "python"
sys.path.insert(0, str(_SRC_PYTHON))

grpc = pytest.importorskip("grpc")


# ---------------------------------------------------------------------------
# Fake engine (mirrors FakeEngine in test_hypervec_http_server.py)
# ---------------------------------------------------------------------------

class FakeEngine:
    def list_collections(self):
        return ["demo"]

    def has_collection(self, collection_name):
        return collection_name == "demo"

    def describe_collection(self, collection_name):
        if collection_name != "demo":
            raise FileNotFoundError(f"collection '{collection_name}' does not exist.")
        return {"collection_name": collection_name, "dim": 4, "total": 2, "version": 1}

    def create_collection(self, collection_name, *, schema, index_params):
        if collection_name == "demo":
            raise FileExistsError(f"collection '{collection_name}' already exists.")
        return {"collection_name": collection_name, "dim": None, "total": 0, "version": 0}

    def drop_collection(self, collection_name):
        return {"dropped": True, "collection_name": collection_name, "existed": True}

    def insert(self, collection_name, data):
        return {"insert_count": len(data), "total": len(data)}

    def flush(self, collection_name):
        return {"flushed": True, "collection_name": collection_name, "version": 1, "total": 2}

    def load_collection(self, collection_name):
        return {"loaded": True, "collection_name": collection_name}

    def close_collection(self, collection_name):
        return {"closed": True, "collection_name": collection_name}

    def search(self, collection_name, *, data, limit, **kwargs):
        return [[{"id": f"doc{i}", "distance": float(i) * 0.1, "entity": {}} for i in range(limit)]]

    def get_version(self, collection_name):
        return {"collection_name": collection_name, "version": 2, "updated_at": 1.0,
                "index_checksum": "sha256:abc", "index_size_bytes": 4}

    def sync_check(self, collection_name, *, client_version, client_checksum=None):
        return {"needs_sync": client_version != 2, "server_version": 2,
                "client_version": client_version}

    def index_path_for_download(self, collection_name):
        return Path(__file__)

    def upload_index(self, collection_name, source_path, *, version=None, checksum=None):
        return {"uploaded": True, "collection_name": collection_name, "version": version or 3}


# ---------------------------------------------------------------------------
# Helpers to load the server module
# ---------------------------------------------------------------------------

def load_grpc_module():
    spec = importlib.util.spec_from_file_location(
        "hypervec_grpc_server",
        _SRC_PYTHON / "hypervec_grpc_server.py",
        submodule_search_locations=[str(_SRC_PYTHON)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hypervec_grpc_server"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def grpc_channel():
    """Start an in-process gRPC server with FakeEngine; yield a channel."""
    from concurrent.futures import ThreadPoolExecutor
    import grpc as _grpc

    module = load_grpc_module()
    import hypervec_pb2_grpc as pb2_grpc

    engine = FakeEngine()
    server = _grpc.server(ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_HyperVecServicer_to_server(module.HyperVecServicer(engine), server)
    port = _free_port()
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    channel = _grpc.insecure_channel(f"localhost:{port}")
    yield channel
    channel.close()
    server.stop(grace=0).wait()


@pytest.fixture(scope="module")
def stub(grpc_channel):
    import hypervec_pb2_grpc as pb2_grpc
    return pb2_grpc.HyperVecStub(grpc_channel)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(stub):
    import hypervec_pb2 as pb2
    resp = stub.Health(pb2.HealthRequest())
    assert resp.status == "ok"


def test_list_collections(stub):
    import hypervec_pb2 as pb2
    resp = stub.ListCollections(pb2.ListCollectionsRequest())
    assert "demo" in list(resp.collections)


def test_has_collection_true(stub):
    import hypervec_pb2 as pb2
    resp = stub.HasCollection(pb2.HasCollectionRequest(collection_name="demo"))
    assert resp.exists is True


def test_has_collection_false(stub):
    import hypervec_pb2 as pb2
    resp = stub.HasCollection(pb2.HasCollectionRequest(collection_name="not_exists"))
    assert resp.exists is False


def test_describe_collection(stub):
    import hypervec_pb2 as pb2
    resp = stub.DescribeCollection(pb2.DescribeCollectionRequest(collection_name="demo"))
    data = json.loads(resp.json_payload)
    assert data["collection_name"] == "demo"


def test_describe_collection_not_found(stub):
    import grpc as _grpc
    import hypervec_pb2 as pb2
    with pytest.raises(_grpc.RpcError) as exc_info:
        stub.DescribeCollection(pb2.DescribeCollectionRequest(collection_name="missing"))
    assert exc_info.value.code() == _grpc.StatusCode.NOT_FOUND


def test_create_collection(stub):
    import hypervec_pb2 as pb2
    schema = {"auto_id": False, "fields": [{"name": "vec", "datatype": "FLOAT_VECTOR", "dim": 4}]}
    resp = stub.CreateCollection(pb2.CreateCollectionRequest(
        collection_name="new_col",
        schema_json=json.dumps(schema),
        index_params_json=json.dumps({"indexes": []}),
    ))
    data = json.loads(resp.json_payload)
    assert data["collection_name"] == "new_col"


def test_create_collection_already_exists(stub):
    import grpc as _grpc
    import hypervec_pb2 as pb2
    schema = {"auto_id": False, "fields": []}
    with pytest.raises(_grpc.RpcError) as exc_info:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection_name="demo",
            schema_json=json.dumps(schema),
            index_params_json="{}",
        ))
    assert exc_info.value.code() == _grpc.StatusCode.ALREADY_EXISTS


def test_drop_collection(stub):
    import hypervec_pb2 as pb2
    resp = stub.DropCollection(pb2.DropCollectionRequest(collection_name="some_col"))
    data = json.loads(resp.json_payload)
    assert data["dropped"] is True


def test_insert(stub):
    import hypervec_pb2 as pb2
    rows = [{"id": "d1", "vector": [0.1, 0.2, 0.3, 0.4]}]
    resp = stub.Insert(pb2.InsertRequest(
        collection_name="demo",
        data_json=json.dumps(rows),
    ))
    data = json.loads(resp.json_payload)
    assert data["insert_count"] == 1


def test_flush(stub):
    import hypervec_pb2 as pb2
    resp = stub.Flush(pb2.FlushRequest(collection_name="demo"))
    data = json.loads(resp.json_payload)
    assert data["flushed"] is True


def test_load_collection(stub):
    import hypervec_pb2 as pb2
    resp = stub.LoadCollection(pb2.LoadCollectionRequest(collection_name="demo"))
    data = json.loads(resp.json_payload)
    assert data["loaded"] is True


def test_close_collection(stub):
    import hypervec_pb2 as pb2
    resp = stub.CloseCollection(pb2.CloseCollectionRequest(collection_name="demo"))
    data = json.loads(resp.json_payload)
    assert data["closed"] is True


def test_search(stub):
    import hypervec_pb2 as pb2
    resp = stub.Search(pb2.SearchRequest(
        collection_name="demo",
        query_json=json.dumps([[0.1, 0.2, 0.3, 0.4]]),
        limit=3,
        search_params_json="{}",
    ))
    results = json.loads(resp.results_json)
    assert len(results) == 1
    assert len(results[0]) == 3


def test_get_version(stub):
    import hypervec_pb2 as pb2
    resp = stub.GetVersion(pb2.GetVersionRequest(collection_name="demo"))
    data = json.loads(resp.json_payload)
    assert data["version"] == 2


def test_sync_check_needs_sync(stub):
    import hypervec_pb2 as pb2
    resp = stub.SyncCheck(pb2.SyncCheckRequest(
        collection_name="demo",
        client_version=0,
    ))
    data = json.loads(resp.json_payload)
    assert data["needs_sync"] is True
    assert data["server_version"] == 2


def test_sync_check_up_to_date(stub):
    import hypervec_pb2 as pb2
    resp = stub.SyncCheck(pb2.SyncCheckRequest(
        collection_name="demo",
        client_version=2,
    ))
    data = json.loads(resp.json_payload)
    assert data["needs_sync"] is False


def test_download_index(stub):
    import hypervec_pb2 as pb2
    resp = stub.DownloadIndex(pb2.DownloadIndexRequest(collection_name="demo"))
    assert len(resp.data) > 0
    assert resp.version == 2


def test_upload_index(stub, tmp_path):
    import hypervec_pb2 as pb2
    fake_index = b"fake-index-bytes"
    resp = stub.UploadIndex(pb2.UploadIndexRequest(
        collection_name="demo",
        data=fake_index,
        version=3,
        checksum="",
    ))
    data = json.loads(resp.json_payload)
    assert data["uploaded"] is True
