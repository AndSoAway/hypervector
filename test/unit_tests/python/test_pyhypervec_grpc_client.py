from __future__ import annotations

from concurrent import futures
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "pyhypervec"))
sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python"))

from pyhypervec import CollectionSchema, DataType, HypervecClient, IndexParams
from pyhypervec import hypervec_service_pb2 as pb2


class FakeGrpcStub:
    def __init__(self) -> None:
        self.calls = []

    def Health(self, request):
        self.calls.append(("Health", request))
        return pb2.HealthResponse(status="ok")

    def Search(self, request):
        self.calls.append(("Search", request))
        return pb2.SearchResponse(
            results=[
                pb2.SearchResult(
                    hits=[pb2.SearchHit(id="a", distance=0.1, entity_json=json.dumps({"id": "a"}))]
                )
            ]
        )

    def DownloadIndex(self, request):
        self.calls.append(("DownloadIndex", request))
        return pb2.DownloadIndexResponse(
            data=b"index",
            version=3,
            index_checksum="sha256:abc",
            index_size_bytes=5,
        )


class FakeEngine:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.collections = {"demo": []}
        self.uploaded_index = b""
        self.index_file = None
        if root is not None:
            self.index_file = root / "demo.index"
            self.index_file.write_bytes(b"index-data")

    def list_collections(self):
        return sorted(self.collections)

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def describe_collection(self, collection_name):
        return {"collection_name": collection_name, "dim": 2, "total": len(self.collections.get(collection_name, []))}

    def create_collection(self, collection_name, *, schema, index_params=None):
        self.collections[collection_name] = []
        return {"created": True, "collection_name": collection_name, "schema": schema, "index_params": index_params}

    def drop_collection(self, collection_name):
        self.collections.pop(collection_name, None)
        return {"dropped": True, "collection_name": collection_name}

    def insert(self, collection_name, data):
        self.collections.setdefault(collection_name, []).extend(data)
        return {"insert_count": len(data), "total": len(self.collections[collection_name])}

    def flush(self, collection_name):
        return {"flushed": True, "collection_name": collection_name}

    def load_collection(self, collection_name):
        return {"loaded": True, "collection_name": collection_name}

    def close_collection(self, collection_name):
        return {"closed": True, "collection_name": collection_name}

    def search(self, collection_name, **kwargs):
        return [[{"id": "vec_1", "distance": 0.01, "entity": {"id": "vec_1"}}]]

    def get_version(self, collection_name):
        data = self.index_file.read_bytes() if self.index_file is not None else b"demo"
        return {
            "version": 9,
            "index_checksum": f"sha256:{hashlib.sha256(data).hexdigest()}",
            "index_size_bytes": len(data),
        }

    def sync_check(self, collection_name, *, client_version, client_checksum=None):
        current = self.get_version(collection_name)
        return {
            "needs_sync": client_version != current["version"] or client_checksum != current["index_checksum"],
            "server_version": current["version"],
            "server_checksum": current["index_checksum"],
        }

    def index_path_for_download(self, collection_name):
        assert self.index_file is not None
        return str(self.index_file)

    def upload_index(self, collection_name, path, version=None, checksum=None):
        self.uploaded_index = Path(path).read_bytes()
        actual_checksum = f"sha256:{hashlib.sha256(self.uploaded_index).hexdigest()}"
        return {
            "uploaded": True,
            "collection_name": collection_name,
            "bytes": len(self.uploaded_index),
            "version": version,
            "checksum": checksum,
            "actual_checksum": actual_checksum,
        }


def test_pyhypervec_grpc_json_requests_use_generated_messages(monkeypatch):
    client = HypervecClient("tcp://localhost:50051")
    stub = FakeGrpcStub()
    monkeypatch.setattr(client, "_grpc_stub", lambda: stub)

    assert client.health()["status"] == "ok"
    result = client.search(
        collection_name="demo",
        data=[[0.1, 0.2]],
        limit=1,
        output_fields=["id"],
    )

    assert result[0][0]["id"] == "a"
    assert stub.calls[0][0] == "Health"
    assert stub.calls[1][0] == "Search"
    assert stub.calls[1][1].collection_name == "demo"
    assert stub.calls[1][1].limit == 1


def test_pyhypervec_grpc_bytes_requests_use_generated_messages(monkeypatch, tmp_path):
    client = HypervecClient("tcp://localhost:50051")
    stub = FakeGrpcStub()
    monkeypatch.setattr(client, "_grpc_stub", lambda: stub)
    target = tmp_path / "index.bin"

    downloaded = client.download_index("demo", target)

    assert target.read_bytes() == b"index"
    assert downloaded["version"] == "3"
    assert stub.calls[0][0] == "DownloadIndex"
    assert stub.calls[0][1].collection_name == "demo"


def test_pyhypervec_tcp_uri_talks_to_real_grpc_server(tmp_path):
    import grpc
    import hypervec_service_pb2_grpc as server_pb2_grpc
    from hypervec_grpc_server import HypervecGrpcServicer

    engine = FakeEngine(tmp_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    server_pb2_grpc.add_HypervecServiceServicer_to_server(HypervecGrpcServicer(engine), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        client = HypervecClient(f"tcp://127.0.0.1:{port}")
        schema = CollectionSchema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        index_params = IndexParams()
        index_params.add_index("vector", index_type="HNSWFlat", metric_type="L2")

        assert client.health()["status"] == "ok"
        assert client.create_collection("stage", schema=schema, index_params=index_params)["created"]
        assert client.list_collections() == ["demo", "stage"]
        assert client.has_collection("stage")
        assert client.describe_collection("stage")["dim"] == 2
        assert client.insert("stage", [{"id": "vec_1", "vector": [0.1, 0.2]}])["insert_count"] == 1
        assert client.flush("stage")["flushed"]
        assert client.load_collection("stage")["loaded"]
        assert client.search(collection_name="stage", data=[[0.1, 0.2]], limit=1)[0][0]["id"] == "vec_1"
        version_info = client.get_version("stage")
        assert version_info["version"] == 9
        assert client.sync_check("stage", client_version=1, client_checksum="wrong")["needs_sync"]
        download_target = tmp_path / "downloaded.index"
        downloaded = client.download_index("stage", download_target)
        assert download_target.read_bytes() == b"index-data"
        assert downloaded["index_checksum"].startswith("sha256:")
        upload_target = tmp_path / "upload.index"
        upload_target.write_bytes(b"upload-data")
        uploaded = client.upload_index("stage", upload_target, version=10, checksum="sha256:upload")
        assert uploaded["uploaded"]
        assert client.close_collection("stage")["closed"]
        assert client.drop_collection("stage")["dropped"]
        assert client.list_collections() == ["demo"]
        assert engine.uploaded_index == b"upload-data"
    finally:
        server.stop(0)
