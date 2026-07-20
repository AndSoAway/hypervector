from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


class FakeEngine:
    def list_collections(self):
        return ["demo"]

    def has_collection(self, collection_name):
        return collection_name == "demo"

    def describe_collection(self, collection_name):
        return {"collection_name": collection_name}

    def create_collection(self, collection_name, *, schema, index_params=None):
        return {"collection_name": collection_name, "schema": schema, "index_params": index_params}

    def drop_collection(self, collection_name):
        return {"dropped": True, "collection_name": collection_name}

    def insert(self, collection_name, data):
        return {"insert_count": len(data), "collection_name": collection_name}

    def flush(self, collection_name):
        return {"flushed": True, "collection_name": collection_name}

    def load_collection(self, collection_name):
        return {"loaded": True, "collection_name": collection_name}

    def close_collection(self, collection_name):
        return {"closed": True, "collection_name": collection_name}

    def search(self, collection_name, **kwargs):
        return [[{"id": "a", "distance": 0.1, "entity": {"id": "a"}}]]

    def get_version(self, collection_name):
        return {"version": 2, "index_checksum": "sha256:abc", "index_size_bytes": 4}

    def sync_check(self, collection_name, *, client_version, client_checksum=None):
        return {"needs_sync": client_version != 2, "server_version": 2}

    def index_path_for_download(self, collection_name):
        return Path(__file__)

    def upload_index(self, collection_name, source_path, *, version=None, checksum=None):
        return {"uploaded": True, "version": version, "collection_name": collection_name}


def load_grpc_module():
    root = Path(__file__).parents[3] / "src" / "python"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "hypervec_grpc_server",
        root / "hypervec_grpc_server.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hypervec_grpc_server"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hypervec_grpc_service_reuses_engine(tmp_path):
    module = load_grpc_module()
    service = module.HypervecGrpcServicer(FakeEngine())

    assert service.Health(None, None).status == "ok"
    assert list(service.ListCollections(None, None).collections) == ["demo"]
    assert service.HasCollection(type("Req", (), {"collection_name": "demo"})(), None).exists
    assert service.DescribeCollection(type("Req", (), {"collection_name": "demo"})(), None).json
    assert service.Search(
        type("Req", (), {
            "collection_name": "demo",
            "data_json": "[[0.1, 0.2]]",
            "limit": 1,
            "search_params_json": "{}",
            "output_fields": ["id"],
            "filter": "",
            "consistency_level": "",
        })(),
        None,
    ).results[0].hits[0].id == "a"
    assert service.DownloadIndex(type("Req", (), {"collection_name": "demo"})(), None).version == 2
