from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


class FakeEngine:
    def health(self):
        return {"status": "ok"}

    def list_collections(self):
        return ["demo"]

    def has_collection(self, collection_name):
        return collection_name == "demo"

    def describe_collection(self, collection_name):
        return {"collection_name": collection_name}

    def describe_collections(self):
        return [
            {"collection_name": "demo"},
            {"collection_name": "other"},
        ]

    def get_version(self, collection_name):
        return {
            "collection_name": collection_name,
            "version": 2,
            "updated_at": 1.0,
            "index_checksum": "sha256:abc",
            "index_size_bytes": 4,
        }

    def sync_check(self, collection_name, *, client_version, client_checksum=None):
        return {
            "needs_sync": client_version != 2 or client_checksum != "sha256:abc",
            "server_version": 2,
            "client_version": client_version,
        }

    def index_path_for_download(self, collection_name):
        return Path(__file__)

    def upload_index(self, collection_name, source_path, *, version=None, checksum=None):
        return {"uploaded": True, "collection_name": collection_name, "version": version}


def load_http_module():
    root = Path(__file__).parents[3] / "src" / "python"
    package = type(sys)("hypervec")
    package.__path__ = [str(root)]
    sys.modules.setdefault("hypervec", package)
    spec = importlib.util.spec_from_file_location(
        "hypervec.hypervec_http_server",
        root / "hypervec_http_server.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hypervec.hypervec_http_server"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hypervec_http_server_sync_routes(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    module = load_http_module()
    client = TestClient(module.create_app(data_root=str(tmp_path), engine=FakeEngine()))

    described = client.get("/collections/describe")
    assert described.json()["collections"] == [
        {"collection_name": "demo"},
        {"collection_name": "other"},
    ]

    assert client.get("/collections/demo/version").json()["version"] == 2
    sync = client.post(
        "/collections/demo/sync-check",
        json={"client_version": 1, "client_checksum": "sha256:old"},
    )
    assert sync.json()["needs_sync"]

    download = client.get("/collections/demo/index")
    assert download.status_code == 200
    assert download.headers["x-hypervec-collection-version"] == "2"

    upload = client.put("/collections/demo/index?version=3", content=b"fake-index")
    assert upload.json()["uploaded"]
