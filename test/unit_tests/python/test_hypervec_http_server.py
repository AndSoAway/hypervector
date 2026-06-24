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


def test_hypervec_http_bundle_and_purge_routes(tmp_path):
    import io
    import zipfile

    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    BUNDLE_FORMAT = "hypervector.collection.bundle.v1"

    # Build a minimal fake bundle for upload tests
    def make_fake_bundle() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            import json
            manifest = {
                "format": BUNDLE_FORMAT,
                "collection_name": "demo",
                "version": 2,
                "dim": 2,
                "total": 1,
                "id_field": "id",
                "vector_field": "vector",
                "text_field": "contents",
                "index_checksum": None,
                "index_size_bytes": 4,
                "scalar_checksum": None,
                "schema_checksum": None,
                "exported_at": 1000.0,
            }
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("index.hypervec", b"fake")
            zf.writestr("scalar.jsonl", b"")
        return buf.getvalue()

    class BundleEngine(FakeEngine):
        def export_collection_bundle(self, collection_name, output_path=None):
            import hashlib as _h
            data = make_fake_bundle()
            path = tmp_path / f"{collection_name}.hypervec-bundle"
            path.write_bytes(data)
            return {
                "collection_name": collection_name,
                "path": str(path),
                "bytes": len(data),
                "version": 2,
                "bundle_format": BUNDLE_FORMAT,
                "bundle_checksum": "sha256:" + _h.sha256(data).hexdigest(),
                "manifest": {},
            }

        def import_collection_bundle(self, collection_name, source_path, *,
                                     checksum=None, mode="replace"):
            return {
                "uploaded": True,
                "collection_name": collection_name,
                "version": 3,
                "total": 1,
                "dim": 2,
                "data_state": "ready",
                "index_checksum": "sha256:abc",
                "index_size_bytes": 4,
            }

        def purge_collection_data(self, collection_name, *, require_exported=True):
            return {
                "purged": True,
                "collection_name": collection_name,
                "metadata_preserved": True,
                "scalar_deleted": True,
                "index_deleted": True,
                "memory_unloaded": True,
                "data_state": "purged",
                "last_known_total": 1,
                "last_purged_at": 1000.0,
            }

    module = load_http_module()
    client = TestClient(module.create_app(data_root=str(tmp_path), engine=BundleEngine()))

    # GET /collections/demo/bundle → 200 binary response
    resp = client.get("/collections/demo/bundle")
    assert resp.status_code == 200
    assert resp.headers["x-hypervec-bundle-format"] == BUNDLE_FORMAT
    assert resp.headers["x-hypervec-collection-version"] == "2"

    # PUT /collections/demo/bundle → 200 JSON
    bundle_bytes = make_fake_bundle()
    resp = client.put("/collections/demo/bundle", content=bundle_bytes)
    assert resp.status_code == 200
    assert resp.json()["uploaded"] is True
    assert resp.json()["data_state"] == "ready"

    # POST /collections/demo/purge-data → 200 JSON
    resp = client.post("/collections/demo/purge-data", json={"require_exported": True})
    assert resp.status_code == 200
    assert resp.json()["purged"] is True
    assert resp.json()["data_state"] == "purged"

    # POST without body (uses default)
    resp = client.post("/collections/demo/purge-data")
    assert resp.status_code == 200


def test_hypervec_http_bundle_404_on_missing_collection(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    class ErrorEngine(FakeEngine):
        def export_collection_bundle(self, collection_name, output_path=None):
            raise FileNotFoundError(f"collection '{collection_name}' does not exist.")

        def import_collection_bundle(self, collection_name, source_path, *,
                                     checksum=None, mode="replace"):
            raise FileNotFoundError(f"collection '{collection_name}' does not exist.")

        def purge_collection_data(self, collection_name, *, require_exported=True):
            raise FileNotFoundError(f"collection '{collection_name}' does not exist.")

    module = load_http_module()
    client = TestClient(module.create_app(data_root=str(tmp_path), engine=ErrorEngine()))

    assert client.get("/collections/missing/bundle").status_code == 404
    assert client.put("/collections/missing/bundle", content=b"x").status_code == 404
    assert client.post("/collections/missing/purge-data").status_code == 404


def test_hypervec_http_purge_409_when_not_exported(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    class NoExportEngine(FakeEngine):
        def purge_collection_data(self, collection_name, *, require_exported=True):
            if require_exported:
                raise ValueError("no recorded export")
            return {"purged": True, "collection_name": collection_name}

    module = load_http_module()
    client = TestClient(module.create_app(data_root=str(tmp_path), engine=NoExportEngine()))

    resp = client.post("/collections/demo/purge-data", json={"require_exported": True})
    assert resp.status_code == 400  # ValueError → 400
