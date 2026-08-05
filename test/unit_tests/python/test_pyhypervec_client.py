from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "pyhypervec"))

from pyhypervec import DataType, HypervecClient


def test_pyhypervec_schema_and_index_params_are_serializable():
    schema = HypervecClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
        description="demo",
    )
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
    schema.add_field("contents", DataType.VARCHAR, max_length=60000)

    index_params = HypervecClient("http://localhost:8080").prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="L2",
        index_type="HNSWFlat",
        params={"M": 16},
    )

    assert schema.to_dict()["fields"][1] == {
        "name": "vector",
        "datatype": "FLOAT_VECTOR",
        "dim": 2,
    }
    assert index_params.to_dict()["indexes"][0]["params"]["M"] == 16


def test_pyhypervec_client_search_uses_milvus_like_payload(monkeypatch):
    calls = []
    client = HypervecClient("http://localhost:8080")

    def fake_request(method, path, *, body=None):
        calls.append((method, path, body))
        return {"results": [[{"id": "a", "distance": 0.1, "entity": {"id": "a"}}]]}

    monkeypatch.setattr(client, "_request", fake_request)
    res = client.search(
        collection_name="demo",
        data=[[0.1, 0.2]],
        limit=3,
        output_fields=["id", "contents"],
        filter="source == 'manual'",
        consistency_level="Bounded",
    )

    assert res[0][0]["id"] == "a"
    assert calls == [
        (
            "POST",
            "/collections/demo/search",
            {
                "data": [[0.1, 0.2]],
                "limit": 3,
                "search_params": {},
                "output_fields": ["id", "contents"],
                "filter": "source == 'manual'",
                "consistency_level": "Bounded",
            },
        )
    ]


def test_pyhypervec_client_collection_description_and_stats_are_milvus_compatible(monkeypatch):
    calls = []
    client = HypervecClient("http://localhost:8080")

    def fake_request(method, path, *, body=None):
        calls.append((method, path, body))
        return {
            "collection_name": "demo",
            "schema": {"description": "HyperVec demo backend | display_name=Demo"},
            "total": 1846,
        }

    monkeypatch.setattr(client, "_request", fake_request)

    desc = client.describe_collection("demo")
    stats = client.get_collection_stats("demo")

    assert desc["description"] == "HyperVec demo backend | display_name=Demo"
    assert stats == {"row_count": 1846}
    assert calls == [
        ("GET", "/collections/demo/describe", None),
        ("GET", "/collections/demo/describe", None),
    ]


def test_pyhypervec_client_describe_collections(monkeypatch):
    calls = []
    client = HypervecClient("http://localhost:8080")

    def fake_request(method, path, *, body=None):
        calls.append((method, path, body))
        return {
            "collections": [
                {
                    "collection_name": "demo",
                    "schema": {"description": "Demo collection"},
                    "total": 3,
                },
                {
                    "collection_name": "empty",
                    "schema": {},
                    "total": 0,
                },
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)

    descs = client.describe_collections()

    assert [desc["collection_name"] for desc in descs] == ["demo", "empty"]
    assert descs[0]["description"] == "Demo collection"
    assert descs[1]["description"] == ""
    assert calls == [("GET", "/collections/describe", None)]


def test_pyhypervec_client_examples(monkeypatch):
    calls = []
    client = HypervecClient("http://localhost:8080")

    def fake_request(method, path, *, body=None):
        calls.append((method, path, body))
        return {
            "examples": [
                {"index_type": "IndexIVFFlat", "cpp_class": "hypervec.IndexIVFFlat"},
                {"index_type": "IndexHNSWPQ", "cpp_class": "hypervec.IndexHNSWPQ"},
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)

    examples = client.examples()

    assert [item["index_type"] for item in examples] == ["IndexIVFFlat", "IndexHNSWPQ"]
    assert calls == [("GET", "/examples", None)]


def test_pyhypervec_client_version_and_sync_payload(monkeypatch):
    calls = []
    client = HypervecClient("http://localhost:8080")

    def fake_request(method, path, *, body=None):
        calls.append((method, path, body))
        if path.endswith("/version"):
            return {"version": 2}
        return {"needs_sync": True}

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.get_version("demo")["version"] == 2
    assert client.sync_check("demo", 1, "sha256:abc")["needs_sync"]
    assert calls == [
        ("GET", "/collections/demo/version", None),
        (
            "POST",
            "/collections/demo/sync-check",
            {"client_version": 1, "client_checksum": "sha256:abc"},
        ),
    ]


def test_pyhypervec_client_download_and_upload_index(monkeypatch, tmp_path):
    calls = []
    client = HypervecClient("http://localhost:8080")
    source = tmp_path / "source.hypervec"
    target = tmp_path / "target.hypervec"
    source.write_bytes(b"index-bytes")

    def fake_request_bytes(method, path, *, body=None, content_type="application/octet-stream"):
        calls.append((method, path, body, content_type))
        if method == "GET":
            return b"downloaded", {
                "X-Hypervec-Collection-Version": "2",
                "X-Hypervec-Index-Checksum": "sha256:abc",
                "X-Hypervec-Index-Size": "10",
            }
        return b'{"uploaded":true,"version":3}', {}

    monkeypatch.setattr(client, "_request_bytes", fake_request_bytes)

    downloaded = client.download_index("demo", target)
    assert target.read_bytes() == b"downloaded"
    assert downloaded["version"] == "2"

    uploaded = client.upload_index("demo", source, version=3, checksum="sha256:def")
    assert uploaded["uploaded"]
    assert calls == [
        ("GET", "/collections/demo/index", None, "application/octet-stream"),
        (
            "PUT",
            "/collections/demo/index?version=3&checksum=sha256%3Adef",
            b"index-bytes",
            "application/octet-stream",
        ),
    ]


def test_pyhypervec_client_http2_json_requests(monkeypatch):
    calls = []
    client = HypervecClient("http://localhost:8080", http2=True)

    def fake_request_http2(method, path, *, body=None, content_type="application/octet-stream"):
        calls.append((method, path, body, content_type))
        return b'{"status":"ok"}', {}

    monkeypatch.setattr(client, "_request_http2", fake_request_http2)

    assert client.health()["status"] == "ok"
    assert calls == [
        ("GET", "/health", None, "application/json"),
    ]


def test_pyhypervec_client_http2_download_uses_case_insensitive_headers(monkeypatch, tmp_path):
    client = HypervecClient("http://localhost:8080", http2=True)
    target = tmp_path / "target.hypervec"

    def fake_request_http2(method, path, *, body=None, content_type="application/octet-stream"):
        return b"downloaded", {
            "x-hypervec-collection-version": "5",
            "x-hypervec-index-checksum": "sha256:abc",
            "x-hypervec-index-size": "10",
        }

    monkeypatch.setattr(client, "_request_http2", fake_request_http2)

    downloaded = client.download_index("demo", target)
    assert downloaded["version"] == "5"
    assert downloaded["index_checksum"] == "sha256:abc"
    assert downloaded["index_size_bytes"] == "10"
