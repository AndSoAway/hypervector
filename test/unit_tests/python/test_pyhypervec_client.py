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
