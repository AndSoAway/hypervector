# pyhypervec and HyperVec HTTP Server

This document describes the first-stage server/client split:

```text
client process -> pyhypervec.HypervecClient -> HTTP -> HyperVec server -> hypervec core
```

`pyhypervec` is a pure Python HTTP client package. The HyperVec server is shipped
inside the `hypervec` Python package and directly uses the compiled HyperVec
Python binding.

## Client

Install the client package from source:

```bash
cd pyhypervec
python -m pip install .
```

Minimal usage:

```python
from pyhypervec import DataType, HypervecClient

client = HypervecClient("http://127.0.0.1:8080")

schema = HypervecClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
    description="demo",
)
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
schema.add_field("contents", DataType.VARCHAR, max_length=60000)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="L2",
    index_type="Flat",
)

client.create_collection(
    collection_name="demo",
    schema=schema,
    index_params=index_params,
)
client.insert(
    "demo",
    data=[
        {"id": "a", "vector": [0, 0], "contents": "zero", "source": "manual"},
        {"id": "b", "vector": [1, 1], "contents": "one", "source": "manual"},
    ],
)
client.flush("demo")
client.load_collection("demo")
results = client.search(
    collection_name="demo",
    data=[[0.1, 0.1]],
    limit=2,
    output_fields=["id", "contents", "source"],
)
```

## Server

Install the compiled `hypervec` package with server dependencies:

```bash
python -m pip install "hypervec[server]"
```

Start the HTTP server:

```bash
python -m hypervec.hypervec_http_server \
  --data-root /data/hypervec \
  --host 0.0.0.0 \
  --port 8080
```

Use one worker for the first implementation. The engine keeps loaded indexes in
process memory.

## HTTP API

- `GET /health`
- `GET /collections`
- `GET /collections/{collection_name}/exists`
- `GET /collections/{collection_name}/describe`
- `POST /collections/{collection_name}/create`
- `DELETE /collections/{collection_name}`
- `POST /collections/{collection_name}/insert`
- `POST /collections/{collection_name}/flush`
- `POST /collections/{collection_name}/load`
- `POST /collections/{collection_name}/search`
- `POST /collections/{collection_name}/close`
