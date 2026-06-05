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

Runtime data is stored under `--data-root`:

```text
<data-root>/
  collections.json
  scalar.db
  collections/
    <collection_name>/
      index.hypervec
```

Scalar fields, document text, metadata, and vectors are persisted in SQLite.
`flush()` rebuilds the HyperVec index from the stored vectors, writes the index
file atomically, and increments the collection version.

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
- `GET /collections/{collection_name}/version`
- `POST /collections/{collection_name}/sync-check`
- `GET /collections/{collection_name}/index`
- `PUT /collections/{collection_name}/index`

## curl Examples

Set the server URL:

```bash
SERVER=http://127.0.0.1:8080
```

Health check:

```bash
curl "$SERVER/health"
```

List collections:

```bash
curl "$SERVER/collections"
```

Create a collection:

```bash
curl -X POST "$SERVER/collections/demo/create" \
  -H "Content-Type: application/json" \
  -d '{
    "schema": {
      "auto_id": false,
      "enable_dynamic_field": true,
      "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": true, "max_length": 64},
        {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 2},
        {"name": "contents", "datatype": "VARCHAR", "max_length": 60000}
      ]
    },
    "index_params": {
      "indexes": [
        {
          "field_name": "vector",
          "metric_type": "L2",
          "index_type": "Flat",
          "params": {}
        }
      ]
    }
  }'
```

Insert rows:

```bash
curl -X POST "$SERVER/collections/demo/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"id": "a", "vector": [0.0, 0.0], "contents": "zero", "source": "manual"},
      {"id": "b", "vector": [1.0, 1.0], "contents": "one", "source": "manual"}
    ]
  }'
```

Build and persist the index. This increments the collection version:

```bash
curl -X POST "$SERVER/collections/demo/flush" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Load the index into server memory:

```bash
curl -X POST "$SERVER/collections/demo/load" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Search:

```bash
curl -X POST "$SERVER/collections/demo/search" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[0.1, 0.1]],
    "limit": 2,
    "output_fields": ["id", "contents", "source"],
    "filter": "source == '\''manual'\''"
  }'
```

Describe the collection:

```bash
curl "$SERVER/collections/demo/describe"
```

Check whether the collection exists:

```bash
curl "$SERVER/collections/demo/exists"
```

Close the in-memory index without deleting data:

```bash
curl -X POST "$SERVER/collections/demo/close" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Drop the collection:

```bash
curl -X DELETE "$SERVER/collections/demo"
```

## Index Sync curl Examples

Get the current collection version:

```bash
curl "$SERVER/collections/demo/version"
```

Check whether a client-side cached index is stale:

```bash
curl -X POST "$SERVER/collections/demo/sync-check" \
  -H "Content-Type: application/json" \
  -d '{
    "client_version": 1,
    "client_checksum": "sha256:old"
  }'
```

Download the current serialized HyperVec index:

```bash
curl -L "$SERVER/collections/demo/index" \
  -o demo.index.hypervec
```

The download response includes sync metadata in headers:

```text
X-Hypervec-Collection-Version
X-Hypervec-Index-Checksum
X-Hypervec-Index-Size
```

Upload a cached index back to the server:

```bash
curl -X PUT "$SERVER/collections/demo/index?version=2&checksum=sha256:<hex>" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @demo.index.hypervec
```

The server rejects an upload whose version is older than the current server
collection version.

## Index Sync

Clients can poll the collection version and download the serialized index when
the server version changes:

```python
version = client.get_version("demo")
sync = client.sync_check("demo", client_version=version["version"] - 1)
if sync["needs_sync"]:
    client.download_index("demo", "./demo.index.hypervec")
```

If a client already has a valid cached index, it can upload it back to the
server:

```python
client.upload_index(
    "demo",
    "./demo.index.hypervec",
    version=version["version"],
    checksum=version["index_checksum"],
)
```

The upload endpoint rejects an index whose version is older than the server's
current collection version. The first implementation is single-process; run the
server with one worker so in-memory index state and SQLite writes remain
consistent.
