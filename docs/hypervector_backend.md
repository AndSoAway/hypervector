# HyperVector Local Backend

`HyperVectorIndexBackend` is a local, serverless replacement for the Milvus
backend shape used by retriever integrations. It stores vectors in a HyperVec
index file and scalar data in local JSON files.

## Storage Layout

Each collection is a directory under `uri`:

```text
<uri>/<collection_name>/
  manifest.json
  index.hypervec
  rows.jsonl
```

- `index.hypervec` stores the HyperVec index.
- `rows.jsonl` stores one scalar row per vector: external id, content, metadata.
- `manifest.json` stores collection configuration and index metadata.

## Minimal Config

```python
config = {
    "uri": "./kb_index",
    "collection_name": "demo",
    "metric_type": "IP",
    "index_params": {"index_type": "HNSWFlat", "M": 32},
}
```

Supported MVP index types are `Flat`, `HNSWFlat`, and `HNSWLVQ` when the Python
binding exposes `IndexHNSWLVQ`.

## API

The class implements the same high-level methods as the Milvus backend:

- `load_index(index_path=None)`
- `build_index(embeddings=..., ids=..., contents=..., metadatas=...)`
- `search(query_embeddings, top_k, **kwargs)`
- `search_with_meta(query_embeddings, top_k, **kwargs)`
- `close()`

The MVP filter supports equality clauses joined by `AND`, for example:

```python
backend.search_with_meta(xq, 5, filter="source == 'manual' and doc_id == '42'")
```

This backend does not provide server semantics, distributed storage, or Milvus
expression compatibility beyond the simple filter subset above.
