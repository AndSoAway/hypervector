# Optional exact reranking in `hypervec_eval`

Use an approximate index to retrieve a wider candidate set, then compute exact
distances for those candidates against the original float32 base:

```sh
hypervec_eval --index compressed.index --queries query.fvecs \
  --ground-truth truth.ivecs --metric l2 --k 10 \
  --rerank-base base.fvecs --rerank-candidates 128 \
  --json-output reranked.json
```

The base must contain exactly the vectors used to build the index, **in index
label order**. The tool checks dimensions and row count and records its path,
size, and SHA-256 in the JSON report; it cannot independently prove ID order.
Keep the base file available across index save/load: raw vectors are *not*
written to compressed index files. The current helper supports squared L2 and
inner product and requires a candidate count at least as large as `k`.

This is useful for PQ, OPQ-PQ, LVQ, and IVF-RaBitQ when the approximate
distance misorders retrieved candidates. It cannot recover a true neighbor
that was never retrieved by the index. Exact-storage graph/IVF indexes and
DiskANN (which already reranks with raw nodes) do not need this additional
stage. Reading the float32 base increases process memory by approximately
`n * d * 4` bytes and exact scoring adds query work.

LVQ now uses mean-centered, per-vector scalar quantization: `nbits` (1–8)
applies **to each dimension**, and `nlocal` is no longer a valid parameter.
For example, LVQ-8 on 300 dimensions uses 308 bytes per vector (300 packed
bytes plus two float32 per-vector parameters). The old whole-vector codebook
format is incompatible; rebuild previously saved LVQ, IVF-LVQ, and HNSW-LVQ
indexes from the original vectors.
