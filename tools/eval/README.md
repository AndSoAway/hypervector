# HyperVec evaluation tools

Four executables form a file-based pipeline; every artifact is SHA-256
fingerprinted so a mismatch is detectable rather than silently scored.

| Stage | Tool | Input | Output |
| --- | --- | --- | --- |
| 1. Prepare | `hypervec_dataset prepare` | source `.fvecs` | disjoint `base.fvecs`, `queries.fvecs`, metadata |
| 1'. Import | `hypervec_convert_hdf5` | ANN `.hdf5` | the same three plus `ground-truth.ivecs` |
| 2. Ground truth | `hypervec_ground_truth` | base, queries | exact `.ivecs`, optional `--provenance` |
| 3. Build | `hypervec_build` | base | persisted index, manifest, optional `--provenance` |
| 4. Evaluate | `hypervec_eval` | index, queries, ground truth | recall, latency, QPS, memory |
Stages 1 and 1' are alternatives: `prepare` holds query rows out of the base so a
query cannot match itself; importing keeps the source's split and neighbours. Give
stage 2 and 3 `--provenance`, then pass both records to stage 4
with `--build-provenance` and `--ground-truth-provenance` to reject an index that
is not the one stage 3 built, or a ground truth computed from a base other than
the indexed one. `--sweep NAME=v1,v2,...` evaluates one runtime parameter over
several values with the index loaded once. `--concurrency` above one shares one
index across threads, so `--assume-read-only` must assert `Search` mutates
nothing; there is no read-only load mode. `--k` must not exceed the stored vector
count.

## Optional exact reranking

Retrieve a wider candidate set approximately, then score those candidates
exactly against the original float32 base:

```sh
hypervec_eval --index compressed.index --queries query.fvecs \
  --ground-truth truth.ivecs --metric l2 --k 10 \
  --rerank-base base.fvecs --rerank-candidates 128 \
  --json-output reranked.json
```

The base must contain exactly the vectors used to build the index, **in index
label order**. The tool checks dimensions and row count and records path, size
and SHA-256, but cannot independently prove ID order. Keep the base available
across index save/load: raw vectors are *not* written to compressed index files.
Only squared L2 and inner product are supported and candidates must be at least
`k`. Reranking is refused for `id_map` indexes, whose labels are external IDs
rather than base rows. Useful for PQ, OPQ-PQ, LVQ and IVF-RaBitQ when
approximate distances misorder candidates; it cannot recover a true neighbour
the index never returned. Exact-storage graph/IVF indexes and DiskANN already
hold or rerank raw vectors. The base adds roughly `n * d * 4` bytes.

`peak_rss_bytes` is a whole-process high-water mark that also covers queries,
ground truth and rerank base, so it is not comparable across algorithms;
`index_load_rss_delta_bytes` is the closer estimate of index footprint.

LVQ now uses mean-centered, per-vector scalar quantization: `nbits` (1–8)
applies **to each dimension**, and `nlocal` is no longer a valid parameter.
LVQ-8 on 300 dimensions uses 308 bytes per vector. The old whole-vector codebook
format is incompatible; rebuild previously saved LVQ, IVF-LVQ and HNSW-LVQ
indexes from the original vectors.
