#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and benchmark the six official 10K indexes for the small Wiki test set."""

import argparse
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/root/vector/hypervector/pyhypervec")
from pyhypervec import HypervecClient


INDEX_SPECS = [
    ("ivfflat", "wiki_ivfflat_10k", "IVFFlat", {"nlist": 1024}),
    ("ivflvq", "wiki_ivflvq_10k", "IVFLVQ", {"nlist": 1024, "nlocal": 16, "nbits": 8}),
    ("ivfpq", "wiki_ivfpq_10k", "IVFPQ", {"nlist": 1024, "m_pq": 8, "nbits": 8}),
    ("hnswflat", "wiki_hnswflat_10k", "HNSWFlat", {"M": 32, "ef_construction": 200}),
    ("hnswlvq", "wiki_hnswlvq_10k", "HNSWLVQ", {"M": 32, "nlocal": 16, "nbits": 10}),
    ("hnswpq", "wiki_hnswpq_10k", "HNSWPQ", {"M": 32, "m": 256, "nbits": 8}),
]
def load_fbin(filename):
    with open(filename, "rb") as f:
        num_vectors, dim = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(num_vectors, dim)


def load_ibin_neighbors(filename, max_read=None):
    with open(filename, "rb") as f:
        nqueries, nneighbors = struct.unpack("ii", f.read(8))
        if max_read is None:
            max_read = nqueries
        else:
            max_read = min(max_read, nqueries)
        data = np.fromfile(f, dtype=np.int32, count=max_read * nneighbors)
    return data.reshape(max_read, nneighbors).tolist(), nneighbors


def parse_vector_id(raw_id):
    try:
        if isinstance(raw_id, str) and "_" in raw_id:
            return int(raw_id.rsplit("_", 1)[-1])
        return int(raw_id)
    except (TypeError, ValueError):
        return None


def parse_index_names(raw):
    aliases = {name: spec for name, *_rest in INDEX_SPECS for spec in [name]}
    if raw.strip().lower() == "all":
        return [spec[0] for spec in INDEX_SPECS]
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in aliases]
    if unknown:
        raise ValueError(f"unsupported index names: {unknown}; supported: {[spec[0] for spec in INDEX_SPECS]}")
    return names


def make_schema_and_index(client, dim, index_type, params):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype="VARCHAR", is_primary=True, max_length=128)
    schema.add_field(field_name="vector", datatype="FLOAT_VECTOR", dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=index_type,
        metric_type="L2",
        params=params,
    )
    return schema, index_params


def flush_collection(client, collection_name):
    start = time.time()
    result = client.flush(collection_name=collection_name)
    print(
        f"flush/index build 完成: {collection_name}, "
        f"total={result.get('total', result.get('n_total', 0))}, "
        f"耗时 {time.time() - start:.2f}s, "
        f"size={result.get('index_size_bytes', 0) / 1024 / 1024:.2f} MB"
    )


def ensure_collection(client, vectors, spec, drop_existing, batch_size, do_insert=True, do_flush=True):
    _name, collection_name, index_type, params = spec
    existing = set(client.list_collections())
    if collection_name in existing:
        if not drop_existing:
            print(f"跳过 collection: {collection_name} 已存在")
            return
        print(f"删除旧 collection: {collection_name}")
        client.drop_collection(collection_name)

    print("-" * 80)
    print(f"创建 collection {collection_name}: index_type={index_type}, params={params}")
    schema, index_params = make_schema_and_index(client, vectors.shape[1], index_type, params)
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    if not do_insert:
        print(f"仅创建 collection，跳过导入: {collection_name}")
        return

    start = time.time()
    inserted = 0
    for begin in tqdm(range(0, len(vectors), batch_size), desc=f"insert {collection_name}", unit="batch"):
        end = min(begin + batch_size, len(vectors))
        batch = [
            {"id": f"wiki_{idx}", "vector": vectors[idx].tolist()}
            for idx in range(begin, end)
        ]
        client.insert(collection_name=collection_name, data=batch)
        inserted += len(batch)
    print(f"插入完成: {inserted} 条，耗时 {time.time() - start:.2f}s")

    if do_flush:
        flush_collection(client, collection_name)
    else:
        print(f"导入完成，跳过 flush/index build: {collection_name}")


def calculate_recall(retrieved_ids, groundtruth_ids, k_values):
    recalls = {}
    for k in k_values:
        denom = min(k, len(groundtruth_ids))
        recalls[k] = len(set(retrieved_ids[:k]) & set(groundtruth_ids[:k])) / denom if denom else 0.0
    return recalls


def worker(host, collection_name, query_vectors, query_indices, search_params, k):
    client = HypervecClient(host)
    normalized_indices = [int(query_idx) % len(query_vectors) for query_idx in query_indices]
    if not normalized_indices:
        return [], []
    if normalized_indices == list(range(normalized_indices[0], normalized_indices[-1] + 1)):
        query_batch = query_vectors[normalized_indices[0]:normalized_indices[-1] + 1]
    else:
        query_batch = query_vectors[normalized_indices]

    start = time.time()
    result = client.search(
        collection_name=collection_name,
        data=query_batch.tolist(),
        limit=k,
        search_params=search_params,
        output_fields=["id"],
    )
    per_query_latency = ((time.time() - start) * 1000) / len(normalized_indices)
    rows = []
    for query_idx, row in zip(normalized_indices, result):
        retrieved_ids = []
        for item in row:
            raw_id = item.get("id") or item.get("entity", {}).get("id")
            vector_id = parse_vector_id(raw_id)
            if vector_id is not None:
                retrieved_ids.append(vector_id)
        rows.append({"query_idx": query_idx, "retrieved_ids": retrieved_ids})
    return [per_query_latency] * len(normalized_indices), rows


def benchmark_collection(host, collection_name, queries, groundtruth, workers, total_queries, top_k, search_params):
    start_time = time.time()
    all_latencies = []
    all_results = []
    query_splits = np.array_split(np.arange(total_queries), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(worker, host, collection_name, queries, query_indices.tolist(), search_params, top_k)
            for query_indices in query_splits
        ]
        for future in tqdm(as_completed(futures), total=workers, desc=f"search {collection_name}"):
            latencies, results = future.result()
            all_latencies.extend(latencies)
            all_results.extend(results)

    total_time = time.time() - start_time
    if not all_latencies:
        return {"collection": collection_name, "qps": 0.0, "success_rate": 0.0, "recalls": {}}

    recall_values = {1: [], 5: [], 10: []}
    valid = 0
    for item in all_results:
        query_idx = item["query_idx"]
        retrieved_ids = item["retrieved_ids"]
        if query_idx >= len(groundtruth) or not retrieved_ids:
            continue
        recalls = calculate_recall(retrieved_ids, groundtruth[query_idx][:top_k], recall_values.keys())
        for k, value in recalls.items():
            recall_values[k].append(value)
        valid += 1

    latencies_np = np.array(all_latencies)
    return {
        "collection": collection_name,
        "qps": len(all_latencies) / total_time,
        "avg_latency": float(np.mean(latencies_np)),
        "p50_latency": float(np.percentile(latencies_np, 50)),
        "p95_latency": float(np.percentile(latencies_np, 95)),
        "p99_latency": float(np.percentile(latencies_np, 99)),
        "total_queries": len(all_latencies),
        "success_rate": len(all_latencies) / total_queries,
        "valid_recall": valid,
        "recalls": {k: float(np.mean(v)) if v else 0.0 for k, v in recall_values.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="HyperVector 10K/100 queries 六索引小规模测试")
    parser.add_argument("--host", default="tcp://localhost:50052")
    parser.add_argument("--base", default="/data/hypervec_data/wiki_10K/base.10K.fbin")
    parser.add_argument("--queries", default="/data/hypervec_data/wiki_10K/queries.100.fbin")
    parser.add_argument("--groundtruth", default="/data/hypervec_data/wiki_10K/groundtruth.10K.neighbors.ibin")
    parser.add_argument("--indexes", default="ivfflat,ivflvq,ivfpq,hnswflat,hnswlvq,hnswpq")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--total-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="只 benchmark，不创建/导入 collection")
    parser.add_argument("--build-only", action="store_true", help="只创建/导入 collection，不执行 benchmark")
    parser.add_argument("--import-only", action="store_true", help="删除现有 collection 后只导入数据")
    parser.add_argument("--build-index-only", action="store_true", help="删除现有 collection 后只创建 collection，触发索引构建")
    args = parser.parse_args()

    selected = set(parse_index_names(args.indexes))
    specs = [spec for spec in INDEX_SPECS if spec[0] in selected]
    vectors = load_fbin(args.base)
    queries = load_fbin(args.queries)
    groundtruth, nneighbors = load_ibin_neighbors(args.groundtruth, max_read=args.total_queries)
    if args.total_queries > len(queries):
        raise ValueError(f"total-queries {args.total_queries} 超过 query 文件数量 {len(queries)}")

    print("=" * 100)
    print("HyperVector 10K/100 queries 六索引测试")
    print("=" * 100)
    print(f"base: {vectors.shape}")
    print(f"queries: {queries.shape}")
    print(f"groundtruth: {len(groundtruth)} x {nneighbors}")
    print(f"indexes: {[spec[0] for spec in specs]}")
    print(f"workers={args.workers}, total_queries={args.total_queries}, top_k={args.top_k}")
    print(f"ef_search={args.ef_search}, nprobe={args.nprobe}")

    client = HypervecClient(args.host, timeout=2400.0)
    if args.import_only:
        for spec in specs:
            ensure_collection(client, vectors, spec, True, args.batch_size, do_insert=True, do_flush=False)
        print("import-only 完成：已删除旧 collection 并完成数据导入，未执行 flush/index build")
        return

    if args.build_index_only:
        existing = set(client.list_collections())
        for _name, collection_name, _index_type, _params in specs:
            if collection_name not in existing:
                raise RuntimeError(f"collection 不存在，无法构建索引，请先执行 --import-only: {collection_name}")
            flush_collection(client, collection_name)
        print("build-index-only 完成：已对已有 collection 执行 flush/index build")
        return

    if not args.skip_build:
        for spec in specs:
            ensure_collection(client, vectors, spec, args.drop_existing, args.batch_size)

    if args.build_only:
        print("build-only 完成，跳过 benchmark")
        return

    summaries = []
    for name, collection_name, index_type, _params in specs:
        search_params = {}
        if name in {"ivfflat", "ivflvq", "ivfpq"}:
            search_params["nprobe"] = args.nprobe
        if name in {"hnswflat", "hnswlvq", "hnswpq"}:
            search_params["ef_search"] = args.ef_search
        print()
        print("=" * 80)
        print(f"测试 {collection_name} ({index_type}), search_params={search_params}")
        print("=" * 80)
        try:
            summary = benchmark_collection(
                args.host,
                collection_name,
                queries,
                groundtruth,
                args.workers,
                args.total_queries,
                args.top_k,
                search_params,
            )
            summary.update({
                "index": name,
                "index_type": index_type,
                "ef_search": search_params.get("ef_search"),
                "nprobe": search_params.get("nprobe"),
            })
        except Exception as exc:
            print(f"测试失败: {collection_name}: {exc}")
            summary = {
                "index": name,
                "index_type": index_type,
                "collection": collection_name,
                "ef_search": search_params.get("ef_search"),
                "nprobe": search_params.get("nprobe"),
                "error": str(exc),
            }
        summaries.append(summary)

    print()
    print("=" * 120)
    print("六索引汇总")
    print("=" * 120)
    header = f"{'index':>10} {'collection':>18} {'nprobe':>8} {'ef':>6} {'QPS':>10} {'Avg':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'status':>10}"
    print(header)
    print("-" * len(header))
    for item in summaries:
        ef_label = "-" if item.get("ef_search") is None else str(item.get("ef_search"))
        nprobe_label = "-" if item.get("nprobe") is None else str(item.get("nprobe"))
        if "error" in item:
            print(f"{item['index']:>10} {item['collection']:>18} {nprobe_label:>8} {ef_label:>6} {'-':>10} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'FAILED':>10}")
            print(f"  error: {item['error']}")
            continue
        recalls = item.get("recalls", {})
        status = "OK" if recalls.get(10, 0.0) >= 0.90 else "LOW_RECALL"
        print(
            f"{item['index']:>10} {item['collection']:>18} {nprobe_label:>8} {ef_label:>6} "
            f"{item.get('qps', 0.0):>10.2f} "
            f"{item.get('avg_latency', 0.0):>8.2f} "
            f"{recalls.get(1, 0.0):>8.4f} "
            f"{recalls.get(5, 0.0):>8.4f} "
            f"{recalls.get(10, 0.0):>8.4f} "
            f"{status:>10}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
