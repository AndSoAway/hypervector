#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperVector 小规模性能测试 - 10K base / 100 queries / 精确 groundtruth。
"""

import argparse
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/root/vector/hypervector/pyhypervec")
from pyhypervec import HypervecClient


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


def calculate_recall(retrieved_ids, groundtruth_ids, k_values):
    recalls = {}
    for k in k_values:
        denom = min(k, len(groundtruth_ids))
        if denom == 0:
            recalls[k] = 0.0
            continue
        recalls[k] = len(set(retrieved_ids[:k]) & set(groundtruth_ids[:k])) / denom
    return recalls


def worker(host, collection_name, query_vectors, query_indices, ef_search, k):
    client = HypervecClient(host)
    normalized_indices = [int(query_idx) % len(query_vectors) for query_idx in query_indices]
    if not normalized_indices:
        return [], []

    if normalized_indices == list(range(normalized_indices[0], normalized_indices[-1] + 1)):
        query_batch = query_vectors[normalized_indices[0]:normalized_indices[-1] + 1]
    else:
        query_batch = query_vectors[normalized_indices]

    start = time.time()
    try:
        result = client.search(
            collection_name=collection_name,
            data=query_batch,
            limit=k,
            search_params={"ef_search": ef_search},
            output_fields=["id"],
        )
    except Exception as exc:
        print(f"查询错误: {exc}")
        return [], []

    batch_latency = (time.time() - start) * 1000
    per_query_latency = batch_latency / len(normalized_indices)
    latencies = [per_query_latency] * len(normalized_indices)
    results_data = []

    for query_idx, row in zip(normalized_indices, result):
        retrieved_ids = []
        for item in row:
            raw_id = item.get("id") or item.get("entity", {}).get("id")
            vector_id = parse_vector_id(raw_id)
            if vector_id is not None:
                retrieved_ids.append(vector_id)
        results_data.append({"query_idx": query_idx, "retrieved_ids": retrieved_ids, "latency": per_query_latency})

    return latencies, results_data


def calculate_recall_stats(all_results, groundtruth, k, recall_k_values):
    all_recalls = {recall_k: [] for recall_k in recall_k_values}
    valid_count = 0

    for result_item in all_results:
        query_idx = result_item["query_idx"]
        retrieved_ids = result_item["retrieved_ids"]
        if query_idx >= len(groundtruth) or not retrieved_ids:
            continue
        recalls = calculate_recall(retrieved_ids, groundtruth[query_idx][:k], recall_k_values)
        for recall_k in recall_k_values:
            all_recalls[recall_k].append(recalls.get(recall_k, 0.0))
        valid_count += 1

    if valid_count == 0:
        return None
    return {
        "valid_count": valid_count,
        "recalls": {
            recall_k: float(np.mean(all_recalls[recall_k])) if all_recalls[recall_k] else 0.0
            for recall_k in recall_k_values
        },
    }


def run_benchmark_for_ef(host, collection_name, queries, groundtruth, ef_search, workers, total_queries, k, recall_k_values):
    print()
    print("=" * 70)
    print(f"开始测试 ef_search={ef_search}")
    print("=" * 70)

    start_time = time.time()
    all_latencies = []
    all_results = []

    query_splits = np.array_split(np.arange(total_queries), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(worker, host, collection_name, queries, query_indices.tolist(), ef_search, k)
            for query_indices in query_splits
        ]
        for future in tqdm(as_completed(futures), total=workers, desc=f"ef={ef_search} worker 完成"):
            latencies, results = future.result()
            all_latencies.extend(latencies)
            all_results.extend(results)

    total_time = time.time() - start_time
    success_rate = len(all_latencies) / total_queries if total_queries else 0.0
    if not all_latencies:
        print("没有成功的查询")
        return {"ef_search": ef_search, "qps": 0.0, "success_rate": 0.0, "recall_stats": None}

    latencies_np = np.array(all_latencies)
    recall_stats = calculate_recall_stats(all_results, groundtruth, k, recall_k_values)
    summary = {
        "ef_search": ef_search,
        "qps": len(all_latencies) / total_time,
        "avg_latency": float(np.mean(latencies_np)),
        "p50_latency": float(np.percentile(latencies_np, 50)),
        "p95_latency": float(np.percentile(latencies_np, 95)),
        "p99_latency": float(np.percentile(latencies_np, 99)),
        "min_latency": float(np.min(latencies_np)),
        "max_latency": float(np.max(latencies_np)),
        "total_queries": len(all_latencies),
        "total_time": total_time,
        "success_rate": success_rate,
        "recall_stats": recall_stats,
    }

    print()
    print(f"【ef_search={ef_search} 测试结果】")
    print("【吞吐量】")
    print(f"  QPS: {summary['qps']:.2f} 查询/秒")
    print("【延迟】")
    print(f"  平均延迟: {summary['avg_latency']:.2f} ms")
    print(f"  P50 延迟: {summary['p50_latency']:.2f} ms")
    print(f"  P95 延迟: {summary['p95_latency']:.2f} ms")
    print(f"  P99 延迟: {summary['p99_latency']:.2f} ms")
    print(f"  最小延迟: {summary['min_latency']:.2f} ms")
    print(f"  最大延迟: {summary['max_latency']:.2f} ms")
    print("【统计】")
    print(f"  总查询数: {len(all_latencies):,}/{total_queries:,}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  成功率: {success_rate:.2%}")

    print("【召回率 (Recall@K)】")
    if recall_stats:
        print(f"  有效召回率计算: {recall_stats['valid_count']}/{len(all_results)}")
        for recall_k in recall_k_values:
            print(f"  Recall@{recall_k}: {recall_stats['recalls'][recall_k]:.4f}")
    else:
        print("  无法计算召回率")

    return summary


def parse_ef_values(raw_efs):
    return [int(item.strip()) for item in raw_efs.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="HyperVector 10K/100 queries 小规模性能与召回率测试")
    parser.add_argument("--host", default="tcp://localhost:50052")
    parser.add_argument("--collection", default="wiki_hnsw_10k")
    parser.add_argument("--queries", default="/root/vector/data/wiki_10K/queries.100.fbin")
    parser.add_argument("--groundtruth", default="/root/vector/data/wiki_10K/groundtruth.10K.neighbors.ibin")
    parser.add_argument("--efs", default="32,64,128", help="逗号分隔的 ef_search 列表")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--total-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers 必须大于 0")
    if args.total_queries <= 0:
        raise ValueError("--total-queries 必须大于 0")

    queries = load_fbin(args.queries)
    if args.total_queries > len(queries):
        raise ValueError(f"total-queries {args.total_queries} 超过 query 文件数量 {len(queries)}")

    groundtruth, nneighbors = load_ibin_neighbors(args.groundtruth, max_read=args.total_queries)
    recall_k_values = [1, 5, 10]
    ef_values = parse_ef_values(args.efs)
    if not ef_values:
        raise ValueError("--efs 不能为空")

    print("=" * 70)
    print("HyperVector 小规模测试 - 10K base / 100 queries")
    print("=" * 70)
    print("配置:")
    print(f"  - 服务地址: {args.host}")
    print(f"  - Collection: {args.collection}")
    print(f"  - 查询向量: {queries.shape}")
    print(f"  - Groundtruth: {len(groundtruth)} 个查询, 每个 {nneighbors} 个邻居")
    print(f"  - 并发 worker: {args.workers}")
    print(f"  - 总查询数/每个 ef: {args.total_queries}")
    print(f"  - 每 worker 查询数: {args.total_queries / args.workers:.2f}")
    print(f"  - Top-K: {args.top_k}")
    print(f"  - ef_search 列表: {ef_values}")

    summaries = []
    for ef_search in ef_values:
        summaries.append(
            run_benchmark_for_ef(
                host=args.host,
                collection_name=args.collection,
                queries=queries,
                groundtruth=groundtruth,
                ef_search=ef_search,
                workers=args.workers,
                total_queries=args.total_queries,
                k=args.top_k,
                recall_k_values=recall_k_values,
            )
        )

    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    header = f"{'ef':>8} {'QPS':>10} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10} {'R@1':>8} {'R@5':>8} {'R@10':>8}"
    print(header)
    print("-" * len(header))
    for item in summaries:
        recalls = item.get("recall_stats", {}).get("recalls", {}) if item.get("recall_stats") else {}
        print(
            f"{item['ef_search']:>8} "
            f"{item.get('qps', 0.0):>10.2f} "
            f"{item.get('avg_latency', 0.0):>10.2f} "
            f"{item.get('p50_latency', 0.0):>10.2f} "
            f"{item.get('p95_latency', 0.0):>10.2f} "
            f"{item.get('p99_latency', 0.0):>10.2f} "
            f"{recalls.get(1, 0.0):>8.4f} "
            f"{recalls.get(5, 0.0):>8.4f} "
            f"{recalls.get(10, 0.0):>8.4f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
