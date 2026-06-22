#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperVector 性能测试 - 100线程并发（支持多个 ef_search，带召回率计算）
"""

import argparse
import sys
import time
import struct
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, '/root/vector/hypervector/pyhypervec')
from pyhypervec import HypervecClient


def load_fbin(filename):
    """加载 .fbin 文件"""
    with open(filename, 'rb') as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        vectors = data.reshape(num_vectors, dim)
    return vectors


def load_ibin_neighbors(filename, max_read=None):
    """加载 .ibin 格式的 groundtruth 邻居索引文件"""
    try:
        with open(filename, 'rb') as f:
            nqueries, nneighbors = struct.unpack('ii', f.read(8))
            if max_read is None:
                max_read = nqueries
            else:
                max_read = min(max_read, nqueries)

            data = np.fromfile(f, dtype=np.int32, count=max_read * nneighbors)
            data = data.reshape(max_read, nneighbors)
            return data.tolist(), nneighbors
    except Exception as e:
        print(f"加载 groundtruth 文件失败: {e}")
        return None, 0


def parse_vector_id(raw_id):
    """将服务端返回的主键 ID 转成 groundtruth 中的 base 向量下标"""
    try:
        if isinstance(raw_id, str) and '_' in raw_id:
            return int(raw_id.rsplit('_', 1)[-1])
        return int(raw_id)
    except (TypeError, ValueError):
        return None


def calculate_recall(retrieved_ids, groundtruth_ids, k_values=[1, 5, 10]):
    """计算召回率 Recall@K"""
    recalls = {}

    for k in k_values:
        denom = min(k, len(groundtruth_ids))
        if denom == 0:
            recalls[k] = 0.0
            continue

        retrieved_set = set(retrieved_ids[:k])
        gt_set = set(groundtruth_ids[:k])
        correct = len(retrieved_set & gt_set)
        recalls[k] = correct / denom

    return recalls


def worker(client, collection_name, query_vectors, query_indices, ef_search, k=10):
    """单个线程执行多次查询"""
    latencies = []
    results_data = []

    for query_idx in query_indices:
        query_idx = query_idx % len(query_vectors)
        query = query_vectors[query_idx].tolist()

        start = time.time()
        try:
            result = client.search(
                collection_name=collection_name,
                data=[query],
                limit=k,
                search_params={'ef_search': ef_search},
                output_fields=['id']
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            retrieved_ids = []
            if result and isinstance(result, list) and len(result) > 0:
                for item in result[0]:
                    raw_id = item.get('id') or item.get('entity', {}).get('id')
                    vector_id = parse_vector_id(raw_id)
                    if vector_id is not None:
                        retrieved_ids.append(vector_id)

            results_data.append({
                'query_idx': query_idx,
                'retrieved_ids': retrieved_ids,
                'latency': latency
            })

        except Exception as e:
            print(f"查询错误: {e}")

    return latencies, results_data


def calculate_recall_stats(all_results, groundtruth, k, recall_k_values):
    if not groundtruth or not all_results:
        return None

    all_recalls = {recall_k: [] for recall_k in recall_k_values}
    valid_count = 0

    for result_item in all_results:
        query_idx = result_item['query_idx']
        retrieved_ids = result_item['retrieved_ids']

        if query_idx < len(groundtruth) and len(retrieved_ids) > 0:
            gt_ids = groundtruth[query_idx][:k]
            recalls = calculate_recall(retrieved_ids, gt_ids, k_values=recall_k_values)

            for recall_k in recall_k_values:
                all_recalls[recall_k].append(recalls.get(recall_k, 0.0))
            valid_count += 1

    if valid_count == 0:
        return None

    return {
        'valid_count': valid_count,
        'recalls': {
            recall_k: float(np.mean(all_recalls[recall_k])) if all_recalls[recall_k] else 0.0
            for recall_k in recall_k_values
        }
    }


def run_benchmark_for_ef(client, collection_name, queries, groundtruth, ef_search,
                         num_workers, total_queries, queries_per_worker, k, recall_k_values):
    print()
    print("=" * 70)
    print(f"开始测试 ef_search={ef_search}")
    print("=" * 70)

    start_time = time.time()
    all_latencies = []
    all_results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        query_splits = np.array_split(np.arange(total_queries), num_workers)
        for query_indices in query_splits:
            future = executor.submit(
                worker,
                client,
                collection_name,
                queries,
                query_indices.tolist(),
                ef_search,
                k
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=num_workers, desc=f"ef={ef_search} 线程完成"):
            latencies, results = future.result()
            all_latencies.extend(latencies)
            all_results.extend(results)

    total_time = time.time() - start_time
    success_rate = len(all_latencies) / total_queries if total_queries else 0.0

    if not all_latencies:
        print("没有成功的查询")
        return {
            'ef_search': ef_search,
            'qps': 0.0,
            'success_rate': 0.0,
            'recall_stats': None,
        }

    latencies_np = np.array(all_latencies)
    qps = len(all_latencies) / total_time
    avg_latency = float(np.mean(latencies_np))
    p50_latency = float(np.percentile(latencies_np, 50))
    p95_latency = float(np.percentile(latencies_np, 95))
    p99_latency = float(np.percentile(latencies_np, 99))
    min_latency = float(np.min(latencies_np))
    max_latency = float(np.max(latencies_np))
    recall_stats = calculate_recall_stats(all_results, groundtruth, k, recall_k_values)

    print()
    print(f"【ef_search={ef_search} 测试结果】")
    print()
    print("【吞吐量】")
    print(f"  QPS: {qps:.2f} 查询/秒")
    print()
    print("【延迟】")
    print(f"  平均延迟: {avg_latency:.2f} ms")
    print(f"  P50 延迟: {p50_latency:.2f} ms")
    print(f"  P95 延迟: {p95_latency:.2f} ms")
    print(f"  P99 延迟: {p99_latency:.2f} ms")
    print(f"  最小延迟: {min_latency:.2f} ms")
    print(f"  最大延迟: {max_latency:.2f} ms")
    print()
    print("【统计】")
    print(f"  总查询数: {len(all_latencies):,}/{total_queries:,}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  成功率: {success_rate:.2%}")
    print()

    if recall_stats:
        print("【召回率 (Recall@K)】")
        print(f"  有效召回率计算: {recall_stats['valid_count']}/{len(all_results)}")
        for recall_k in recall_k_values:
            print(f"  Recall@{recall_k}: {recall_stats['recalls'][recall_k]:.4f}")
        print()
    else:
        print("【召回率 (Recall@K)】")
        print("  无法计算召回率")
        print()

    return {
        'ef_search': ef_search,
        'qps': qps,
        'avg_latency': avg_latency,
        'p50_latency': p50_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'total_queries': len(all_latencies),
        'total_time': total_time,
        'success_rate': success_rate,
        'recall_stats': recall_stats,
    }


def parse_ef_values(raw_efs):
    return [int(item.strip()) for item in raw_efs.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser(description='HyperVector 多 ef_search 并发性能与召回率测试')
    parser.add_argument('--host', default='http://localhost:8080')
    parser.add_argument('--collection', default='wiki_hnsw_1m')
    parser.add_argument('--queries', default='/data/ljx/test_fbin/wiki_all_1M/queries.fbin')
    parser.add_argument('--groundtruth', default='/data/ljx/test_fbin/wiki_all_1M/groundtruth.1M.neighbors.ibin')
    parser.add_argument('--efs', default='64,128,256,512', help='逗号分隔的 ef_search 列表，例如 64,128,256,512')
    parser.add_argument('--workers', type=int, default=100)
    parser.add_argument('--total-queries', type=int, default=10000)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    ef_values = parse_ef_values(args.efs)
    if not ef_values:
        raise ValueError('--efs 不能为空')

    queries_per_worker = args.total_queries // args.workers
    recall_k_values = [1, 5, 10]

    print("=" * 70)
    print("HyperVector 性能测试 - 多 ef_search 并发测试")
    print("=" * 70)
    print()
    print("配置:")
    print(f"  - 服务地址: {args.host}")
    print(f"  - Collection: {args.collection}")
    print(f"  - 并发线程数: {args.workers}")
    print(f"  - 总查询数/每个 ef: {args.total_queries:,}")
    print(f"  - 每线程查询数: {queries_per_worker}")
    print(f"  - Top-K: {args.top_k}")
    print(f"  - ef_search 列表: {ef_values}")
    print()

    print("加载查询向量...")
    queries = load_fbin(args.queries)
    print(f"  查询向量: {queries.shape}")

    groundtruth = None
    print("加载 groundtruth 数据（用于计算召回率）...")
    try:
        groundtruth, nneighbors = load_ibin_neighbors(args.groundtruth, max_read=len(queries))
        if groundtruth:
            print(f"  Groundtruth: {len(groundtruth)} 个查询, 每个 {nneighbors} 个邻居")
        else:
            print("  Groundtruth 加载失败，将跳过召回率计算")
    except Exception as e:
        print(f"  无法加载 groundtruth: {e}")
        print("  将跳过召回率计算")

    client = HypervecClient(args.host)
    summaries = []

    for ef_search in ef_values:
        summary = run_benchmark_for_ef(
            client=client,
            collection_name=args.collection,
            queries=queries,
            groundtruth=groundtruth,
            ef_search=ef_search,
            num_workers=args.workers,
            total_queries=args.total_queries,
            queries_per_worker=queries_per_worker,
            k=args.top_k,
            recall_k_values=recall_k_values,
        )
        summaries.append(summary)

    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    header = (
        f"{'ef':>8} {'QPS':>10} {'Avg(ms)':>10} {'P50(ms)':>10} "
        f"{'P95(ms)':>10} {'P99(ms)':>10} {'R@1':>8} {'R@5':>8} {'R@10':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        recalls = item.get('recall_stats', {}).get('recalls', {}) if item.get('recall_stats') else {}
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


if __name__ == '__main__':
    main()
