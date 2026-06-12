#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperVector Wiki 1M 性能基准测试

测试指标：
1. 吞吐量（QPS）
2. 延迟分布（P50, P95, P99）
3. 不同 ef_search 参数的性能对比

使用 100 个并发用户模拟真实场景
"""

import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# 添加路径
sys.path.insert(0, '/root/vector/hypervector/pyhypervec')
sys.path.insert(0, '/root/vector/hypervector')

from pyhypervec import HypervecClient
from fbin_utils import read_fbin

# 全局变量
latencies = []
lock = threading.Lock()
errors = 0

def search_one(client, query_vector, query_id, collection_name, ef_search):
    """
    执行一次查询

    Args:
        client: HypervecClient 实例
        query_vector: 查询向量
        query_id: 查询 ID
        collection_name: Collection 名称
        ef_search: HNSW ef_search 参数

    Returns:
        dict: 查询结果
    """
    global errors

    start = time.time()
    try:
        results = client.search(
            collection_name=collection_name,
            data=[query_vector.tolist()],
            limit=10,
            search_params={'ef_search': ef_search},
            output_fields=['id']
        )
        latency = (time.time() - start) * 1000  # 转换为毫秒

        with lock:
            latencies.append(latency)

        return {
            'success': True,
            'latency': latency,
            'results': results[0] if results else []
        }
    except Exception as e:
        with lock:
            errors += 1
        return {'success': False, 'error': str(e)}

def run_throughput_test(
    client,
    queries,
    collection_name='wiki_hnsw_1m',
    num_queries=10000,
    concurrent_users=100,
    ef_search=200
):
    """
    运行吞吐量测试

    Args:
        client: HypervecClient 实例
        queries: 查询向量数组
        collection_name: Collection 名称
        num_queries: 总查询数
        concurrent_users: 并发用户数
        ef_search: HNSW ef_search 参数

    Returns:
        dict: 测试结果统计
    """
    global latencies, errors
    latencies = []
    errors = 0

    print(f"\n开始吞吐量测试（ef_search={ef_search}）...")
    print(f"  总查询数: {num_queries:,}")
    print(f"  并发用户: {concurrent_users}")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        # 提交所有查询任务
        futures = [
            executor.submit(
                search_one,
                client,
                queries[i % len(queries)],  # 循环使用查询向量
                i,
                collection_name,
                ef_search
            )
            for i in range(num_queries)
        ]

        # 等待完成（带进度显示）
        completed = 0
        for future in futures:
            future.result()
            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - start_time
                qps_current = completed / elapsed
                print(f"  进度: {completed}/{num_queries} ({completed*100//num_queries}%) - 当前 QPS: {qps_current:.0f}")

    end_time = time.time()
    total_time = end_time - start_time

    # 计算统计指标
    if len(latencies) == 0:
        return None

    qps = len(latencies) / total_time
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    avg = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    return {
        'ef_search': ef_search,
        'num_queries': len(latencies),
        'errors': errors,
        'total_time': total_time,
        'qps': qps,
        'latency_avg': avg,
        'latency_min': min_lat,
        'latency_max': max_lat,
        'latency_p50': p50,
        'latency_p95': p95,
        'latency_p99': p99
    }

def print_report(results_list):
    """打印测试报告"""
    print("\n" + "=" * 80)
    print("HyperVector Wiki 1M 性能测试报告")
    print("=" * 80)
    print()
    print("数据集信息:")
    print("  数据集: Wiki 1M (1,000,000 vectors, 768 dimensions)")
    print("  索引类型: HNSW (M=32, ef_construction=200)")
    print()
    print("测试配置:")
    print(f"  并发用户数: {results_list[0]['num_queries'] // 100}")
    print(f"  每轮查询数: {results_list[0]['num_queries']:,}")
    print()
    print("-" * 80)
    print(f"{'ef_search':<12} {'QPS':<10} {'平均(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10}")
    print("-" * 80)

    for result in results_list:
        print(
            f"{result['ef_search']:<12} "
            f"{result['qps']:<10.2f} "
            f"{result['latency_avg']:<10.2f} "
            f"{result['latency_p50']:<10.2f} "
            f"{result['latency_p95']:<10.2f} "
            f"{result['latency_p99']:<10.2f}"
        )

    print("-" * 80)
    print()

    # 详细结果
    print("详细结果:")
    print()
    for result in results_list:
        print(f"【ef_search = {result['ef_search']}】")
        print(f"  吞吐量 (QPS):   {result['qps']:.2f} 查询/秒")
        print(f"  总耗时:         {result['total_time']:.2f} 秒")
        print(f"  成功查询:       {result['num_queries']:,}")
        print(f"  失败查询:       {result['errors']}")
        print(f"  延迟统计:")
        print(f"    最小:         {result['latency_min']:.2f} ms")
        print(f"    平均:         {result['latency_avg']:.2f} ms")
        print(f"    P50:          {result['latency_p50']:.2f} ms")
        print(f"    P95:          {result['latency_p95']:.2f} ms")
        print(f"    P99:          {result['latency_p99']:.2f} ms")
        print(f"    最大:         {result['latency_max']:.2f} ms")
        print()

    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='HyperVector Wiki 1M 性能测试')
    parser.add_argument(
        '--queries-file',
        default='/data/ljx/test_fbin/wiki_all_1M/query.public.10K.fbin',
        help='查询向量文件路径'
    )
    parser.add_argument(
        '--collection',
        default='wiki_hnsw_1m',
        help='Collection 名称'
    )
    parser.add_argument(
        '--num-queries',
        type=int,
        default=10000,
        help='总查询数'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=100,
        help='并发用户数'
    )
    parser.add_argument(
        '--ef-search',
        type=str,
        default='128,256,512',
        help='ef_search 参数列表，逗号分隔'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HyperVector Wiki 1M 性能基准测试")
    print("=" * 80)
    print()

    # ========================================
    # 步骤 1：读取查询向量
    # ========================================
    print("步骤 1/3：读取查询向量")
    print("-" * 80)
    try:
        queries, nq, dq = read_fbin(args.queries_file)
        print(f"✅ 查询向量加载成功：{nq:,} 条")
    except FileNotFoundError:
        print(f"❌ 查询文件不存在：{args.queries_file}")
        print("提示：请确认查询文件路径是否正确")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取失败：{e}")
        sys.exit(1)

    print()

    # ========================================
    # 步骤 2：连接 HyperVector
    # ========================================
    print("步骤 2/3：连接 HyperVector Server")
    print("-" * 80)
    try:
        client = HypervecClient('http://localhost:8080')
        collections = client.list_collections()
        print(f"✅ 连接成功")

        if args.collection not in collections:
            print(f"❌ Collection '{args.collection}' 不存在")
            print(f"可用的 Collections: {collections}")
            print("提示：请先运行 import_wiki_1m.py 导入数据")
            sys.exit(1)

        print(f"✅ Collection '{args.collection}' 存在")
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        sys.exit(1)

    print()

    # ========================================
    # 步骤 3：运行性能测试
    # ========================================
    print("步骤 3/3：运行性能测试")
    print("-" * 80)

    # 解析 ef_search 参数
    ef_search_list = [int(x.strip()) for x in args.ef_search.split(',')]
    print(f"测试参数: ef_search = {ef_search_list}")
    print()

    results_list = []

    for ef_search in ef_search_list:
        result = run_throughput_test(
            client=client,
            queries=queries,
            collection_name=args.collection,
            num_queries=min(args.num_queries, nq),
            concurrent_users=args.concurrent,
            ef_search=ef_search
        )

        if result:
            results_list.append(result)
            print(f"✅ ef_search={ef_search} 测试完成 - QPS: {result['qps']:.2f}, P99: {result['latency_p99']:.2f}ms")
        else:
            print(f"❌ ef_search={ef_search} 测试失败")

    # ========================================
    # 输出报告
    # ========================================
    if results_list:
        print_report(results_list)
    else:
        print("\n❌ 所有测试失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
