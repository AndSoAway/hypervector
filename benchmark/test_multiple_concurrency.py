#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多并发性能对比测试

目的：测试不同并发数下的系统性能表现
用于分析延迟和 QPS 如何随并发数变化
"""

import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加路径
sys.path.insert(0, '/root/vector/hypervector/pyhypervec')
from pyhypervec import HypervecClient

# 全局变量
latencies_lock = threading.Lock()

def search_one(client, query, query_id):
    """执行单次查询"""
    start = time.time()
    try:
        results = client.search(
            collection_name='wiki_hnsw_1m',
            data=[query.tolist()],
            limit=10,
            search_params={'ef_search': 128}
        )
        latency = (time.time() - start) * 1000  # 毫秒
        return {'success': True, 'latency': latency, 'query_id': query_id}
    except Exception as e:
        return {'success': False, 'error': str(e), 'query_id': query_id}

def test_concurrency(concurrent_users, num_queries=1000):
    """测试指定并发数下的性能"""
    print(f"\n{'='*80}")
    print(f"测试并发数: {concurrent_users}")
    print(f"{'='*80}")

    client = HypervecClient('http://localhost:8080', timeout=120.0)

    # 生成查询向量
    print(f"生成 {num_queries} 个查询向量...")
    queries = [np.random.randn(768).astype('float32') for _ in range(num_queries)]

    latencies = []
    errors = 0

    print(f"开始测试...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        # 提交所有查询任务
        futures = [
            executor.submit(search_one, client, queries[i % len(queries)], i)
            for i in range(num_queries)
        ]

        # 收集结果
        completed = 0
        for future in futures:
            result = future.result()
            completed += 1

            if result['success']:
                latencies.append(result['latency'])
            else:
                errors += 1

            # 每 100 个查询显示一次进度
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                current_qps = completed / elapsed if elapsed > 0 else 0
                print(f"  进度: {completed}/{num_queries} ({completed*100//num_queries}%)  |  "
                      f"当前 QPS: {current_qps:.2f}")

    total_time = time.time() - start_time

    # 计算统计
    if not latencies:
        return {
            'concurrent': concurrent_users,
            'num_queries': num_queries,
            'errors': errors,
            'success': 0
        }

    latencies_sorted = sorted(latencies)

    return {
        'concurrent': concurrent_users,
        'num_queries': num_queries,
        'success': len(latencies),
        'errors': errors,
        'total_time': total_time,
        'qps': len(latencies) / total_time,
        'avg': sum(latencies) / len(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'p50': latencies_sorted[int(len(latencies_sorted) * 0.50)],
        'p95': latencies_sorted[int(len(latencies_sorted) * 0.95)],
        'p99': latencies_sorted[int(len(latencies_sorted) * 0.99)]
    }

def main():
    print("=" * 80)
    print("HyperVector 多并发性能对比测试")
    print("=" * 80)
    print()
    print("测试目标：")
    print("  - 对比不同并发数下的性能表现")
    print("  - 分析延迟和 QPS 如何随并发数变化")
    print("  - 判断系统瓶颈类型")
    print()
    print("测试配置：")
    print("  - Collection: wiki_hnsw_1m")
    print("  - 数据规模: 1,000,000 vectors, 768 dim")
    print("  - 每轮查询数: 1,000")
    print("  - 测试并发数: 1, 5, 10, 20, 50, 100")
    print("  - ef_search: 128")
    print()

    # 检查连接
    print("检查 HyperVector Server...")
    client = HypervecClient('http://localhost:8080', timeout=60.0)

    try:
        collections = client.list_collections()
        if 'wiki_hnsw_1m' not in collections:
            print("❌ Collection 'wiki_hnsw_1m' 不存在")
            return
        print("✅ 连接成功")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    print()
    input("按 Enter 开始测试...")

    # 测试不同并发数
    concurrency_levels = [1, 5, 10, 20, 50, 100]
    results = []

    for concurrent in concurrency_levels:
        result = test_concurrency(concurrent, num_queries=1000)
        results.append(result)

        # 显示本轮结果
        if result['success'] > 0:
            print(f"\n✅ 测试完成")
            print(f"   QPS: {result['qps']:.2f}  |  "
                  f"平均: {result['avg']:.2f}ms  |  "
                  f"P50: {result['p50']:.2f}ms  |  "
                  f"P99: {result['p99']:.2f}ms")
        else:
            print(f"\n❌ 测试失败，所有查询都出错")

        # 短暂休息
        if concurrent < 100:
            print("\n等待 5 秒后进行下一轮测试...")
            time.sleep(5)

    # 输出汇总报告
    print()
    print("=" * 80)
    print("多并发性能对比报告")
    print("=" * 80)
    print()

    # 表格
    print(f"{'并发数':<8} {'QPS':<10} {'平均(ms)':<12} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10}")
    print("-" * 80)

    for r in results:
        if r['success'] > 0:
            print(f"{r['concurrent']:<8} "
                  f"{r['qps']:<10.2f} "
                  f"{r['avg']:<12.2f} "
                  f"{r['p50']:<10.2f} "
                  f"{r['p95']:<10.2f} "
                  f"{r['p99']:<10.2f}")

    print()
    print("=" * 80)
    print("诊断分析")
    print("=" * 80)
    print()

    # 分析趋势
    if len(results) >= 2 and results[0]['success'] > 0 and results[-1]['success'] > 0:
        single_avg = results[0]['avg']
        multi_avg = results[-1]['avg']
        single_qps = results[0]['qps']
        multi_qps = results[-1]['qps']

        latency_ratio = multi_avg / single_avg
        qps_ratio = multi_qps / single_qps

        print(f"【延迟变化】")
        print(f"  单并发平均延迟:   {single_avg:.2f} ms")
        print(f"  100并发平均延迟:  {multi_avg:.2f} ms")
        print(f"  延迟放大倍数:      {latency_ratio:.2f}x")
        print()

        print(f"【QPS 变化】")
        print(f"  单并发 QPS:        {single_qps:.2f}")
        print(f"  100并发 QPS:       {multi_qps:.2f}")
        print(f"  QPS 提升倍数:      {qps_ratio:.2f}x")
        print()

        # 判断瓶颈类型
        print(f"【瓶颈分析】")

        if latency_ratio > 50 and qps_ratio < 5:
            print("  ❌ 典型的并发串行化问题")
            print()
            print("  特征：")
            print(f"    - 延迟随并发数线性增长（{latency_ratio:.1f}x）")
            print(f"    - QPS 几乎不变（仅增长 {qps_ratio:.1f}x）")
            print()
            print("  结论：")
            print("    读操作被强制串行执行，100 个并发请求在排队")
            print()
            print("  原因：")
            print("    1. collection 的 RLock 让所有 search 互斥")
            print("    2. SWIG 未释放 GIL，Python 线程无法并发进入 C++")
            print("    3. HNSW 搜索被强制单线程（omp_set_num_threads(1)）")

        elif latency_ratio < 5 and qps_ratio > 20:
            print("  ✅ 并发机制良好")
            print()
            print("  特征：")
            print(f"    - 延迟基本稳定（仅增长 {latency_ratio:.1f}x）")
            print(f"    - QPS 随并发数线性增长（{qps_ratio:.1f}x）")
            print()
            print("  结论：")
            print("    系统能有效利用并发，瓶颈在单查询延迟本身")
            print()
            print("  优化方向：")
            print("    降低单查询延迟（降低 ef_search、优化回表、绕过 HTTP）")

        else:
            print("  ⚠️  混合问题")
            print()
            print("  特征：")
            print(f"    - 延迟有增长但不严重（{latency_ratio:.1f}x）")
            print(f"    - QPS 有提升但不够（{qps_ratio:.1f}x）")
            print()
            print("  结论：")
            print("    既有并发瓶颈，也有性能瓶颈")
            print()
            print("  建议：")
            print("    1. 先优化并发机制")
            print("    2. 再优化单查询性能")

    print()
    print("=" * 80)
    print("详细数据")
    print("=" * 80)
    print()

    for r in results:
        if r['success'] > 0:
            print(f"并发数 = {r['concurrent']}")
            print(f"  总查询:    {r['success']} 成功, {r['errors']} 失败")
            print(f"  总耗时:    {r['total_time']:.2f} 秒")
            print(f"  QPS:       {r['qps']:.2f}")
            print(f"  延迟:")
            print(f"    最小:    {r['min']:.2f} ms")
            print(f"    平均:    {r['avg']:.2f} ms")
            print(f"    P50:     {r['p50']:.2f} ms")
            print(f"    P95:     {r['p95']:.2f} ms")
            print(f"    P99:     {r['p99']:.2f} ms")
            print(f"    最大:    {r['max']:.2f} ms")
            print()

if __name__ == '__main__':
    main()
