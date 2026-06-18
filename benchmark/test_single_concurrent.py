#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单并发性能测试

目的：测试在没有并发竞争的情况下，单次查询的真实延迟
用于判断 100 并发下的高延迟是否由并发串行化导致
"""

import sys
import time
import numpy as np

# 添加路径
sys.path.insert(0, '/root/vector/hypervector/pyhypervec')
from pyhypervec import HypervecClient

def main():
    print("=" * 80)
    print("HyperVector 单并发性能测试")
    print("=" * 80)
    print()
    print("测试目标：")
    print("  - 测量单个客户端顺序查询的延迟")
    print("  - 判断是否存在并发串行化问题")
    print()
    print("测试配置：")
    print("  - Collection: wiki_hnsw_1m")
    print("  - 数据规模: 1,000,000 vectors, 768 dim")
    print("  - 查询次数: 1,000")
    print("  - 并发数: 1 (顺序执行)")
    print("  - ef_search: 128")
    print()

    # 连接服务器
    print("连接 HyperVector Server...")
    client = HypervecClient('http://localhost:8080', timeout=60.0)

    try:
        collections = client.list_collections()
        if 'wiki_hnsw_1m' not in collections:
            print("❌ Collection 'wiki_hnsw_1m' 不存在")
            print(f"   当前 Collections: {collections}")
            return
        print("✅ 连接成功")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    print()

    # 生成测试查询
    num_queries = 1000
    print(f"生成 {num_queries} 个随机查询向量...")
    queries = [np.random.randn(768).astype('float32') for _ in range(num_queries)]
    print("✅ 查询向量生成完成")
    print()

    # 开始测试
    print("-" * 80)
    print("开始测试...")
    print("-" * 80)

    latencies = []
    start_time = time.time()

    for i, query in enumerate(queries):
        query_start = time.time()

        try:
            results = client.search(
                collection_name='wiki_hnsw_1m',
                data=[query.tolist()],
                limit=10,
                search_params={'ef_search': 128}
            )

            latency = (time.time() - query_start) * 1000  # 转毫秒
            latencies.append(latency)

            if (i + 1) % 100 == 0:
                avg_so_far = sum(latencies) / len(latencies)
                print(f"  已完成: {i+1:4d}/{num_queries}  |  当前平均延迟: {avg_so_far:6.2f} ms")

        except Exception as e:
            print(f"  查询 {i+1} 失败: {e}")
            continue

    total_time = time.time() - start_time

    # 计算统计
    if not latencies:
        print("❌ 没有成功的查询")
        return

    latencies_sorted = sorted(latencies)
    avg = sum(latencies) / len(latencies)
    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    min_lat = min(latencies)
    max_lat = max(latencies)
    qps = len(latencies) / total_time

    # 输出结果
    print()
    print("=" * 80)
    print("单并发测试结果")
    print("=" * 80)
    print()
    print(f"测试配置:")
    print(f"  查询总数:       {len(latencies):,}")
    print(f"  并发数:         1 (顺序执行)")
    print(f"  总耗时:         {total_time:.2f} 秒")
    print()
    print(f"吞吐量:")
    print(f"  QPS:            {qps:.2f} 查询/秒")
    print()
    print(f"延迟统计:")
    print(f"  最小延迟:       {min_lat:.2f} ms")
    print(f"  平均延迟:       {avg:.2f} ms")
    print(f"  P50 (中位数):   {p50:.2f} ms")
    print(f"  P95:            {p95:.2f} ms")
    print(f"  P99:            {p99:.2f} ms")
    print(f"  最大延迟:       {max_lat:.2f} ms")
    print()
    print("=" * 80)
    print("诊断分析")
    print("=" * 80)
    print()

    # 诊断建议
    if avg < 20:
        print("✅ 结果：单查询延迟很低（< 20ms）")
        print()
        print("【诊断结论】")
        print("  问题根源：并发串行化导致延迟劣化")
        print()
        print("【证据】")
        print(f"  - 单并发平均延迟: {avg:.2f} ms")
        print(f"  - 100 并发平均延迟: ~547 ms")
        print(f"  - 延迟放大: {547/avg:.1f} 倍")
        print()
        print("【优化方向】")
        print("  1. 将 collection 的 RLock 改为读写锁")
        print("     - 读操作（search）之间可以并发")
        print("     - 只有写操作才互斥")
        print()
        print("  2. SWIG 绑定释放 GIL")
        print("     - 在 index->Search() 外层加 Py_BEGIN_ALLOW_THREADS")
        print("     - 让 Python 线程能真正并发进入 C++ 检索")
        print()
        print("  3. HNSW 搜索启用多线程")
        print("     - 去掉 omp_set_num_threads(1)")
        print("     - 改为可配置的线程数")
        print()
        print("  4. pyhypervec 客户端连接池")
        print("     - 使用长连接而非每次新建")
        print("     - HTTP/2 多路复用")

    elif avg < 100:
        print("⚠️  结果：单查询延迟中等（20-100ms）")
        print()
        print("【诊断结论】")
        print("  可能是混合问题：部分并发串行化 + HNSW 性能")
        print()
        print("【证据】")
        print(f"  - 单并发平均延迟: {avg:.2f} ms")
        print(f"  - 100 并发平均延迟: ~547 ms")
        print(f"  - 延迟放大: {547/avg:.1f} 倍")
        print()
        print("【建议】")
        print("  1. 先优化并发机制（读写锁、GIL、连接池）")
        print("  2. 再调整 HNSW 参数（降低 ef_search）")
        print("  3. 检查是否有不必要的回表查询")

    else:
        print("❌ 结果：单查询延迟很高（> 100ms）")
        print()
        print("【诊断结论】")
        print("  问题根源：HNSW 在 1M 数据集上本身就慢")
        print()
        print("【证据】")
        print(f"  - 单并发平均延迟: {avg:.2f} ms")
        print(f"  - 100 并发平均延迟: ~547 ms")
        print(f"  - 延迟放大: {547/avg:.1f} 倍（相对较小）")
        print()
        print("【优化方向】")
        print("  1. 降低 ef_search 参数")
        print("     - 当前: 128")
        print("     - 尝试: 32 或 64")
        print()
        print("  2. 优化 HNSW 参数")
        print("     - 检查 M 和 ef_construction 是否合理")
        print("     - 可能需要调整索引构建参数")
        print()
        print("  3. 检查回表开销")
        print("     - 是否每次查询都回 SQLite 查标量字段")
        print("     - 考虑增加内存缓存")
        print()
        print("  4. 绕过 HTTP 直接测试 C++")
        print("     - 排除 HTTP/序列化开销")
        print("     - 定位真正的瓶颈")

    print()
    print("=" * 80)
    print("建议下一步")
    print("=" * 80)
    print()
    print("运行多并发对比测试，查看延迟如何随并发数变化：")
    print("  python3 test_multiple_concurrency.py")
    print()

if __name__ == '__main__':
    main()
