#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 .fbin 格式向量文件的工具函数
"""

import numpy as np
import struct

def read_fbin(filename):
    """
    读取 .fbin 格式的向量文件

    文件格式：
    [4 bytes] n - 向量数量 (int32)
    [4 bytes] d - 向量维度 (int32)
    [n*d*4 bytes] 向量数据 (float32)

    Args:
        filename: .fbin 文件路径

    Returns:
        vectors: numpy array (n, d)
        n: 向量数量
        d: 向量维度
    """
    print(f"读取文件：{filename}")

    with open(filename, 'rb') as f:
        # 读取头部
        n = struct.unpack('i', f.read(4))[0]
        d = struct.unpack('i', f.read(4))[0]

        print(f"  向量数量：{n:,}")
        print(f"  向量维度：{d}")

        # 读取向量数据
        data = np.fromfile(f, dtype=np.float32, count=n * d)
        vectors = data.reshape(n, d)

        print(f"  数据大小：{vectors.nbytes / 1024 / 1024:.2f} MB")

    return vectors, n, d

def read_ground_truth(filename, n_queries, k=100):
    """
    读取 ground truth 文件（用于计算召回率）

    文件格式：
    [4 bytes] nq - 查询数量
    [4 bytes] k - 每个查询的结果数
    [nq*k*4 bytes] ground truth IDs (int32)

    Args:
        filename: ground truth 文件路径
        n_queries: 查询数量
        k: 每个查询返回的结果数

    Returns:
        gt: numpy array (n_queries, k)
    """
    print(f"读取 ground truth：{filename}")

    with open(filename, 'rb') as f:
        nq = struct.unpack('i', f.read(4))[0]
        gt_k = struct.unpack('i', f.read(4))[0]

        print(f"  查询数量：{nq:,}")
        print(f"  每查询返回：{gt_k} 个结果")

        # 读取 ground truth
        data = np.fromfile(f, dtype=np.int32, count=nq * gt_k)
        gt = data.reshape(nq, gt_k)

    return gt[:n_queries, :k]

if __name__ == '__main__':
    # 测试读取
    import sys

    if len(sys.argv) < 2:
        print("用法: python fbin_utils.py <fbin_file>")
        sys.exit(1)

    vectors, n, d = read_fbin(sys.argv[1])
    print(f"\n✅ 成功读取 {n:,} 条 {d} 维向量")
    print(f"前 5 个向量的第一维：{vectors[:5, 0]}")
