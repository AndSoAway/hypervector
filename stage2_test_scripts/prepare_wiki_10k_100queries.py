#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Wiki 1M 数据集中生成 10K base、100 query 和精确 TopK groundtruth。
"""

import argparse
import struct
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_fbin(filename):
    with open(filename, "rb") as f:
        num_vectors, dim = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(num_vectors, dim)


def write_fbin(filename, vectors):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32, order="C")
    with path.open("wb") as f:
        f.write(struct.pack("ii", vectors.shape[0], vectors.shape[1]))
        f.write(vectors.tobytes())


def write_ibin(filename, neighbors):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    neighbors = np.asarray(neighbors, dtype=np.int32, order="C")
    with path.open("wb") as f:
        f.write(struct.pack("ii", neighbors.shape[0], neighbors.shape[1]))
        f.write(neighbors.tobytes())


def exact_l2_topk(queries, base, top_k, block_size):
    gt = np.empty((queries.shape[0], top_k), dtype=np.int32)
    base_norm = np.sum(base * base, axis=1, dtype=np.float32)

    for start in tqdm(range(0, queries.shape[0], block_size), desc="生成 groundtruth", unit="block"):
        end = min(start + block_size, queries.shape[0])
        q = queries[start:end]
        q_norm = np.sum(q * q, axis=1, dtype=np.float32)[:, None]
        distances = q_norm + base_norm[None, :] - 2.0 * np.matmul(q, base.T)
        distances = np.maximum(distances, 0.0)
        candidate = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
        candidate_distances = np.take_along_axis(distances, candidate, axis=1)
        order = np.argsort(candidate_distances, axis=1)
        gt[start:end] = np.take_along_axis(candidate, order, axis=1).astype(np.int32)

    return gt


def main():
    parser = argparse.ArgumentParser(description="生成 Wiki 10K/100 queries 小测试集")
    parser.add_argument("--base", default="/root/vector/data/wiki_all_1M/base.1M.fbin")
    parser.add_argument("--queries", default="/root/vector/data/wiki_all_1M/queries.fbin")
    parser.add_argument("--output-dir", default="/root/vector/data/wiki_10K")
    parser.add_argument("--base-size", type=int, default=10000)
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20240626)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("生成 Wiki 10K / 100 queries 小测试集")
    print("=" * 80)
    print(f"base 输入: {args.base}")
    print(f"query 输入: {args.queries}")
    print(f"输出目录: {output_dir}")
    print(f"base size: {args.base_size:,}")
    print(f"query size: {args.query_size:,}")
    print(f"groundtruth topK: {args.top_k}")
    print(f"随机 seed: {args.seed}")
    print()

    base = load_fbin(args.base)
    queries = load_fbin(args.queries)
    if args.base_size > base.shape[0]:
        raise ValueError(f"base-size {args.base_size} 超过原始 base 数量 {base.shape[0]}")
    if args.query_size > queries.shape[0]:
        raise ValueError(f"query-size {args.query_size} 超过原始 query 数量 {queries.shape[0]}")
    if args.top_k > args.base_size:
        raise ValueError("top-k 不能超过 base-size")

    rng = np.random.default_rng(args.seed)
    sample_ids = np.sort(rng.choice(base.shape[0], size=args.base_size, replace=False).astype(np.int64))
    small_base = np.ascontiguousarray(base[sample_ids], dtype=np.float32)
    small_queries = np.ascontiguousarray(queries[:args.query_size], dtype=np.float32)

    base_out = output_dir / "base.10K.fbin"
    queries_out = output_dir / "queries.100.fbin"
    gt_out = output_dir / "groundtruth.10K.neighbors.ibin"
    ids_out = output_dir / "base.10K.original_ids.npy"

    print("写出小数据集...")
    write_fbin(base_out, small_base)
    write_fbin(queries_out, small_queries)
    np.save(ids_out, sample_ids)

    print("精确暴力计算 groundtruth...")
    gt = exact_l2_topk(small_queries, small_base, args.top_k, args.block_size)
    write_ibin(gt_out, gt)

    print()
    print("完成")
    print(f"base: {base_out} shape={small_base.shape}")
    print(f"queries: {queries_out} shape={small_queries.shape}")
    print(f"groundtruth: {gt_out} shape={gt.shape}")
    print(f"原始 base 行号: {ids_out}")
    print("说明: groundtruth 中的 id 是 10K 小集合内部下标，匹配导入脚本生成的 wiki_0..wiki_9999。")


if __name__ == "__main__":
    main()
