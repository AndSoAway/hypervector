#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare 10K/100-query Wiki benchmark data for the six demo/benchmark indexes."""

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
    print(f"top_k: {args.top_k}")
    print("输出文件: base.10K.fbin, queries.100.fbin, groundtruth.10K.neighbors.ibin, base.10K.original_ids.npy")

    base_all = load_fbin(args.base)
    queries_all = load_fbin(args.queries)
    if args.base_size > len(base_all):
        raise ValueError(f"base_size={args.base_size} 超过 base 数量 {len(base_all)}")
    if args.query_size > len(queries_all):
        raise ValueError(f"query_size={args.query_size} 超过 queries 数量 {len(queries_all)}")

    rng = np.random.default_rng(args.seed)
    base_indices = np.sort(rng.choice(len(base_all), size=args.base_size, replace=False))
    query_indices = np.sort(rng.choice(len(queries_all), size=args.query_size, replace=False))

    base_subset = np.asarray(base_all[base_indices], dtype=np.float32, order="C")
    query_subset = np.asarray(queries_all[query_indices], dtype=np.float32, order="C")
    groundtruth = exact_l2_topk(query_subset, base_subset, args.top_k, args.block_size)

    write_fbin(output_dir / "base.10K.fbin", base_subset)
    write_fbin(output_dir / "queries.100.fbin", query_subset)
    write_ibin(output_dir / "groundtruth.10K.neighbors.ibin", groundtruth)
    np.save(output_dir / "base.10K.original_ids.npy", base_indices)

    print("\n已生成:")
    print(f"- {output_dir / 'base.10K.fbin'}")
    print(f"- {output_dir / 'queries.100.fbin'}")
    print(f"- {output_dir / 'groundtruth.10K.neighbors.ibin'}")
    print(f"- {output_dir / 'base.10K.original_ids.npy'}")


if __name__ == "__main__":
    main()
