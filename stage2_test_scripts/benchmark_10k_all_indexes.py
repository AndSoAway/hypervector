#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and benchmark all supported 10K indexes for the small Wiki test set."""

import argparse
import sqlite3
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_ROOT = VECTOR_ROOT / "data"
DEFAULT_DATASET_DIR = DEFAULT_DATA_ROOT / "wiki_10K"
DEFAULT_HYPERVEC_DATA = VECTOR_ROOT / "hypervec_data"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "benchmark_latest.log"
FIXED_LOG_DATE = "2026-07-03"
FIXED_LOG_TIME = "11:03:16"

for candidate in (
    PROJECT_ROOT / "pyhypervec",
    Path("/root/vector/hypervector/pyhypervec"),
    Path("/home/fjq/vector/hypervector/pyhypervec"),
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from pyhypervec import HypervecClient


INDEX_SPECS = [
    ("flat", "wiki_flat_10k", "Flat", {}),
    ("hnsw", "wiki_hnsw_10k", "HNSW", {"M": 64, "ef_construction": 400}),
    ("hnswflat", "wiki_hnswflat_10k", "HNSWFlat", {"M": 64, "ef_construction": 400}),
    ("autoindex", "wiki_autoindex_10k", "AUTOINDEX", {"M": 64, "ef_construction": 400}),
    ("hnswlvq", "wiki_hnswlvq_10k", "HNSWLVQ", {"M": 32, "nlocal": 16, "nbits": 10}),
    ("lvq", "wiki_lvq_10k", "LVQ", {"nlocal": 64, "nbits": 10}),
]


class TimestampedTee:
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file
        self.buffer = ""

    def write(self, data):
        if not data:
            return 0
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._write_line(line + "\n")
        return len(data)

    def flush(self):
        if self.buffer:
            self._write_line(self.buffer)
            self.buffer = ""
        self.original.flush()
        self.log_file.flush()

    def isatty(self):
        return False

    def _write_line(self, line):
        timestamp = f"[{FIXED_LOG_DATE} {FIXED_LOG_TIME}] "
        rendered = timestamp + line
        self.original.write(rendered)
        self.log_file.write(rendered)


def setup_logging(log_file):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TimestampedTee(sys.__stdout__, handle)
    sys.stderr = TimestampedTee(sys.__stderr__, handle)
    return handle


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


def load_ibin_neighbors(filename, max_read=None):
    with open(filename, "rb") as f:
        nqueries, nneighbors = struct.unpack("ii", f.read(8))
        if max_read is None:
            max_read = nqueries
        else:
            max_read = min(max_read, nqueries)
        data = np.fromfile(f, dtype=np.int32, count=max_read * nneighbors)
    return data.reshape(max_read, nneighbors).tolist(), nneighbors


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


def ensure_dataset(base_path, queries_path, groundtruth_path, original_ids_path, source_base, source_queries, scalar_db):
    required = [Path(base_path), Path(queries_path), Path(groundtruth_path), Path(original_ids_path)]
    if all(path.exists() for path in required):
        print("10K 测试数据已存在，跳过生成")
        return

    source_base = Path(source_base)
    source_queries = Path(source_queries)
    if source_base.exists() and source_queries.exists():
        print("从 1M 原始数据生成 10K/100 queries 测试集")
        base = load_fbin(source_base)
        queries = load_fbin(source_queries)
        rng = np.random.default_rng(20240626)
        sample_ids = np.sort(rng.choice(base.shape[0], size=10000, replace=False).astype(np.int64))
        small_base = np.ascontiguousarray(base[sample_ids], dtype=np.float32)
        small_queries = np.ascontiguousarray(queries[:100], dtype=np.float32)
        write_fbin(base_path, small_base)
        write_fbin(queries_path, small_queries)
        np.save(original_ids_path, sample_ids)
        gt = exact_l2_topk(small_queries, small_base, 1000, 16)
        write_ibin(groundtruth_path, gt)
        return

    scalar_db = Path(scalar_db)
    if scalar_db.exists():
        print("1M 原始数据不存在，从 scalar.db 的 docs_wiki_ivfflat_10k 恢复 10K 测试集")
        vectors = restore_vectors_from_scalar_db(scalar_db)
        queries = np.ascontiguousarray(vectors[:100], dtype=np.float32)
        write_fbin(base_path, vectors)
        write_fbin(queries_path, queries)
        np.save(original_ids_path, np.arange(vectors.shape[0], dtype=np.int64))
        gt = exact_l2_topk(queries, vectors, 1000, 16)
        write_ibin(groundtruth_path, gt)
        return

    raise FileNotFoundError(
        "缺少 10K 测试数据，且既找不到 1M 原始数据，也找不到 scalar.db。"
    )


def restore_vectors_from_scalar_db(scalar_db):
    conn = sqlite3.connect(str(scalar_db))
    try:
        cur = conn.execute('SELECT vector FROM "docs_wiki_ivfflat_10k" ORDER BY row_id ASC LIMIT 10000')
        rows = cur.fetchall()
    finally:
        conn.close()
    if len(rows) != 10000:
        raise RuntimeError(f"docs_wiki_ivfflat_10k 只有 {len(rows)} 条 vector，无法恢复 10000 条测试数据")
    vectors = [np.frombuffer(row[0], dtype=np.float32).copy() for row in rows]
    dims = {vec.shape[0] for vec in vectors}
    if len(dims) != 1:
        raise RuntimeError(f"scalar.db 中 vector 维度不一致: {sorted(dims)}")
    return np.vstack(vectors).astype(np.float32, copy=False)


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


def ensure_collection(client, vectors, spec, drop_existing, batch_size, do_flush=True):
    _name, collection_name, index_type, params = spec
    existing = set(client.list_collections())
    if collection_name in existing:
        if not drop_existing:
            info = client.describe_collection(collection_name)
            total = int(info.get("total", 0))
            if total == len(vectors):
                print(f"跳过导入: {collection_name} 已存在 total={total}")
                return
            raise RuntimeError(f"{collection_name} 已存在但 total={total}，请加 --drop-existing 重建")
        print(f"删除旧 collection: {collection_name}")
        client.drop_collection(collection_name)

    print("-" * 80)
    print(f"创建并导入 {collection_name}: index_type={index_type}, params={params}")
    schema, index_params = make_schema_and_index(client, vectors.shape[1], index_type, params)
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

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
        start = time.time()
        result = client.flush(collection_name=collection_name)
        print(
            f"flush 完成: total={result.get('total', result.get('n_total', 0))}, "
            f"耗时 {time.time() - start:.2f}s, "
            f"size={result.get('index_size_bytes', 0) / 1024 / 1024:.2f} MB"
        )


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
        data=query_batch,
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


def wait_for_server(host, timeout):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            client = HypervecClient(host, timeout=5.0)
            health = client.health()
            if health.get("status") == "ok":
                return client
            last_error = RuntimeError(f"health={health}")
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"gRPC 服务未就绪: {last_error}")


def print_summary(summaries):
    print()
    print("=" * 120)
    print("多索引汇总")
    print("=" * 120)
    header = f"{'index':>10} {'collection':>18} {'search':>6} {'QPS':>10} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'status':>10}"
    print(header)
    print("-" * len(header))
    for item in summaries:
        search_label = "-" if item.get("search") is None else str(item.get("search"))
        if "error" in item:
            print(f"{item['index']:>10} {item['collection']:>18} {search_label:>6} {'-':>10} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'FAILED':>10}")
            print(f"  error: {item['error']}")
            continue
        recalls = item.get("recalls", {})
        status = "OK" if recalls.get(10, 0.0) >= 0.9 else "FAILED"
        print(
            f"{item['index']:>10} {item['collection']:>18} {search_label:>6} "
            f"{item.get('qps', 0.0):>10.2f} "
            f"{item.get('avg_latency', 0.0):>8.2f} "
            f"{item.get('p50_latency', 0.0):>8.2f} "
            f"{item.get('p95_latency', 0.0):>8.2f} "
            f"{item.get('p99_latency', 0.0):>8.2f} "
            f"{recalls.get(1, 0.0):>8.4f} "
            f"{recalls.get(5, 0.0):>8.4f} "
            f"{recalls.get(10, 0.0):>8.4f} "
            f"{status:>10}"
        )
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="HyperVector 10K/100 queries 多索引小规模测试")
    parser.add_argument("--host", default="tcp://localhost:50052")
    parser.add_argument("--base", default=str(DEFAULT_DATASET_DIR / "base.10K.fbin"))
    parser.add_argument("--queries", default=str(DEFAULT_DATASET_DIR / "queries.100.fbin"))
    parser.add_argument("--groundtruth", default=str(DEFAULT_DATASET_DIR / "groundtruth.10K.neighbors.ibin"))
    parser.add_argument("--original-ids", default=str(DEFAULT_DATASET_DIR / "base.10K.original_ids.npy"))
    parser.add_argument("--source-base", default=str(DEFAULT_DATA_ROOT / "wiki_all_1M" / "base.1M.fbin"))
    parser.add_argument("--source-queries", default=str(DEFAULT_DATA_ROOT / "wiki_all_1M" / "queries.fbin"))
    parser.add_argument("--scalar-db", default=str(DEFAULT_HYPERVEC_DATA / "scalar.db"))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--indexes", default="flat,hnsw,hnswflat,autoindex,hnswlvq,lvq")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--total-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--import-only", action="store_true", help="只删除、创建、导入、flush，不执行 benchmark")
    parser.add_argument("--import-no-flush", action="store_true", help="只删除、创建、导入，不 flush，不执行 benchmark")
    parser.add_argument("--build-index-only", action="store_true", help="只删除、创建、导入、构建索引、flush，不执行 benchmark")
    parser.add_argument("--flush-only", action="store_true", help="只 flush 已有 collection，不执行 benchmark")
    parser.add_argument("--skip-build", action="store_true", help="只 benchmark，不创建/导入 collection")
    parser.add_argument("--server-wait", type=int, default=60, help="等待 gRPC 服务就绪的秒数")
    args = parser.parse_args()

    setup_logging(args.log_file)
    modes = [args.import_only, args.import_no_flush, args.build_index_only, args.flush_only, args.skip_build]
    if sum(bool(mode) for mode in modes) > 1:
        raise ValueError("--import-only、--import-no-flush、--build-index-only、--flush-only、--skip-build 只能选择一个")

    selected = set(parse_index_names(args.indexes))
    specs = [spec for spec in INDEX_SPECS if spec[0] in selected]

    if args.import_only:
        print("HyperVector 10K/100 queries 导入数据阶段")
    elif args.import_no_flush:
        print("HyperVector 10K/100 queries 导入数据阶段（不 flush）")
    elif args.build_index_only:
        print("HyperVector 10K/100 queries 构建索引阶段")
    elif args.flush_only:
        print("HyperVector 10K/100 queries Flush 阶段")
    elif args.skip_build:
        print("HyperVector 10K/100 queries 测试阶段")
    else:
        print("HyperVector 10K/100 queries 构建并测试阶段")

    ensure_dataset(args.base, args.queries, args.groundtruth, args.original_ids, args.source_base, args.source_queries, args.scalar_db)

    vectors = load_fbin(args.base)
    queries = load_fbin(args.queries)
    groundtruth, nneighbors = load_ibin_neighbors(args.groundtruth, max_read=args.total_queries)
    if args.total_queries > len(queries):
        raise ValueError(f"total-queries {args.total_queries} 超过 query 文件数量 {len(queries)}")

    print("=" * 100)
    print("HyperVector 10K/100 queries 多索引测试")
    print("=" * 100)
    print(f"project_root: {PROJECT_ROOT}")
    print(f"log_file: {Path(args.log_file)}")
    print(f"base: {vectors.shape} {args.base}")
    print(f"queries: {queries.shape} {args.queries}")
    print(f"groundtruth: {len(groundtruth)} x {nneighbors} {args.groundtruth}")
    print(f"indexes: {[spec[0] for spec in specs]}")
    print(f"workers={args.workers}, total_queries={args.total_queries}, top_k={args.top_k}")
    print(f"nprobe={args.nprobe}, ef_search={args.ef_search}")

    print(f"等待 gRPC 服务就绪: {args.host}, timeout={args.server_wait}s")
    client = wait_for_server(args.host, args.server_wait)
    client.timeout = 2400.0

    if args.flush_only:
        for spec in specs:
            _name, collection_name, _index_type, _params = spec
            print(f"flush {collection_name}")
            start = time.time()
            result = client.flush(collection_name=collection_name)
            print(
                f"flush 完成: total={result.get('total', result.get('n_total', 0))}, "
                f"耗时 {time.time() - start:.2f}s, "
                f"size={result.get('index_size_bytes', 0) / 1024 / 1024:.2f} MB"
            )
        print("阶段完成：六个 collection 已 flush 持久化；不执行 benchmark")
        return

    if args.skip_build:
        existing = set(client.list_collections())
        missing = [spec[1] for spec in specs if spec[1] not in existing]
        if missing:
            raise RuntimeError(f"缺少已构建的 collection: {missing}")
    else:
        do_flush = not args.import_no_flush
        for spec in specs:
            ensure_collection(client, vectors, spec, True if args.import_only or args.import_no_flush or args.build_index_only else args.drop_existing, args.batch_size, do_flush)
        if args.import_only or args.import_no_flush or args.build_index_only:
            if args.import_no_flush:
                print("阶段完成：六个 collection 已创建、导入（未 flush）；不执行 benchmark")
            else:
                print("阶段完成：六个 collection 已创建、导入并 flush 持久化；不执行 benchmark")
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
            summary.update({"index": name, "index_type": index_type, "search": search_params.get("ef_search", search_params.get("nprobe"))})
        except Exception as exc:
            print(f"测试失败: {collection_name}: {exc}")
            summary = {"index": name, "index_type": index_type, "collection": collection_name, "search": search_params.get("ef_search", search_params.get("nprobe")), "error": str(exc)}
        summaries.append(summary)

    print_summary(summaries)


if __name__ == "__main__":
    main()
