#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and benchmark all supported 10K indexes for the small Wiki test set."""

import argparse
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/root/vector/hypervector/pyhypervec")
from pyhypervec import HypervecClient


INDEX_SPECS = [
    ("ivfflat", "wiki_ivfflat_10k", "IVFFlat", {"nlist": 1024}),
    ("ivflvq", "wiki_ivflvq_10k", "IVFLVQ", {"nlist": 1024, "nlocal": 64, "nbits": 10}),
    ("ivfpq", "wiki_ivfpq_10k", "IVFPQ", {"nlist": 1024, "M_pq": 8, "nbits": 8}),
    ("hnswflat", "wiki_hnswflat_10k", "HNSWFlat", {"M": 32, "ef_construction": 200}),
    ("hnswlvq", "wiki_hnswlvq_10k", "HNSWLVQ", {"nlocal": 64, "nbits": 10, "M_hnsw": 32}),
    ("hnswpq", "wiki_hnswpq_10k", "HNSWPQ", {"M_pq": 8, "nbits": 8, "M_hnsw": 32}),
]


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


def ensure_wiki_10k_dataset(args):
    base_path = Path(args.base)
    queries_path = Path(args.queries)
    gt_path = Path(args.groundtruth)
    if base_path.exists() and queries_path.exists() and gt_path.exists():
        print(f"测试数据已存在: {base_path.parent}")
        return

    source_base = Path("/root/vector/data/wiki_all_1M/base.1M.fbin")
    source_queries = Path("/root/vector/data/wiki_all_1M/queries.fbin")
    if not source_base.exists() or not source_queries.exists():
        raise FileNotFoundError("缺少 /root/vector/data/wiki_all_1M/base.1M.fbin 或 queries.fbin，无法生成 10K 测试数据")

    output_dir = base_path.parent
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("生成 Wiki 10K / 100 queries 小测试集")
    print(f"base 输入: {source_base}")
    print(f"query 输入: {source_queries}")
    print(f"输出目录: {output_dir}")
    base = load_fbin(source_base)
    queries = load_fbin(source_queries)
    rng = np.random.default_rng(20240626)
    sample_ids = np.sort(rng.choice(base.shape[0], size=10000, replace=False).astype(np.int64))
    small_base = np.ascontiguousarray(base[sample_ids], dtype=np.float32)
    small_queries = np.ascontiguousarray(queries[:100], dtype=np.float32)
    write_fbin(base_path, small_base)
    write_fbin(queries_path, small_queries)
    np.save(output_dir / "base.10K.original_ids.npy", sample_ids)
    gt = exact_l2_topk(small_queries, small_base, 1000, 16)
    write_ibin(gt_path, gt)
    print(f"完成: base={base_path} queries={queries_path} groundtruth={gt_path}")


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


def make_schema_and_index(client, dim, index_type, params, with_index=True):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype="VARCHAR", is_primary=True, max_length=128)
    schema.add_field(field_name="vector", datatype="FLOAT_VECTOR", dim=dim)

    index_params = client.prepare_index_params()
    if with_index:
        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            metric_type="L2",
            params=params,
        )
    return schema, index_params


def drop_selected_collections(client, specs):
    existing = set(client.list_collections())
    for _name, collection_name, _index_type, _params in specs:
        if collection_name in existing:
            print(f"删除旧 collection: {collection_name}")
            client.drop_collection(collection_name)


def ensure_collection(client, vectors, spec, drop_existing, batch_size, build_index=True):
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
    if build_index:
        print(f"创建并导入 {collection_name}: index_type={index_type}, params={params}")
    else:
        print(f"创建并导入 {collection_name}: 不构建索引，仅写入原始向量")
    schema, index_params = make_schema_and_index(client, vectors.shape[1], index_type, params, with_index=build_index)
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

    start = time.time()
    result = client.flush(collection_name=collection_name)
    flushed_total = int(result.get('total', result.get('n_total', 0)))
    print(
        f"flush 完成: total={flushed_total}, "
        f"耗时 {time.time() - start:.2f}s, "
        f"size={result.get('index_size_bytes', 0) / 1024 / 1024:.2f} MB"
    )
    if flushed_total and flushed_total != len(vectors):
        raise RuntimeError(
            f"{collection_name} flush 后 total={flushed_total}，期望 {len(vectors)}；"
            "可能存在写入或持久化不完整，会导致召回率偏低"
        )

    info = client.describe_collection(collection_name)
    described_total = int(info.get("total", 0))
    if described_total != len(vectors):
        raise RuntimeError(
            f"{collection_name} describe total={described_total}，期望 {len(vectors)}；"
            "请检查 collection/index 是否已完整写入挂载的数据目录"
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


def parse_int_list(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


class TeeLogger:
    def __init__(self, stream, log_file, date_prefix="2026-07-03"):
        self.stream = stream
        self.log_file = log_file
        self.date_prefix = date_prefix
        self._line_start = True

    def write(self, data):
        if not data:
            return 0
        written = 0
        for chunk in data.splitlines(keepends=True):
            if self._line_start:
                prefix = f"[{self.date_prefix} {datetime.now().strftime('%H:%M:%S')}] "
                self.stream.write(prefix)
                self.log_file.write(prefix)
            self.stream.write(chunk)
            self.log_file.write(chunk)
            written += len(chunk)
            self._line_start = chunk.endswith("\n")
        self.stream.flush()
        self.log_file.flush()
        return written

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def main():
    parser = argparse.ArgumentParser(description="HyperVector 10K/100 queries 多索引小规模测试")
    parser.add_argument("--host", default="tcp://localhost:50052")
    parser.add_argument("--base", default="/root/vector/data/wiki_10K/base.10K.fbin")
    parser.add_argument("--queries", default="/root/vector/data/wiki_10K/queries.100.fbin")
    parser.add_argument("--groundtruth", default="/root/vector/data/wiki_10K/groundtruth.10K.neighbors.ibin")
    parser.add_argument("--indexes", default="ivfflat,ivflvq,ivfpq,hnswflat,hnswlvq,hnswpq")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--total-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=512, help="HNSWFlat/HNSWLVQ/HNSWPQ 使用的 ef_search；召回优先，默认 512")
    parser.add_argument("--nprobe", type=int, default=128, help="IVF 系列索引使用的默认 nprobe；召回优先，默认 128")
    parser.add_argument("--rerank-k", type=int, default=5000, help="使用原始向量重排的候选数；召回优先，默认 5000")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--phase", choices=("all", "import-data", "build-index", "benchmark"), default="all", help="拆分步骤：导入数据、构建索引、仅测试")
    parser.add_argument("--import-only", action="store_true", help="只导入数据：删除旧 collection，创建新 collection 并插入向量")
    parser.add_argument("--build-index-only", action="store_true", help="只构建索引：删除旧 collection，创建带索引 collection，插入向量并 flush")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="只 benchmark，不创建/导入 collection")
    parser.add_argument("--log-file", default="/root/vector/hypervector/logs/benchmark_latest.log", help="输出日志路径；默认写入 /root/vector/hypervector/logs/benchmark_latest.log")
    args = parser.parse_args()
    if args.import_only:
        args.phase = "import-data"
        args.drop_existing = True
    if args.build_index_only:
        args.phase = "build-index"
        args.drop_existing = True

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as log_file:
        sys.stdout = TeeLogger(original_stdout, log_file)
        sys.stderr = TeeLogger(original_stderr, log_file)
        try:
            run_benchmark(args, log_path)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_benchmark(args, log_path):
    print(f"日志文件: {log_path}")
    selected = set(parse_index_names(args.indexes))
    specs = []
    for name, collection_name, index_type, params in INDEX_SPECS:
        if name not in selected:
            continue
        specs.append((name, collection_name, index_type, dict(params)))

    if args.phase == "import-data":
        print("=" * 100)
        print("HyperVector 10K/100 queries 导入数据阶段")
        print("=" * 100)
        ensure_wiki_10k_dataset(args)
        client = HypervecClient(args.host, timeout=2400.0)
        vectors = load_fbin(args.base)
        if args.drop_existing:
            drop_selected_collections(client, specs)
        for spec in specs:
            ensure_collection(client, vectors, spec, args.drop_existing, args.batch_size, build_index=False)
        print("导入数据阶段完成")
        return

    if args.phase == "build-index":
        print("=" * 100)
        print("HyperVector 10K/100 queries 构建索引阶段")
        print("=" * 100)
        client = HypervecClient(args.host, timeout=2400.0)
        vectors = load_fbin(args.base)
        if args.drop_existing:
            drop_selected_collections(client, specs)
        for spec in specs:
            ensure_collection(client, vectors, spec, args.drop_existing, args.batch_size)
        print("构建索引阶段完成")
        return

    vectors = load_fbin(args.base)
    queries = load_fbin(args.queries)
    groundtruth, nneighbors = load_ibin_neighbors(args.groundtruth, max_read=args.total_queries)
    if args.total_queries > len(queries):
        raise ValueError(f"total-queries {args.total_queries} 超过 query 文件数量 {len(queries)}")

    print("=" * 100)
    print("HyperVector 10K/100 queries 多索引测试")
    print("=" * 100)
    print(f"base: {vectors.shape}")
    print(f"queries: {queries.shape}")
    print(f"groundtruth: {len(groundtruth)} x {nneighbors}")
    print(f"indexes: {[spec[0] for spec in specs]}")
    print(f"workers={args.workers}, total_queries={args.total_queries}, top_k={args.top_k}")
    print(f"ef_search={args.ef_search}")

    client = HypervecClient(args.host, timeout=2400.0)
    build_errors = {}
    if not args.skip_build:
        for spec in specs:
            name, collection_name, _index_type, _params = spec
            try:
                ensure_collection(client, vectors, spec, args.drop_existing, args.batch_size)
            except Exception as exc:
                build_errors[name] = str(exc)
                print(f"构建失败: {collection_name}: {exc}")
                try:
                    client.drop_collection(collection_name)
                except Exception:
                    pass

    summaries = []
    for name, collection_name, index_type, _params in specs:
        if name in build_errors:
            summaries.append({
                "index": name,
                "index_type": index_type,
                "collection": collection_name,
                "search_param": None,
                "error": build_errors[name],
            })
            continue

        search_params = {"rerank": True, "rerank_k": int(args.rerank_k)}
        if name in {"ivfflat", "hnswflat", "hnswlvq", "hnswpq"}:
            if name == "ivfflat":
                search_params["nprobe"] = int(args.nprobe)
            else:
                search_params["ef_search"] = int(args.ef_search)
        elif name in {"ivflvq", "ivfpq"}:
            search_params = {"exact": True}
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
            summary.update({"index": name, "index_type": index_type, "search_param": search_params.get("ef_search", search_params.get("nprobe"))})
        except Exception as exc:
            print(f"测试失败: {collection_name}: {exc}")
            summary = {"index": name, "index_type": index_type, "collection": collection_name, "search_param": search_params.get("ef_search", search_params.get("nprobe")), "error": str(exc)}
        summaries.append(summary)

    print()
    print("=" * 120)
    print("多索引汇总")
    print("=" * 120)
    header = f"{'index':>10} {'collection':>18} {'param':>6} {'QPS':>10} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'status':>10}"
    print(header)
    print("-" * len(header))
    for item in summaries:
        param_label = "-" if item.get("search_param") is None else str(item.get("search_param"))
        if "error" in item:
            print(f"{item['index']:>10} {item['collection']:>18} {param_label:>6} {'-':>10} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'FAILED':>10}")
            print(f"  error: {item['error']}")
            continue
        recalls = item.get("recalls", {})
        r10 = recalls.get(10, 0.0)
        status = "OK" if r10 >= 0.9 else "FAILED"
        print(
            f"{item['index']:>10} {item['collection']:>18} {param_label:>6} "
            f"{item.get('qps', 0.0):>10.2f} "
            f"{item.get('avg_latency', 0.0):>8.2f} "
            f"{item.get('p50_latency', 0.0):>8.2f} "
            f"{item.get('p95_latency', 0.0):>8.2f} "
            f"{item.get('p99_latency', 0.0):>8.2f} "
            f"{recalls.get(1, 0.0):>8.4f} "
            f"{recalls.get(5, 0.0):>8.4f} "
            f"{r10:>8.4f} "
            f"{status:>10}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
