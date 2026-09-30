#!/usr/bin/env python3
"""End-to-end smoke test for the file-based evaluation pipeline."""

import json
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def write_fvecs(path, rows):
    dimension = len(rows[0])
    with path.open("wb") as output:
        for row in rows:
            if len(row) != dimension:
                raise ValueError("inconsistent test vector dimension")
            output.write(struct.pack(f"<i{dimension}f", dimension, *row))


def run(*arguments):
    return subprocess.run(
        [str(argument) for argument in arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def assert_fingerprint(artifact, path):
    assert artifact["path"] == str(path.resolve())
    assert artifact["size_bytes"] == path.stat().st_size
    assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def assert_process_memory(resources, measurement_point):
    assert resources["memory_scope"] == "process"
    assert resources["measurement_point"] == measurement_point
    peak_rss = resources["peak_rss_bytes"]
    if peak_rss is None:
        assert resources["peak_rss_source"] in (
            "unsupported",
            "getrusage.ru_maxrss",
        )
    else:
        assert isinstance(peak_rss, int)
        assert peak_rss > 0
        assert resources["peak_rss_source"] == "getrusage.ru_maxrss"


def read_provenance(path):
    record = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key:
            record[key] = value
    return record


def main():
    ground_truth_tool = Path(sys.argv[1]).resolve()
    build_tool = Path(sys.argv[2]).resolve()
    eval_tool = Path(sys.argv[3]).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        base = directory / "base.fvecs"
        queries = directory / "queries.fvecs"
        ground_truth = directory / "ground-truth.ivecs"
        index = directory / "flat.index"
        build_report_path = directory / "build-report.json"
        report_path = directory / "report.json"
        write_fvecs(base, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        write_fvecs(queries, [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])

        run(
            ground_truth_tool,
            "--base",
            base,
            "--queries",
            queries,
            "--output",
            ground_truth,
            "--k",
            2,
            "--metric",
            "l2",
        )
        run(
            build_tool,
            "--input",
            base,
            "--output",
            index,
            "--index-type",
            "flat",
            "--metric",
            "l2",
            "--json-output",
            build_report_path,
        )
        run(
            eval_tool,
            "--index",
            index,
            "--queries",
            queries,
            "--ground-truth",
            ground_truth,
            "--metric",
            "l2",
            "--k",
            2,
            "--warmup-runs",
            0,
            "--measured-runs",
            2,
            "--query-batch-size",
            1,
            "--json-output",
            report_path,
        )

        build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
        assert build_report["format"] == "hypervec-build-report-v1"
        assert build_report["dataset"]["semantic_metric"] == "l2"
        assert build_report["dataset"]["vector_count"] == 3
        assert build_report["index"]["requested_type"] == "flat"
        assert build_report["index"]["metric"] == "l2"
        assert build_report["index"]["requested_parameters"] == []
        assert build_report["timing"]["build_seconds"] >= 0.0
        assert Path(build_report["dataset"]["base"]).is_absolute()
        assert_fingerprint(build_report["artifacts"]["base"], base)
        assert build_report["artifacts"]["training_queries"] is None
        assert_fingerprint(build_report["artifacts"]["index"], index)
        assert_process_memory(
            build_report["resources"], "after_index_write"
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["format"] == "hypervec-eval-report-v2"
        # The parameter family collapses several index types; the persisted
        # format name must identify the concrete algorithm.
        assert report["index"]["parameter_family"] == "generic"
        # An L2 "flat" request builds IndexFlatL2, which persists under its own
        # IFlm tag; the old parameter family reported all of these as "generic".
        assert report["index"]["format_name"] == "flat_l2"
        assert report["index"]["format_tag"] == "IFlm"
        assert report["execution"]["environment"]["omp_max_threads"] > 0
        assert "index_load_rss_delta_bytes" in report["resources"]
        assert report["index"]["metric"] == "l2"
        assert report["index"]["vector_count"] == 3
        assert report["workload"]["query_count"] == 4
        assert report["workload"]["semantic_metric"] == "l2"
        assert report["workload"]["k"] == 2
        assert report["execution"]["warmup_runs"] == 0
        assert report["execution"]["measured_runs"] == 2
        assert report["execution"]["search_parameters"] == []
        assert report["metrics"]["recall_at_k"] == 1.0
        latency = report["metrics"]["batch_latency_ms"]
        assert latency["query_batch_size"] == 1
        assert latency["sample_count"] == 8
        assert latency["percentile_method"] == "nearest-rank"
        assert 0.0 <= latency["p50"] <= latency["p95"] <= latency["p99"]
        assert Path(report["index"]["path"]).is_absolute()
        assert Path(report["workload"]["queries"]).is_absolute()
        assert_fingerprint(report["artifacts"]["index"], index)
        assert_fingerprint(report["artifacts"]["queries"], queries)
        assert_fingerprint(
            report["artifacts"]["ground_truth"], ground_truth
        )
        assert_process_memory(report["resources"], "after_search")
        assert report["rerank"]["enabled"] is False

        rerank_report_path = directory / "rerank-report.json"
        run(
            eval_tool,
            "--index", index,
            "--queries", queries,
            "--ground-truth", ground_truth,
            "--metric", "l2",
            "--k", 2,
            "--warmup-runs", 0,
            "--measured-runs", 1,
            "--rerank-base", base,
            "--rerank-candidates", 3,
            "--json-output", rerank_report_path,
        )
        rerank_report = json.loads(rerank_report_path.read_text(encoding="utf-8"))
        assert rerank_report["rerank"]["enabled"] is True
        assert rerank_report["rerank"]["candidates"] == 3
        assert_fingerprint(rerank_report["rerank"]["exact_base"], base)
        assert rerank_report["metrics"]["recall_at_k"] == 1.0

        # An unsatisfiable depth must be rejected, not silently cap recall.
        oversized = subprocess.run(
            [str(eval_tool), "--index", str(index), "--queries", str(queries),
             "--ground-truth", str(ground_truth), "--metric", "l2",
             "--k", "99", "--warmup-runs", "0", "--measured-runs", "1"],
            check=False, text=True, capture_output=True)
        assert oversized.returncode == 1
        assert "exceeds the number of vectors" in oversized.stderr

        # A flat index accepts no runtime parameter, so sweep HNSW over ef_search.
        hnsw = directory / "hnsw.index"
        run(build_tool, "--input", base, "--output", hnsw,
            "--index-type", "hnsw_flat", "--metric", "l2",
            "--index-param", "m_hnsw=int:8")
        sweep_path = directory / "sweep-report.json"
        run(eval_tool, "--index", hnsw, "--queries", queries,
            "--ground-truth", ground_truth, "--metric", "l2", "--k", 2,
            "--warmup-runs", 0, "--measured-runs", 1,
            "--sweep", "ef_search=8,16,32",
            "--json-output", sweep_path)
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        points = sweep["search_sweep"]["points"]
        assert [point["value"] for point in points] == [8, 16, 32]
        assert all(point["recall_at_k"] == 1.0 for point in points)
        assert sweep["index"]["format_name"] == "hnsw_flat"

        # Provenance: a ground truth must name the base the index was built
        # from; a stale or mismatched file must be rejected.
        gt_prov = directory / "gt.prov"
        build_prov = directory / "build.prov"
        run(ground_truth_tool, "--base", base, "--queries", queries,
            "--output", ground_truth, "--k", 2, "--metric", "l2",
            "--provenance", gt_prov)
        run(build_tool, "--input", base, "--output", index,
            "--index-type", "flat", "--metric", "l2",
            "--provenance", build_prov)
        assert read_provenance(gt_prov)["base_sha256"] ==             read_provenance(build_prov)["base_sha256"]
        run(eval_tool, "--index", index, "--queries", queries,
            "--ground-truth", ground_truth, "--metric", "l2", "--k", 2,
            "--warmup-runs", 0, "--measured-runs", 1,
            "--build-provenance", build_prov,
            "--ground-truth-provenance", gt_prov)

        # Row-count-matching but different bytes: the hash check, not a shape
        # error, must be what rejects this ground truth.
        stale_gt = directory / "stale.ivecs"
        run(ground_truth_tool, "--base", base, "--queries", queries,
            "--output", stale_gt, "--k", 3, "--metric", "l2")
        mismatched = subprocess.run(
            [str(eval_tool), "--index", str(index), "--queries", str(queries),
             "--ground-truth", str(stale_gt), "--metric", "l2", "--k", "2",
             "--warmup-runs", "0", "--measured-runs", "1",
             "--ground-truth-provenance", str(gt_prov)],
            check=False, text=True, capture_output=True)
        assert mismatched.returncode == 1
        assert "not the file hypervec_ground_truth" in mismatched.stderr

        concurrent_report_path = directory / "concurrent-report.json"
        run(
            eval_tool,
            "--index",
            index,
            "--queries",
            queries,
            "--ground-truth",
            ground_truth,
            "--metric",
            "l2",
            "--k",
            2,
            "--warmup-runs",
            1,
            "--measured-runs",
            2,
            "--concurrency",
            4,
            "--assume-read-only",
            "--json-output",
            concurrent_report_path,
        )
        concurrent = json.loads(concurrent_report_path.read_text(encoding="utf-8"))
        assert concurrent["execution"]["concurrency"] == 4
        assert concurrent["metrics"]["recall_at_k"] == 1.0
        assert concurrent["metrics"]["batch_latency_ms"]["query_batch_size"] == 1
        assert concurrent["metrics"]["batch_latency_ms"]["sample_count"] == 8

        cosine_base = directory / "cosine-base.fvecs"
        cosine_queries = directory / "cosine-queries.fvecs"
        cosine_ground_truth = directory / "cosine-ground-truth.ivecs"
        cosine_index = directory / "cosine-flat.index"
        cosine_report_path = directory / "cosine-report.json"
        write_fvecs(cosine_base, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        write_fvecs(cosine_queries, [[0.8, 0.6]])

        run(
            ground_truth_tool,
            "--base",
            cosine_base,
            "--queries",
            cosine_queries,
            "--output",
            cosine_ground_truth,
            "--k",
            2,
            "--metric",
            "cosine",
        )
        run(
            build_tool,
            "--input",
            cosine_base,
            "--output",
            cosine_index,
            "--index-type",
            "flat",
            "--metric",
            "cosine",
        )
        run(
            eval_tool,
            "--index",
            cosine_index,
            "--queries",
            cosine_queries,
            "--ground-truth",
            cosine_ground_truth,
            "--metric",
            "cosine",
            "--k",
            2,
            "--warmup-runs",
            0,
            "--measured-runs",
            1,
            "--json-output",
            cosine_report_path,
        )
        cosine_report = json.loads(
            cosine_report_path.read_text(encoding="utf-8")
        )
        assert cosine_report["index"]["metric"] == "inner_product"
        assert cosine_report["workload"]["semantic_metric"] == "cosine"
        assert cosine_report["metrics"]["recall_at_k"] == 1.0


if __name__ == "__main__":
    main()
