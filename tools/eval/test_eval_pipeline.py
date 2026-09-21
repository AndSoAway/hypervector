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
        assert report["format"] == "hypervec-eval-report-v1"
        assert report["index"]["family"] == "generic"
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
